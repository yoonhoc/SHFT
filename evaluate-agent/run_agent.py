"""SWE-QA-Pro·SWE-QA 문항마다 에이전트를 저자 러너와 같게 돌려 답과 궤적을 저장한다.

에이전트(`ToolCallingAgent`: 저자 프롬프트·도구·도구 호출 상한·강제 답변), 모델 설정 읽기(`registry`),
vLLM 서버(`VLLMServer`)는 SWE-QA-Pro 저자 코드를 그대로 쓴다. 저자 러너(`scripts/run_agent.py`)와 다른 점:
- 모델 설정은 이 폴더의 models.yaml 에서 읽는다. 학습한 모델은 --adapter 로 어댑터를 서버에 얹거나 (Qwen3),
  --model-path 로 합친 가중치 폴더를 준다 (Gemma 4: vLLM 0.20.2 가 Gemma 4 어댑터를 적용하지 않는다).
- 저자 러너는 궤적을 버리고, 저자 궤적은 도구를 부른 턴만 담는다. 여기서는 궤적을 남기고, 도구 없이 끝낸 턴
  (정지 턴·강제 답변 턴)의 원문과 토큰 사용량도 더한다 (정지 서술 분석).
- 문항마다 JSON 하나로 저장하고 파일이 있으면 건너뛴다 (저자는 jsonl 에 이어 쓰고 질문 글로 건너뛴다).
- vLLM 서버에 보내는 동시 요청 수가 저자는 최대 4, 여기는 --max-workers.
저장소 경로는 저자 러너와 같게 준다: 저장소 폴더의 부모로 작업 폴더를 옮기고 "repos/<이름>" 처럼 상대 경로로.

읽는 것: dataset.load_items(--benchmark), --repo-root 밑의 저장소
쓰는 것: <output-dir>/<run-id>/<benchmark>/results/<index>.json (`Result`), run_agent.log, run_agent_args.json,
        run_agent_spec.json (실제로 쓴 모델 설정), run_agent_stdout.log (진행 상황), serve_agent.log (vLLM 서버 출력)
        requests/<index>.jsonl 에 LLM 호출마다 보낸 messages 와 받은 응답 원문 (한 줄 = 호출 하나; --no-record 로 끈다)
vLLM 서버는 이 스크립트가 CUDA_VISIBLE_DEVICES 의 GPU 에 띄우고, 끝나면 내린다.
"""

import argparse
import contextlib
import dataclasses
import json
import logging
import os
import threading
from pathlib import Path

from sweqapro import registry
from sweqapro.agent import AgentState, ToolCallingAgent
from sweqapro.vllm_server import VLLMServer

import tool_sandbox
from dataset import BENCHMARKS, REPO_ROOTS, ROLLOUT_SETS, VALIDATION_SET, Item, load_items
from models import Result
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

# --adapter 로 얹은 어댑터의 이름. 서버는 원본 model_id 로 뜨고, 에이전트는 이 이름으로 요청한다
ADAPTER_NAME = "adapter"


def record_response_turn(state: AgentState, step: int, num_turns: int) -> None:
    """저자 궤적이 빼는 도구 없는 턴(정지 턴·강제 답변 턴)을 궤적에 더한다. 에이전트의 _agent_step 끝에서 부른다.

    저자 결과에는 마지막 답 턴의 <finish> 안쪽만 남으므로, 대화 기록에 들어간 원문과 그 호출의 토큰 사용량을 옮겨 둔다.
    num_turns 는 스텝 전의 궤적 길이. 스텝이 도구를 불렀으면 저자가 이미 궤적에 더했으므로 아무것도 하지 않는다.
    """
    if len(state["trajectory"]) != num_turns:
        return
    message = state["history_manager"].history[-1][0]
    token_usage = {
        "prompt_tokens": state["prompt_tokens"],
        "completion_tokens": state["completion_tokens"],
        "total_tokens": state["total_tokens"],
    }
    turn = {"step": step, "response": message.content, "tool_calls": [], "token_usage": token_usage}
    state["trajectory"].append(turn)


class SandboxedTools:
    """도구 호출을 그 문항의 저장소만 보이는 샌드박스에서 실행한다 (tool_sandbox.py). 저자 _run_tool 자리에 끼운다."""

    @staticmethod
    def _run_tool(call: dict, repo_path: str) -> str:
        return tool_sandbox.run_tool(call, repo_path)


class Agent(SandboxedTools, ToolCallingAgent):
    """저자 에이전트 그대로, 도구는 샌드박스에서 돌리고 도구 없는 턴만 궤적에 더한다 (record_response_turn)."""

    def _agent_step(self, state: AgentState) -> AgentState:
        step = state.get("current_step", 0)
        num_turns = len(state["trajectory"])
        state = super()._agent_step(state)
        record_response_turn(state, step, num_turns)
        return state


class Server(VLLMServer):
    """저자 VLLMServer 그대로, vLLM 출력만 log_path 로 보낸다 (run_agent_stdout.log 의 진행 막대와 섞이지 않게)."""

    def __init__(self, log_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = open(log_path, "a", encoding="utf-8")

    def __enter__(self) -> "Server":
        # 서버 프로세스는 띄울 때의 sys.stdout·sys.stderr 를 이어받는다
        with contextlib.redirect_stdout(self.log_file), contextlib.redirect_stderr(self.log_file):
            super().__enter__()
        print(f"[run_agent] vLLM server ready at {self.base_url}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        with contextlib.redirect_stdout(self.log_file), contextlib.redirect_stderr(self.log_file):
            super().__exit__(exc_type, exc, tb)
        self.log_file.close()


class RecordingLLM:
    """LLM 을 감싸 호출마다 보낸 messages 와 받은 응답 원문을 문항별 jsonl 에 남긴다 (기본, --no-record 로 끈다).

    저자 에이전트는 llm.invoke(messages) 만 부르므로 invoke 만 감싸고 나머지는 그대로 넘긴다. 어느 문항의 호출인지는
    process_job 이 스레드마다 둔 index 로 안다. 파일: <run_dir>/requests/<index>.jsonl, 한 줄 = 호출 하나.
    """

    current = threading.local()

    def __init__(self, llm, record_dir: Path):
        self.llm = llm
        self.record_dir = record_dir
        self.record_dir.mkdir(parents=True, exist_ok=True)

    def invoke(self, messages, **kwargs):
        response = self.llm.invoke(messages, **kwargs)
        index = getattr(self.current, "index", None)
        record = {
            "messages": [message.model_dump() for message in messages],
            "kwargs": kwargs,
            "content": response.content,
            "tool_calls": getattr(response, "tool_calls", None),
            "response_metadata": getattr(response, "response_metadata", None),
            "usage_metadata": getattr(response, "usage_metadata", None),
        }
        name = f"{index:03d}.jsonl" if index is not None else "unknown.jsonl"
        with (self.record_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return response

    def __getattr__(self, name):
        return getattr(self.llm, name)


class AgentRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.output_dir).expanduser().resolve() / args.run_id / args.benchmark
        self.result_dir = self.run_dir / "results"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(self.run_dir / "run_agent.log")
        tee_output(self.run_dir / "run_agent_stdout.log")
        save_args(args, self.run_dir / "run_agent_args.json")

        self.repo_root = Path(args.repo_root or REPO_ROOTS[args.benchmark]).expanduser().resolve()
        self.spec = load_spec(args)
        (self.run_dir / "run_agent_spec.json").write_text(json.dumps(dataclasses.asdict(self.spec), indent=2))
        self.agent = None
        # 저자 러너처럼 저장소 폴더의 부모(eval/)에서 돌며 에이전트에게 "repos/<이름>" 을 준다.
        # 이 경로는 system prompt·첫 질문·강제 답변 prompt·도구 결과에 그대로 들어간다
        os.chdir(self.repo_root.parent)

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Item]:
        items = load_items(self.args.benchmark)
        if self.args.num_samples is not None:
            items = items[: self.args.num_samples]

        jobs = []
        missing = set()
        for item in items:
            if (self.result_dir / f"{item.index:03d}.json").exists():
                continue
            if not self.repo_path(item).exists():
                missing.add(item.repo)
            jobs.append(item)
        # GPU 에 서버를 띄우기 전에 멈춘다
        if missing:
            raise ValueError(f"Repos not found under {self.repo_root}: {sorted(missing)}")
        return jobs

    def repo_path(self, item: Item) -> Path:
        # 저자 러너 _resolve_repo_path 와 같은 규칙 ("<owner>/<name>" 이면 name), 저자처럼 상대 경로로
        return Path(self.repo_root.name) / item.repo.split("/")[-1]

    def build_llms(self, base_url: str) -> None:
        """저자 러너와 같은 LLM 둘 (도구 있는 것·없는 것). --adapter 를 얹었으면 그 이름으로 요청한다."""
        self.llm_spec = self.spec
        if self.args.adapter is not None:
            self.llm_spec = dataclasses.replace(self.spec, model_id=ADAPTER_NAME)
        self.llm = registry.build_llm(self.llm_spec, base_url=base_url, with_tools=True)
        self.llm_no_tools = registry.build_llm(self.llm_spec, base_url=base_url, with_tools=False)
        if self.args.record:
            self.llm = RecordingLLM(self.llm, self.run_dir / "requests")
            self.llm_no_tools = RecordingLLM(self.llm_no_tools, self.run_dir / "requests")

    def agent_kwargs(self) -> dict:
        """저자 러너 run() 과 같은 에이전트 인자. 저자 에이전트의 단계별 화면 출력은 끈다."""
        return {
            "llm": self.llm,
            "llm_no_tools": self.llm_no_tools,
            "provider": "openai",
            "model_label": self.args.model,
            "max_iterations": self.spec.max_iterations,
            "history_window": self.spec.history_window,
            "max_context_length": self.spec.max_context_length,
            "context_warning_threshold": self.spec.context_warning_threshold,
            "quiet": True,
        }

    def build_agent(self, base_url: str) -> None:
        self.build_llms(base_url)
        self.agent = Agent(**self.agent_kwargs())

    def build_result(self, item: Item, output: dict, **extra) -> Result:
        """저자 query() 의 결과를 Result 로. 기준선·oracle 이 더하는 항목은 extra 로 받는다."""
        result = Result(
            item=item,
            answer=output["answer"],
            status=output["status"],
            stop_reason=output.get("stop_reason"),
            steps_completed=output.get("steps_completed", 0),
            max_iterations=self.spec.max_iterations,
            retry_attempts=output.get("retry_attempts"),
            total_time=output.get("total_time"),
            error=output.get("error"),
            trajectory=output.get("trajectory", []),
            **extra,
        )
        if result.status != "success":
            logger.warning("Question %d: %s", item.index, result.answer[:200])
        return result

    def process_job(self, item: Item) -> None:
        RecordingLLM.current.index = item.index
        output = self.agent.query(question=item.question, repo_path=str(self.repo_path(item)))
        result = self.build_result(item, output)
        (self.result_dir / f"{item.index:03d}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_spec(args: argparse.Namespace) -> registry.ModelSpec:
    # 저자 registry 는 저자 configs/models.yaml 을 읽으므로 이 폴더의 설정 파일로 바꿔 읽는다
    registry.MODELS_YAML = Path(args.config).expanduser()
    spec = registry.resolve(args.model)
    if spec.provider != "vllm-local":
        raise ValueError(f"Only vllm-local models are supported, got {spec.provider}")
    if args.model_path is not None and args.adapter is not None:
        raise ValueError("Give either --model-path or --adapter, not both")
    if args.model_path is not None:
        spec = dataclasses.replace(spec, model_id=str(Path(args.model_path).expanduser().resolve()))
    # models.yaml 의 tokenizer 폴더는 이 폴더 기준 상대 경로다. vLLM 은 상대 경로를 HF 저장소 id 로 읽으므로 절대 경로로 바꾼다
    if spec.vllm and "tokenizer" in spec.vllm:
        vllm = dict(spec.vllm)
        vllm["tokenizer"] = str((Path(__file__).resolve().parent / vllm["tokenizer"]).resolve())
        spec = dataclasses.replace(spec, vllm=vllm)
    if args.adapter is not None:
        spec = add_adapter(spec, Path(args.adapter).expanduser().resolve())
    if args.max_iterations is not None:
        spec = dataclasses.replace(spec, max_iterations=args.max_iterations)
    return spec


def add_adapter(spec: registry.ModelSpec, adapter_dir: Path) -> registry.ModelSpec:
    """저자 vllm-local 경로 그대로 vllm 블록에 어댑터를 더한다 (vllm serve --enable-lora --lora-modules)."""
    if spec.extra.get("adapter") is False:
        raise ValueError(f"{spec.name} does not take an adapter in vLLM; serve merged weights with --model-path")
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    vllm = dict(spec.vllm or {})
    vllm["enable_lora"] = True
    vllm["max_lora_rank"] = adapter_config["r"]
    vllm["lora_modules"] = f"{ADAPTER_NAME}={adapter_dir}"
    return dataclasses.replace(spec, vllm=vllm)


def build_parser() -> argparse.ArgumentParser:
    """공통 인자. 다른 실행기(sufficiency 등)는 이 파서에 인자를 더해 쓴다."""
    parser = argparse.ArgumentParser(
        description="Run the SWE-QA-Pro agent on a benchmark and save each answer with its trajectory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="output dir",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="run id (one per evaluated model)",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=BENCHMARKS + ROLLOUT_SETS + [VALIDATION_SET],
        required=True,
        help="benchmark",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="model key in --config",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="merged checkpoint to serve instead of the model_id of --model (settings stay those of --model)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="LoRA adapter checkpoint to serve on the model_id of --model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "models.yaml"),
        help="model settings (same format as the author's configs/models.yaml)",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="dir with the benchmark repositories; default: where the author's clone_repos.sh puts them",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="tool-call limit; default: max_iterations of --model",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="max workers (questions sent to the vLLM server at once)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="if set, only run the first N questions (for a quick sample run)",
    )
    parser.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="record every LLM request (messages as sent) and response under <run-id>/<benchmark>/requests/<index>.jsonl",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def run_jobs(runner: AgentRunner, tag: str) -> None:
    """남은 문항이 있으면 vLLM 서버를 띄우고 돈다. 각 실행기(run_agent, budget_tracker, autorater)의 main() 이 부른다."""
    jobs = runner.create_jobs()
    print(f"[{tag}] {len(jobs)} questions to run -> {runner.result_dir}", flush=True)
    if not jobs:
        return
    server = Server(
        runner.run_dir / "serve_agent.log",
        runner.spec.model_id,
        with_tools=True,
        vllm=runner.spec.vllm,
        agent_vllm=runner.spec.agent_vllm,
    )
    with server:
        runner.build_agent(server.base_url)
        runner.run(jobs)


def main() -> None:
    run_jobs(AgentRunner(parse_args()), "run_agent")


if __name__ == "__main__":
    main()
