"""궤적의 모든 지점 i = 0..n 에서 학습 전 에이전트 모델에게 답 y(s_i) 를 강제로 받는다.

스텝 0..i-1 을 SWE-QA-Pro 저자 에이전트의 상태로 되돌린 뒤, 저자 러너가 최대 스텝에서 부르는 `_force_finish` 를
그대로 부른다 (강제 답변 프롬프트, 퇴화·<finish> 없음이면 원본 대화로 한 번 더, 도구 호출 거부 모두 저자 코드).
사유 문구만 FORCE_REASON 으로 준다. 지점마다 NUM_ANSWERS 개를 온도 TEMPERATURE 로 받는다.
실패 처리도 저자 러너를 따른다: <finish> 없음·도구 호출처럼 저자 러너가 문항을 다시 돌리는 실패는 같은 지점에서 최대
MAX_ATTEMPTS 번 다시 받고, 끝내 실패하면 답 없이 이유(error)만 남긴다 (judge.py 가 저자처럼 "Error: <이유>" 를 채점한다).
결과는 궤적마다 `<output-dir>/<run-id>/answers/<trajectory_id>.json` (Run), 파일이 있으면 건너뛴다.
judge.py 가 같은 파일에 점수를, label.py 가 정지 스텝을 채운다. 에이전트 모델은 미리 vLLM 으로 띄워 둔다 (README 참고).
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sweqapro.agent import _FINISH_RE, AgentState, ToolCallingAgent
from sweqapro.history import ConversationHistory
from sweqapro.tool_schemas import TOOL_SCHEMAS

from dataset import Trajectory, load_trajectories
from golden_step.models import Answer, Run, Step
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

# 지점마다 받는 답 수와 온도. 온도 0 이면 답이 모두 같아져 평균 점수가 의미 없다.
# 온도는 Qwen3 가 생각 모드에 권장하는 0.6 을 모든 모델에 같게 쓰고,
# top-p·top-k 등 나머지는 각 모델의 기본값(generation_config.json, vLLM 이 읽음)을 따른다
NUM_ANSWERS = 3
TEMPERATURE = 0.6
# 저자 에이전트가 기억하는 최근 스텝 수 (저자 configs/models.yaml 의 defaults.history_window)
HISTORY_WINDOW = 25
# 저자 러너(sweqapro/agent.py 의 retry_token_keywords)가 문항을 처음부터 다시 돌리는 실패 중 모델 출력에 관한 것.
# 궤적이 고정이라 여기서는 같은 지점에서 강제 답변만 다시 받는다. 문맥 길이 관련 단어는 뺐다 (같은 지점은 다시 해도 넘친다)
RETRY_KEYWORDS = [
    "tool_call",
    "repetitive/looping content",
    "no finish found",
    "tool mentioned but not called",
    "no <finish>",
]
MAX_ATTEMPTS = 7
# 강제 답변 프롬프트 첫 줄에 들어가는 사유. 저자 러너는 최대 스텝에서 "MAX TURNS REACHED" 를 넘기지만,
# 옛 i* 판(SWE-QA-Pro 40.39)과 같게 둔다
FORCE_REASON = "STOP AND ANSWER NOW"


class Job(BaseModel):
    trajectory: Trajectory


class AnswerRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.output_dir).expanduser() / args.run_id
        self.answer_dir = self.run_dir / "answers"
        self.answer_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(self.run_dir / "answer.log")
        tee_output(self.run_dir / "answer_stdout.log")
        save_args(args, self.run_dir / "answer_args.json")

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Job]:
        trajectories = load_trajectories(self.args.trajectories)
        if self.args.num_samples is not None:
            trajectories = trajectories[: self.args.num_samples]

        jobs = []
        for trajectory in trajectories:
            if (self.answer_dir / f"{trajectory.trajectory_id:04d}.json").exists():
                continue
            jobs.append(Job(trajectory=trajectory))
        return jobs

    def build_agent(self) -> ToolCallingAgent:
        # 요청끼리 에이전트를 나눠 쓰면 내부 상태가 섞이므로 요청마다 새로 만든다
        llm_no_tools = ChatOpenAI(
            api_key=self.args.api_key,
            base_url=self.args.base_url,
            model=self.args.model,
            temperature=TEMPERATURE,
        )
        return ToolCallingAgent(
            llm=llm_no_tools.bind_tools(TOOL_SCHEMAS, parallel_tool_calls=False),
            llm_no_tools=llm_no_tools,
            provider="openai",
            model_label=self.args.model,
            history_window=HISTORY_WINDOW,
            quiet=True,
        )

    def process_job(self, job: Job) -> None:
        trajectory = job.trajectory
        # 한 궤적의 (지점, 답) 요청을 inner_workers 개씩 같이 보낸다. 앞부분이 같은 요청끼리 vLLM 의 KV cache 를 나눠 써서,
        # KV 가 작은 모델(Qwen3-8B)도 동시 요청을 늘릴 수 있다. 1 이면 차례로 (Gemma 4 E4B 라벨은 이렇게 받았다)
        futures = {}
        with ThreadPoolExecutor(max_workers=self.args.inner_workers) as executor:
            for i in range(trajectory.n + 1):
                for idx in range(NUM_ANSWERS):
                    futures[(i, idx)] = executor.submit(force_answer, self.build_agent(), trajectory, i)
        steps = []
        for i in range(trajectory.n + 1):
            answers = []
            for idx in range(NUM_ANSWERS):
                answer = futures[(i, idx)].result()
                if answer.answer is None:
                    logger.warning(
                        "Trajectory %d step %d answer %d: %s", trajectory.trajectory_id, i, idx, answer.error
                    )
                answers.append(answer)
            steps.append(Step(step=i, answers=answers))

        run = Run(trajectory_id=trajectory.trajectory_id, steps=steps)
        out_path = self.answer_dir / f"{trajectory.trajectory_id:04d}.json"
        out_path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def force_answer(agent: ToolCallingAgent, trajectory: Trajectory, i: int) -> Answer:
    """지점 i 에서 강제 답변 하나. 저자 러너가 다시 돌리는 실패면 최대 MAX_ATTEMPTS 번 다시 받는다.

    저자 코드의 실패 판정(퇴화·<finish> 없음·도구 호출)과 문맥 한도를 넘은 요청은 답 없이 이유만 남긴다.
    연결 오류 같은 나머지는 궤적 작업 전체를 실패시켜, 다시 돌릴 때 이 궤적을 다시 한다.
    """
    error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            state = agent._force_finish(build_state(trajectory, i), FORCE_REASON)
        except (RuntimeError, openai.BadRequestError) as e:
            error = str(e)[:200]
            if not any(keyword in error.lower() for keyword in RETRY_KEYWORDS):
                break
            continue
        final_answer = state["final_answer"]
        if final_answer:
            # 저자 코드는 모델 출력 전체를 기록 끝에 한 턴으로 덧붙여 두고, final_answer 는 그 안의 첫 <finish> 블록이다
            output = state["history_manager"].history[-1][0].content
            reasoning = output[: _FINISH_RE.search(output).start()].strip()
            return Answer(reasoning=reasoning, answer=final_answer)
        error = "empty final answer"
    return Answer(answer=None, error=error)


def build_state(trajectory: Trajectory, i: int) -> AgentState:
    """스텝 0..i-1 까지 본 저자 에이전트 상태. prompt (system·user)는 학습 행과 같게 궤적의 원본 유지."""
    history = ConversationHistory(HISTORY_WINDOW)
    for step_idx, step in enumerate(trajectory.steps[:i]):
        call_ids = [f"call_{step_idx}_{call_idx}" for call_idx in range(len(step.tool_calls))]
        assistant = AIMessage(
            content=step.response,
            tool_calls=[
                {"name": call.tool, "args": call.args, "id": call_id}
                for call, call_id in zip(step.tool_calls, call_ids)
            ],
        )
        tools = [
            ToolMessage(content=call.result, tool_call_id=call_id) for call, call_id in zip(step.tool_calls, call_ids)
        ]
        history.add_interaction([assistant] + tools)
    preamble = [
        SystemMessage(content=trajectory.system_prompt),
        HumanMessage(content=trajectory.user_prompt),
    ]
    return AgentState(
        question=trajectory.question,
        repo_path=trajectory.repo_path,
        current_step=i,
        final_answer="",
        tool_calls=[],
        history_manager=history,
        trajectory=[],
        total_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        preamble=preamble,
        tool_stats={"counts": {}, "records": []},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force the agent model to answer at every step of every trajectory",
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
        help="run id (e.g. qwen3-8b)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8091/v1",
        help="vLLM server of the agent model before fine-tuning",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="api key",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="served model name at --base-url",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="max workers (trajectories answered at once)",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=1,
        help="requests sent at once within a trajectory (steps x answers); raise it for models with a small KV cache",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default="sft",
        help="trajectories: sft, or a results dir of evaluate-agent/run_agent.py (the agent model's own rollouts)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="if set, only answer for the first N trajectories (for a quick sample run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = AnswerRunner(args)
    runner.run(runner.create_jobs())


if __name__ == "__main__":
    main()
