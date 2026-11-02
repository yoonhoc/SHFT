"""Claude Message Batches API 로 run_agent.py 의 답을 다시 채점하는 공통 부분 (token 값 절반). 벤치마크마다 prompt·parser 만 다르다.

벤치마크별 실행은 sweqapro.py, sweqa.py. 두 단계로 돌린다.
- submit: 아직 점수가 없는 문항을 batch 하나로 보내고 batch id 를 적어 둔다.
- collect: batch 가 끝났으면 점수를 쓰고 쓴 token 수를 찍은 뒤 batch id 를 지운다. 실패한 문항은 비워 두므로 submit 을 다시 하면 그 문항만 간다.
Luna 점수(results/)는 두고 results-<judge-model>/ 에 따로 쓴다. effort 는 모델 기본.
주소는 LITELLM_ANTHROPIC_BASE_URL (LiteLLM 의 Anthropic pass-through, 예: http://<proxy>/anthropic), 키는 LITELLM_API_KEY 에서 읽는다.
Claude Code 가 읽는 ANTHROPIC_API_KEY·ANTHROPIC_BASE_URL 과 겹치지 않게 LiteLLM 이름을 쓴다.

읽는 것: <output-dir>/<run-id>/<benchmark>/results/*.json
쓰는 것: <output-dir>/<run-id>/<benchmark>/results-<judge-model>/*.json, judge-<judge-model>_batch.txt, judge-<judge-model>.log
"""

import argparse
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import anthropic
import httpx
from anthropic.types.messages import MessageBatchIndividualResponse

from score.judge import candidate_of
from models import Result
from utils import save_args, setup_logging

logger = logging.getLogger(__name__)

# thinking 도 이 안에 들어간다. 쓴 token 만 값을 내므로 넉넉히 둔다
MAX_TOKENS = 16000


class BatchJudge(ABC):
    system_prompt: str | None = None

    def __init__(self, args: argparse.Namespace):
        self.args = args
        base_url = os.environ.get("LITELLM_ANTHROPIC_BASE_URL")
        api_key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not api_key:
            raise ValueError("Set LITELLM_ANTHROPIC_BASE_URL and LITELLM_API_KEY")
        self.client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
        run_dir = Path(args.output_dir).expanduser() / args.run_id / args.benchmark
        tag = args.judge_model.replace("/", "-")  # proxy 모델 이름(anthropic/…)은 경로에 못 쓴다
        self.result_dir = run_dir / "results"
        self.score_dir = run_dir / f"results-{tag}"
        self.score_dir.mkdir(parents=True, exist_ok=True)
        # 보내고 아직 거둬들이지 않은 batch 의 id. 있으면 submit 하지 않는다 (같은 문항을 두 번 내지 않게)
        self.batch_path = run_dir / f"judge-{tag}_batch.txt"

        setup_logging(run_dir / f"judge-{tag}.log")
        logging.getLogger().addHandler(logging.StreamHandler())  # 상태를 화면에도
        save_args(args, run_dir / f"judge-{tag}_args.json")

    @abstractmethod
    def make_messages(self, result: Result, candidate: str) -> list[dict]: ...

    @abstractmethod
    def read_scores(self, text: str) -> dict[str, int] | None: ...

    def submit(self) -> None:
        if self.batch_path.exists():
            raise ValueError(f"Collect the submitted batch first: {self.batch_path}")
        requests = []
        for path in sorted(self.result_dir.glob("*.json")):
            if (self.score_dir / path.name).exists():
                continue
            result = Result.model_validate_json(path.read_text())
            candidate = candidate_of(result)
            if candidate is None:
                continue
            params = {"model": self.args.judge_model, "max_tokens": MAX_TOKENS}
            params["messages"] = self.make_messages(result, candidate)
            if self.system_prompt is not None:
                params["system"] = self.system_prompt
            requests.append({"custom_id": path.stem, "params": params})
        if not requests:
            logger.info("Nothing to score")
            return
        batch = self.client.messages.batches.create(requests=requests)
        self.batch_path.write_text(batch.id)
        logger.info("Submitted %d answers: %s", len(requests), batch.id)

    def collect(self) -> None:
        batch = self.client.messages.batches.retrieve(self.batch_path.read_text())
        if batch.processing_status != "ended":
            logger.info("Not ended: %s", batch.request_counts)
            return
        input_tokens = 0
        output_tokens = 0
        for entry in self.fetch_results(batch.id):
            if entry.result.type != "succeeded":
                logger.warning("Question %s: %s", entry.custom_id, entry.result.type)
                continue
            message = entry.result.message
            input_tokens += message.usage.input_tokens
            output_tokens += message.usage.output_tokens
            texts = [block.text for block in message.content if block.type == "text"]
            score = self.read_scores("".join(texts).strip())
            if score is None:
                logger.warning("Question %s: no score (stop reason %s)", entry.custom_id, message.stop_reason)
                continue
            result = Result.model_validate_json((self.result_dir / f"{entry.custom_id}.json").read_text())
            result.score = score
            (self.score_dir / f"{entry.custom_id}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        self.batch_path.unlink()
        logger.info("Collected %s: %s", batch.id, batch.request_counts)
        logger.info("Tokens: input %d, output %d", input_tokens, output_tokens)


    def fetch_results(self, batch_id: str) -> list[MessageBatchIndividualResponse]:
        # SDK 의 batches.results() 는 Anthropic 이 준 results_url(https://api.anthropic.com/…)로 바로 가서
        # proxy 키가 거절된다(401). 같은 경로를 proxy 주소(base_url) 밑으로 부른다
        response = self.client.get(
            f"/v1/messages/batches/{batch_id}/results",
            cast_to=httpx.Response,
            options={"headers": {"Accept": "application/binary"}},
        )
        entries = []
        for line in response.text.splitlines():
            if line.strip():
                entries.append(MessageBatchIndividualResponse.model_validate_json(line))
        return entries


def run(judge: BatchJudge) -> None:
    if judge.args.action == "submit":
        judge.submit()
    else:
        judge.collect()


def parse_args(benchmarks: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the answers of run_agent.py again with Claude through the Message Batches API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["submit", "collect"],
        help="submit a batch of the unscored answers, or collect the scores of the submitted batch",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "out"),  # 채점은 run_agent.py 의 실행분 안에 쓴다
        help="output dir of the runner (budget_tracker/out, autorater/out for the baselines)",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="run id of run_agent.py",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=str,
        choices=benchmarks,
        default=benchmarks[0],
        help="benchmark",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="claude-sonnet-5-5",
        help="Claude model id (the pass-through sends it to Anthropic as is)",
    )
    return parser.parse_args()
