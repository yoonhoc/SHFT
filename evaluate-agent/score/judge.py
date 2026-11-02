"""run_agent.py 가 받은 답을 각 벤치마크 저자의 채점기로 채점해 같은 파일에 점수를 채운다.

- SWE-QA-Pro: SWE-QA-Pro 저자 채점기의 `_score_one` 그대로 (저자 채점 프롬프트, 5 항목 × 1~10 = 5~50).
  저자가 GPT-5 로 채점할 때 쓰는 Responses API 경로로 부른다.
- SWE-QA: SWE-QA 저자 채점기(`Benchmark construction/score/llm-as-a-judge.py`)의 `score_answer` 그대로
  (저자 채점 프롬프트, 5 항목 × 1~20 = 5~100). 저자 경로(Chat Completions)로 부르되, 저자 채점 모델(GPT-5, Azure)
  전용 인자(extra_body 의 생각 예산)는 빼고 추론 강도를 넣는다.
채점 모델은 둘 다 GPT-6 Luna, 추론 강도 medium, Flex 처리 (build-dataset/golden_step/judge.py 와 같다).
두 저자 채점기 모두 빈 답과 "No answer found" 는 채점하지 않는다. 끝내 실패한 문항의 "Error: <이유>" 는 다른 답과 같이 채점한다.
이미 점수가 있는 문항은 건너뛰므로, run_agent.py 가 도는 중에 돌렸거나 채점 실패로 빈 문항이 남았으면 한 번 더 돌린다.
API 키는 환경 변수 OPENAI_API_KEY 에서 읽는다.
"""

import argparse
import functools
import logging
import runpy
import time
from pathlib import Path

from openai import OpenAI
from sweqapro.scoring.judge import _load_prompt_template, _make_client, _score_one

from dataset import BENCHMARKS, THIRD_PARTY_DIR, VALIDATION_SET, Item
from models import Result
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
SERVICE_TIER = "flex"
# Flex 는 느려서 문서 권장대로 요청당 제한 시간 15분
TIMEOUT = 900
# 두 저자 채점기 모두 API 오류(Flex 자원 부족 429 포함)와 점수 읽기 실패를 None 으로 돌려준다
MAX_ATTEMPTS = 10
# 파일 이름에 '-' 가 있어 import 로는 못 부른다
SWEQA_JUDGE_PATH = THIRD_PARTY_DIR / "SWE-QA-Bench" / "Benchmark construction" / "score" / "llm-as-a-judge.py"


class JudgeRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.output_dir).expanduser() / args.run_id / args.benchmark
        self.result_dir = self.run_dir / "results"

        setup_logging(self.run_dir / "judge.log")
        tee_output(self.run_dir / "judge_stdout.log")
        save_args(args, self.run_dir / "judge_args.json")

        # validation 문항(RL, ground truth 있음)은 SWE-QA-Pro 채점기로 (같은 저자, 같은 문항 형식)
        if args.benchmark in ("sweqapro", VALIDATION_SET):
            self.client = _make_client("openai").with_options(timeout=TIMEOUT)
            self.client.responses.create = functools.partial(self.client.responses.create, service_tier=SERVICE_TIER)
            self.template = _load_prompt_template()
        else:
            self.client = make_sweqa_client()
            self.sweqa_score_answer = runpy.run_path(str(SWEQA_JUDGE_PATH))["score_answer"]

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Result]:
        jobs = []
        for path in sorted(self.result_dir.glob("*.json")):
            result = Result.model_validate_json(path.read_text())
            if self.args.num_samples is not None and result.item.index >= self.args.num_samples:
                continue
            if result.score is not None or candidate_of(result) is None:
                continue
            jobs.append(result)
        return jobs

    def process_job(self, result: Result) -> None:
        candidate = candidate_of(result)
        for attempt in range(MAX_ATTEMPTS):
            result.score = self.score_answer(result.item, candidate)
            if result.score is not None:
                break
            time.sleep(min(60, 2**attempt))
        if result.score is None:
            logger.warning("Question %d: no score after %d attempts", result.item.index, MAX_ATTEMPTS)
            return
        out_path = self.result_dir / f"{result.item.index:03d}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def score_answer(self, item: Item, candidate: str) -> dict[str, int] | None:
        if self.args.benchmark in ("sweqapro", VALIDATION_SET):
            return _score_one(
                "openai",
                self.client,
                JUDGE_MODEL,
                self.template,
                item.question,
                item.answer,
                candidate,
                reasoning_effort=REASONING_EFFORT,
            )
        return self.sweqa_score_answer(item.question, item.answer, candidate, self.client, JUDGE_MODEL)


def candidate_of(result: Result) -> str | None:
    """채점할 답. 두 저자 채점기처럼 빈 답과 "No answer found" 는 None."""
    candidate = result.answer.strip()
    if not candidate or candidate == "No answer found":
        return None
    return candidate


def make_sweqa_client() -> OpenAI:
    """SWE-QA 채점기가 부르는 client.chat.completions.create(model, messages, extra_body) 를 GPT-6 Luna 용으로 바꾼다.

    extra_body 는 저자 채점 모델(GPT-5, Azure) 전용 생각 예산이라 빼고, 추론 강도와 Flex 처리를 넣는다.
    """
    client = OpenAI(timeout=TIMEOUT)
    create = client.chat.completions.create

    def create_for_judge(model: str, messages: list[dict], extra_body: dict | None = None):
        return create(model=model, messages=messages, reasoning_effort=REASONING_EFFORT, service_tier=SERVICE_TIER)

    client.chat.completions.create = create_for_judge
    return client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the answers of run_agent.py with the judge of each benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "out"),  # 채점은 run_agent.py 의 실행분 안에 쓴다
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
        choices=BENCHMARKS + [VALIDATION_SET],
        required=True,
        help="benchmark",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="max workers (answers scored at once)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="if set, only score the first N questions (for a quick sample run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = JudgeRunner(args)
    runner.run(runner.create_jobs())


if __name__ == "__main__":
    main()
