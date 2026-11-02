"""run_agent.py 의 SWE-QA-Pro 답을 GPT-6 Sol 로 다시 채점한다 (validation 세트도 같은 채점기로).

저자 채점기의 `_score_one` 을 judge.py 와 같은 Responses API 경로로 그대로 부른다 (저자 채점 prompt, 5 항목 × 1~10 = 5~50).
judge.py 처럼 Flex 처리, 요청당 제한 시간 15분, 답마다 최대 10번 다시 보낸다 (Flex 는 자원이 모자라면 429 를 돌려주고 과금하지 않는다).
Luna 점수(results/)는 두고 results-<judge-model>/ 에 따로 쓴다. 이미 점수가 있는 문항은 건너뛰므로 빈 문항이 남으면 한 번 더 돌린다.
주소는 LITELLM_BASE_URL (OpenAI 형식, 예: http://<proxy>/v1), 키는 LITELLM_API_KEY 에서 읽는다.

읽는 것: <output-dir>/<run-id>/<benchmark>/results/*.json
쓰는 것: <output-dir>/<run-id>/<benchmark>/results-<judge-model>/*.json, judge-<judge-model>.log

    python -m score.gpt_judge.sweqapro -r <run-id>
    python -m score.gpt_judge.sweqapro -o budget_tracker/out -r <run-id>
"""

import argparse
import functools
import logging
import os
import time
from pathlib import Path

from openai import OpenAI
from sweqapro.scoring.judge import _load_prompt_template, _score_one

from dataset import VALIDATION_SET
from score.judge import MAX_ATTEMPTS, SERVICE_TIER, TIMEOUT, candidate_of
from models import Result
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

JUDGE_MODEL = "gpt-6-sol"
# judge.py 의 Luna 와 같은 강도
REASONING_EFFORT = "medium"


class SWEQAProJudgeRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        base_url = os.environ.get("LITELLM_BASE_URL")
        api_key = os.environ.get("LITELLM_API_KEY")
        if not base_url or not api_key:
            raise ValueError("Set LITELLM_BASE_URL and LITELLM_API_KEY")
        run_dir = Path(args.output_dir).expanduser() / args.run_id / args.benchmark
        self.result_dir = run_dir / "results"
        if not self.result_dir.exists():
            raise ValueError(f"No results: {self.result_dir}")  # -o 를 빠뜨리면 여기서 멈춘다
        self.score_dir = run_dir / f"results-{JUDGE_MODEL}"
        self.score_dir.mkdir(exist_ok=True)

        setup_logging(run_dir / f"judge-{JUDGE_MODEL}.log")
        tee_output(run_dir / f"judge-{JUDGE_MODEL}_stdout.log")
        save_args(args, run_dir / f"judge-{JUDGE_MODEL}_args.json")

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT)
        self.client.responses.create = functools.partial(self.client.responses.create, service_tier=SERVICE_TIER)
        self.template = _load_prompt_template()

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Result]:
        jobs = []
        for path in sorted(self.result_dir.glob("*.json")):
            if (self.score_dir / path.name).exists():
                continue
            result = Result.model_validate_json(path.read_text())
            if self.args.num_samples is not None and result.item.index >= self.args.num_samples:
                continue
            if candidate_of(result) is None:
                continue
            jobs.append(result)
        return jobs

    def process_job(self, result: Result) -> None:
        candidate = candidate_of(result)
        score = None
        for attempt in range(MAX_ATTEMPTS):
            score = _score_one(
                "openai",
                self.client,
                JUDGE_MODEL,
                self.template,
                result.item.question,
                result.item.answer,
                candidate,
                reasoning_effort=REASONING_EFFORT,
            )
            if score is not None:
                break
            time.sleep(min(60, 2**attempt))
        if score is None:
            logger.warning("Question %d: no score after %d attempts", result.item.index, MAX_ATTEMPTS)
            return
        result.score = score
        out_path = self.score_dir / f"{result.item.index:03d}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the SWE-QA-Pro answers of run_agent.py again with GPT-6 Sol (Flex processing)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        choices=["sweqapro", VALIDATION_SET],
        default="sweqapro",
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
    runner = SWEQAProJudgeRunner(args)
    runner.run(runner.create_jobs())


if __name__ == "__main__":
    main()
