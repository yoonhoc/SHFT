"""answer.py 가 받은 강제 답변을 채점해 같은 파일에 점수를 채운다.

SFT 궤적에는 벤치마크 정답이 없으므로, 기준 답은 그 궤적에서 Claude 가 쓴 최종 답(<finish> 안쪽)이다.
자기 롤아웃(--trajectories <results dir>)은 문항의 ground truth 답(Trajectory.reference)이 기준 답이다.
채점은 SWE-QA-Pro 저자 채점기의 `_score_one` 을 그대로 부른다 (저자 채점 프롬프트, 5 항목 × 1~10).
채점 모델은 GPT-6 Luna 이고, 저자가 GPT-5 로 채점할 때 쓰는 "openai" 경로(Responses API, 추론 강도 지정)로 부른다.
요청은 Flex 처리(Standard 의 반값, 대신 느리고 자원이 모자라면 429)로 보낸다. 저자 `_score_one` 은 service_tier 를
넘기지 않으므로, 넘겨주는 클라이언트의 `responses.create` 에 미리 채워 둔다. 채점 프롬프트와 요청 본문은 저자와 같다.
답하지 못한 강제 답변은 저자 러너처럼 "Error: <이유>" 를 답으로 채점한다 (저자 채점기는 이 글도 다른 답과 같이 채점한다).
문맥 한도를 넘어 받지 못한 답은 채점하지 않는다. 이미 점수가 있는 답은 건너뛰므로, 다시 돌리면 아직 점수가 없는 답만 채점한다.
API 키는 저자 코드처럼 환경 변수 OPENAI_API_KEY 에서 읽는다.
"""

import argparse
import functools
import logging
import time
from pathlib import Path

from pydantic import BaseModel
from sweqapro.scoring.judge import _load_prompt_template, _make_client, _score_one

from dataset import Trajectory, load_trajectories
from golden_step.models import Run
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

# 저자 기본값은 gpt-5 · low. low 는 채점을 너무 얕게 할 수 있어 medium
JUDGE_MODEL = "gpt-6-luna"
REASONING_EFFORT = "medium"
SERVICE_TIER = "flex"
# Flex 는 느려서 문서 권장대로 요청당 제한 시간 15분
TIMEOUT = 900
# 저자 _score_one 은 API 오류(Flex 자원 부족 429 포함)와 점수 읽기 실패를 모두 None 으로 돌려준다
MAX_ATTEMPTS = 10
# vLLM 이 문맥 한도를 넘은 요청을 거절할 때의 문구. 모델 탓이 아니라 채점하지 않는다
CONTEXT_OVERFLOW_MARKER = "maximum context length"


class Job(BaseModel):
    trajectory: Trajectory


class JudgeRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.output_dir).expanduser() / args.run_id
        self.answer_dir = self.run_dir / "answers"

        setup_logging(self.run_dir / "judge.log")
        tee_output(self.run_dir / "judge_stdout.log")
        save_args(args, self.run_dir / "judge_args.json")

        self.client = _make_client("openai").with_options(timeout=TIMEOUT)
        self.client.responses.create = functools.partial(self.client.responses.create, service_tier=SERVICE_TIER)
        self.template = _load_prompt_template()

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Job]:
        trajectories = load_trajectories(self.args.trajectories)
        if self.args.num_samples is not None:
            trajectories = trajectories[: self.args.num_samples]

        jobs = []
        for trajectory in trajectories:
            # answer.py 가 아직 답을 받지 않은 궤적은 다음에 다시 돌릴 때 채점한다
            if (self.answer_dir / f"{trajectory.trajectory_id:04d}.json").exists():
                jobs.append(Job(trajectory=trajectory))
        return jobs

    def process_job(self, job: Job) -> None:
        path = self.answer_dir / f"{job.trajectory.trajectory_id:04d}.json"
        run = Run.model_validate_json(path.read_text())
        for step in run.steps:
            for answer in step.answers:
                if answer.score is not None:
                    continue
                if answer.answer is not None:
                    candidate = answer.answer
                elif answer.error is not None and CONTEXT_OVERFLOW_MARKER not in answer.error:
                    # 저자 러너가 실패한 문항에 남기는 답과 같은 모양
                    candidate = f"Error: {answer.error}"
                else:
                    continue
                answer.score = self.score_answer(job.trajectory, candidate)
                if answer.score is None:
                    logger.warning(
                        "Trajectory %d: no score at step %d after %d attempts",
                        run.trajectory_id,
                        step.step,
                        MAX_ATTEMPTS,
                    )
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    def score_answer(self, trajectory: Trajectory, candidate: str) -> dict[str, int] | None:
        for attempt in range(MAX_ATTEMPTS):
            score = _score_one(
                "openai",
                self.client,
                JUDGE_MODEL,
                self.template,
                trajectory.question,
                trajectory.reference or trajectory.answer,
                candidate,
                reasoning_effort=REASONING_EFFORT,
            )
            if score is not None:
                return score
            time.sleep(min(60, 2**attempt))
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the forced answers against the reference (final answer of the SFT trajectory, or ground truth for own rollouts)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="output dir of answer.py",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="run id of answer.py",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="max workers (trajectories scored at once)",
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
        help="if set, only score the first N trajectories (for a quick sample run)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = JudgeRunner(args)
    runner.run(runner.create_jobs())


if __name__ == "__main__":
    main()
