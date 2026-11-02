"""정지 스텝 h = i* 에서 teacher LLM 이 정지 서술 z_h 를 쓴다.

teacher 는 에이전트가 본 대화(스텝 0..h-1, system 만 teacher 용으로 바꿈)와 정지 응답에 들어갈 답 y(s_h)
(golden_step/label.py 가 고른 답)를 읽고, 그 답 직전에 에이전트가 할 말("지금까지 무엇을 확인했고, 그것으로 답하기에
충분하다")을 쓴다. 말투는 Claude 가 SFT 궤적 끝에서 실제로 쓴 정지 서술 3개를 예시로 준다.
h = n 인 궤적은 Claude 의 최종 응답을 그대로 쓰므로 건너뛴다.

결과는 `<output-dir>/<run-id>/rationales/<trajectory_id>-<step>.txt`, 파일이 있으면 건너뛴다.
teacher 모델은 미리 vLLM 으로 띄워 둔다 (README 참고).
"""

import argparse
import logging
import re
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from dataset import Trajectory, load_sft_trajectories, load_trajectories
from golden_step.models import Run
from runner import Runner
from utils import save_args, setup_logging, tee_output

logger = logging.getLogger(__name__)

INSTRUCTION = """You are helping build training data for a codebase question-answering agent.

You will see a conversation in which the agent (the "assistant") explores a code repository with tools
to answer a user's question. The conversation stops at the point where the agent has gathered enough
evidence. You will then be given the final answer the agent is about to write.

Your job: write the agent's last message before that final answer, in the agent's own voice.
The message should say what has been established (naming the specific files or functions seen in the
conversation) and that this is enough to answer the question. 2-4 sentences. Do not announce any further
investigation. Do not write the answer itself. Output only the message.

Here are examples of how this agent writes that message:
"""

REQUEST = """The agent is now about to write this final answer:
<answer>
{answer}
</answer>

Write the agent's message right before this answer."""

# 예시로 쓸 SFT 궤적: Claude 가 궤적 끝에서 쓴 정지 서술 (모두 "Perfect!" 로 시작)
EXAMPLE_TRAJECTORY_IDS = [0, 23, 35]


class Job(BaseModel):
    trajectory: Trajectory
    step: int
    answer: str


class RationaleRunner(Runner):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = Path(args.output_dir).expanduser() / args.run_id
        self.answer_dir = Path(args.golden_step_dir).expanduser() / (args.golden_step_run_id or args.run_id) / "answers"
        self.rationale_dir = self.run_dir / "rationales"
        self.rationale_dir.mkdir(parents=True, exist_ok=True)

        setup_logging(self.run_dir / "rationale.log")
        tee_output(self.run_dir / "rationale_stdout.log")
        save_args(args, self.run_dir / "rationale_args.json")

        self.trajectories = {t.trajectory_id: t for t in load_trajectories(args.trajectories)}
        # 예시 서술은 늘 SFT 궤적(Claude)에서 가져온다
        examples = {t.trajectory_id: t for t in load_sft_trajectories()}
        self.system_prompt = build_system_prompt(examples)
        self.client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800)

    @property
    def max_workers(self) -> int:
        return self.args.max_workers

    def create_jobs(self) -> list[Job]:
        jobs = []
        for path in sorted(self.answer_dir.glob("*.json")):
            run = Run.model_validate_json(path.read_text())
            trajectory = self.trajectories[run.trajectory_id]
            # i* = n 이면 Claude 의 최종 응답을 그대로 쓴다
            if run.halting_step is None or run.halting_step == trajectory.n:
                continue
            if (self.rationale_dir / f"{run.trajectory_id:04d}-{run.halting_step:02d}.txt").exists():
                continue
            if run.halting_answer is None:
                logger.warning("Trajectory %d: no answer at step %d, skip", run.trajectory_id, run.halting_step)
                continue
            jobs.append(Job(trajectory=trajectory, step=run.halting_step, answer=run.halting_answer))
        return jobs

    def process_job(self, job: Job) -> None:
        messages = job.trajectory.prefix_messages(job.step)
        # 에이전트용 system 프롬프트를 teacher 용으로 바꾼다
        messages[0] = {"role": "system", "content": self.system_prompt}
        messages.append({"role": "user", "content": REQUEST.format(answer=job.answer)})
        response = self.client.chat.completions.create(
            model=self.args.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.args.max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        text = response.choices[0].message.content or ""
        rationale = clean_rationale(text)
        if rationale != text.strip():
            logger.info("Trajectory %d: trimmed the teacher output", job.trajectory.trajectory_id)
        out_path = self.rationale_dir / f"{job.trajectory.trajectory_id:04d}-{job.step:02d}.txt"
        out_path.write_text(rationale, encoding="utf-8")


def build_system_prompt(trajectories: dict[int, Trajectory]) -> str:
    """지시문 + 예시 궤적에서 Claude 가 <finish> 앞에 쓴 정지 서술."""
    examples = []
    for trajectory_id in EXAMPLE_TRAJECTORY_IDS:
        example = trajectories[trajectory_id].final_response.split("<finish>")[0].strip()
        examples.append("<example>\n" + example + "\n</example>")
    return INSTRUCTION + "\n".join(examples)


def clean_rationale(text: str) -> str:
    """teacher 가 서술 뒤에 도구 호출이나 <finish> 답까지 이어 쓰면 그 앞까지만, 서술을 태그로 감싸면 태그만 뗀다."""
    text = text.split("<tool_call>")[0].split("<finish>")[0]
    return re.sub(r"</?(message|answer)>", "", text).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a stop rationale with a teacher LLM at the stop step of every truncated trajectory",
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
        help="run id (same as golden_step/)",
    )
    parser.add_argument(
        "--golden-step-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "golden_step" / "out"),
        help="output dir of golden_step/ (answers with halting steps)",
    )
    parser.add_argument(
        "--golden-step-run-id",
        type=str,
        default=None,
        help="run id of golden_step/ to read; default: --run-id (set it when an ablation writes under its own run id)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8102/v1",
        help="vLLM server of the teacher model",
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
        default="Qwen/Qwen3.8-27B",
        help="served teacher model name",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default="sft",
        help="trajectories: sft, or a results dir of evaluate-agent/run_agent.py (the agent model's own rollouts)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="max tokens of a rationale",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=64,
        help="max workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = RationaleRunner(args)
    runner.run(runner.create_jobs())


if __name__ == "__main__":
    main()
