"""judge.py 가 매긴 점수로 궤적마다 정지 스텝 i* 와 그 자리에 넣을 답을 골라 같은 파일에 채운다.

지점마다 강제 답변들의 평균 총점을 내고, 가장 높은 지점을 i* 로 한다 (동점이면 가장 이른 지점).
모델이 답하지 못한 강제 답변은 judge.py 가 저자처럼 "Error: <이유>" 로 채점한 점수를 그대로 센다.
문맥 한도를 넘어 받지 못한 답(채점 안 함)과 채점 결과를 못 읽은 답은 뺀다.
그 자리의 정지 응답에 넣을 답은 i* 지점의 답 중 총점이 가장 높은 것이다 (동점이면 앞의 것).
"""

import argparse
import statistics
from pathlib import Path

from golden_step.models import Run, Step


def compute_mean_score(step: Step) -> float | None:
    """한 지점의 평균 총점 (채점된 답만, 답하지 못해 "Error: …" 로 채점된 답 포함). 채점된 답이 없으면 None."""
    totals = []
    for answer in step.answers:
        # 문맥 한도를 넘어 채점하지 않은 답과 점수를 못 읽은 답은 점수가 없다
        if answer.total is not None:
            totals.append(answer.total)
    if not totals:
        return None
    return statistics.mean(totals)


def find_best_step(run: Run) -> int | None:
    """평균 점수가 가장 높은 지점 i*. 동점이면 가장 이른 지점, 채점된 지점이 없으면 None."""
    best_step = None
    best_score = None
    for step in run.steps:
        score = compute_mean_score(step)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_step = step.step
            best_score = score
    return best_step


def select_answer(step: Step) -> str | None:
    """한 지점의 답 중 총점이 가장 높은 것. 동점이면 앞의 것."""
    best_answer = None
    best_total = None
    for answer in step.answers:
        if answer.answer is None or answer.total is None:
            continue
        if best_total is None or answer.total > best_total:
            best_answer = answer.answer
            best_total = answer.total
    return best_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the best-scored step i* of every trajectory as its stop step",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    answer_dir = Path(args.output_dir).expanduser() / args.run_id / "answers"
    halting_steps = []
    num_truncated, num_unscored = 0, 0
    for path in sorted(answer_dir.glob("*.json")):
        run = Run.model_validate_json(path.read_text())
        run.halting_step = find_best_step(run)
        if run.halting_step is None:
            num_unscored += 1
            run.halting_answer = None
        else:
            run.halting_answer = select_answer(run.steps[run.halting_step])
            halting_steps.append(run.halting_step)
            # steps 는 지점 0..n 이라 마지막 지점이 n
            if run.halting_step < len(run.steps) - 1:
                num_truncated += 1
        path.write_text(run.model_dump_json(indent=2), encoding="utf-8")

    print(f"Labeled {len(halting_steps)} trajectories in {answer_dir} ({num_unscored} without any score)")
    if halting_steps:
        print(f"mean i* = {statistics.mean(halting_steps):.2f}, truncated = {num_truncated}")


if __name__ == "__main__":
    main()
