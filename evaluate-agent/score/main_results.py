"""원고 표 1 의 행만, 표 1 의 순서로 점수와 호출 수를 낸다 (summary.py 의 계산을 그대로 쓰고 행 목록만 고정).

행 ↔ 평가 run id 는 ROWS 에 적는다 (validation 으로 고른 checkpoint, train/README.md "선택된 checkpoint" 절과 같다).
아직 결과가 없는 행은 "-" 로 둔다. 새 행이 생기면 ROWS 에 한 줄 더한다.
점수는 벤치마크마다 원 논문 judge 계열로 다시 채점한 것: SWE-QA-Pro 는 GPT-6 Sol (gpt_judge/sweqapro.py),
SWE-QA 는 Claude Sonnet 5.5 (claude_judge/sweqa.py). validation 은 checkpoint 를 고른 GPT-6 Luna (judge.py) 그대로.
호출 수는 모두 results/ 에서 센다.

읽는 것: out/, budget_tracker/out/, autorater/out/ 의 <run-id>/<benchmark>/results/*.json, <run-id>/sweqapro/results-gpt-6-sol/*.json,
         <run-id>/sweqa/results-claude-sonnet-5-5/*.json

    python -m score.main_results
    python -m score.main_results -b valid     # 같은 행을 validation 60문항으로
"""

import argparse
import statistics
from pathlib import Path

from dataset import BENCHMARKS, NUM_QUESTIONS, VALIDATION_SET
from models import Result
from score.summary import COLUMN_NAMES, compute_row, format_cells, load_results

# 벤치마크마다 점수를 읽는 폴더 (없으면 results/ 의 Luna 점수)
SCORE_DIRS = {"sweqapro": "results-gpt-6-sol", "sweqa": "results-claude-sonnet-5-5"}

# (모델, 묶음, 표 라벨, 평가 run id). 묶음이 바뀌는 곳에 빈 줄을 넣는다.
ROWS = [
    ("Qwen3-8B", "On-Policy", "Base", "qwen3-8b-nothink"),
    ("Qwen3-8B", "On-Policy", "Budget Tracker", "qwen3-8b-nothink-bats"),
    ("Qwen3-8B", "On-Policy", "AutoRater", "qwen3-8b-nothink-sufficiency"),
    ("Qwen3-8B", "On-Policy", "Plain", "qwen3-8b-nothink-own-plain-e4"),
    ("Qwen3-8B", "On-Policy", "SHFT", "qwen3-8b-nothink-own-sis-w5-e1"),
    ("Qwen3-8B", "Off-Policy", "SFT", "qwen3-8b-nothink-sft-e4"),
    ("Qwen3-8B", "Off-Policy", "Budget Tracker", "qwen3-8b-nothink-sft-e4-bats"),
    ("Qwen3-8B", "Off-Policy", "AutoRater", "qwen3-8b-nothink-sft-e4-sufficiency"),
    ("Qwen3-8B", "Off-Policy", "Plain", "qwen3-8b-nothink-expert-plain-e4"),
    ("Qwen3-8B", "Off-Policy", "SHFT", "qwen3-8b-nothink-expert-sis-w5-e4"),
    ("Qwen3-8B", "Off + On", "SHFT (Off) + Plain (On)", "qwen3-8b-nothink-expert-sis-own-plain-e3"),
    ("Qwen3-8B", "Off + On", "SHFT (Off) + SHFT (On)", "qwen3-8b-nothink-expert-sis-own-sis-w5-e1"),
    ("Ministral-3-8B", "On-Policy", "Base", "ministral-8b"),
    ("Ministral-3-8B", "On-Policy", "Budget Tracker", "ministral-8b-bats"),
    ("Ministral-3-8B", "On-Policy", "AutoRater", "ministral-8b-sufficiency"),
    ("Ministral-3-8B", "On-Policy", "Plain", "ministral-8b-own-plain-e4"),
    ("Ministral-3-8B", "On-Policy", "SHFT", "ministral-8b-own-sis-w5-e2"),
    ("Ministral-3-8B", "Off-Policy", "SFT", "ministral-8b-sft-e2"),
    ("Ministral-3-8B", "Off-Policy", "Budget Tracker", "ministral-8b-sft-e2-bats"),
    ("Ministral-3-8B", "Off-Policy", "AutoRater", "ministral-8b-sft-e2-sufficiency"),
    ("Ministral-3-8B", "Off-Policy", "Plain", "ministral-8b-expert-plain-e4"),
    ("Ministral-3-8B", "Off-Policy", "SHFT", "ministral-8b-expert-sis-w5-e3"),
    ("Ministral-3-8B", "Off + On", "SHFT (Off) + Plain (On)", "ministral-8b-expert-sis-own-plain-e4"),
    ("Ministral-3-8B", "Off + On", "SHFT (Off) + SHFT (On)", "ministral-8b-expert-sis-own-sis-w5-e3"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the rows of Table 1 (score and tool calls) in the order of the manuscript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dirs",
        type=str,
        nargs="+",
        default=["out", "budget_tracker/out", "autorater/out"],
        help="output dirs of run_agent.py, relative to this folder",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        nargs="+",
        default=BENCHMARKS,
        choices=BENCHMARKS + [VALIDATION_SET],
        help="benchmarks to show as column groups",
    )
    return parser.parse_args()


def find_run_dir(run_id: str, output_dirs: list[Path]) -> Path | None:
    for output_dir in output_dirs:
        if (output_dir / run_id).is_dir():
            return output_dir / run_id
    return None


def load_scores(score_dir: Path) -> list[int]:
    scores = []
    for path in sorted(score_dir.glob("*.json")):
        result = Result.model_validate_json(path.read_text())
        if result.total is not None:
            scores.append(result.total)
    return scores


def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent.parent
    output_dirs = []
    for output_dir in args.output_dirs:
        output_dirs.append(here / output_dir)

    header = f"{'Score':>18}{'Calls':>8}"
    print(f"{'':44s}" + "".join(f"{COLUMN_NAMES[benchmark]:>26}" for benchmark in args.benchmarks))
    print(f"{'row':44s}{header * len(args.benchmarks)}")
    previous = None
    for model, group, label, run_id in ROWS:
        if previous is not None and (model, group) != previous:
            print()
        if previous is None or model != previous[0]:
            print(f"== {model}")
        previous = (model, group)
        line = f"{group + '  ' + label:44s}"
        run_dir = find_run_dir(run_id, output_dirs)
        for benchmark in args.benchmarks:
            if run_dir is None or not (run_dir / benchmark / "results").exists():
                line += f"{'-':>18}{'-':>8}"
                continue
            row = compute_row(load_results(run_dir / benchmark))
            if benchmark in SCORE_DIRS:
                scores = load_scores(run_dir / benchmark / SCORE_DIRS[benchmark])
                row["num_judged"] = len(scores)
                row["score"] = statistics.mean(scores) if scores else None
            line += format_cells(row, NUM_QUESTIONS[benchmark])
        print(f"{line}   {run_id}")
    print("\n* 실주행 도중 (끝난 문항만으로 셈), - 아직 결과 없음")
    print("점수: SWE-QA-Pro = GPT-6 Sol, SWE-QA = Claude Sonnet 5.5, validation = GPT-6 Luna")


if __name__ == "__main__":
    main()
