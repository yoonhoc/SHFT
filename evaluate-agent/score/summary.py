"""run_agent.py·judge.py 결과로 실행분마다 점수와 호출 수를 SWE-QA-Pro·SWE-QA 나란히 낸다 (논문 표 2 의 열).

- Score: 채점된 문항의 평균 총점 (SWE-QA-Pro 5~50, SWE-QA 5~100). 채점 도중이면 "(채점된 수/문항 수)" 를 붙인다.
- Calls: 문항마다 실제로 실행한 도구 호출 수 (궤적의 tool_calls 합). 답 턴과 도구 없이 지나간 턴은 세지 않는다
  (저자 러너의 steps_completed 는 그 둘까지 세서 1~2 크다). 끝내 실패한 문항은 도구 호출 상한에서 답하다 멈춘 것으로
  보고 상한(max_iterations - 1: 마지막 턴은 강제 답변)으로 센다.

행은 out/ 와 ../ablation/<절제>/out/ 밑의 실행분 전부 (-r 로 고를 수 있다). 모델 키(run_agent_spec.json 의 name)로 묶고,
묶음 안에서는 원본 → SFT → 나머지(run-id 순).

--criteria 를 주면 채점 항목(correctness, completeness, relevance, clarity, reasoning)별 평균과 답의 단어 수를 따로 낸다
(원고 부록 표: 점수 차이가 어느 항목에서 나는지).

읽는 것: <output-dir>/<run-id>/<benchmark>/results/*.json, run_agent_spec.json

    python -m score.summary
    python -m score.summary --criteria -r qwen3-8b-nothink qwen3-8b-nothink-own-sis-w5
"""

import argparse
import json
from decimal import ROUND_HALF_UP, Decimal
import statistics
from pathlib import Path

from dataset import BENCHMARKS, NUM_QUESTIONS, VALIDATION_SET
from models import Result

# 표 머리글
COLUMN_NAMES = {"sweqapro": "SWE-QA-Pro", "sweqa": "SWE-QA", VALIDATION_SET: "validation (RL)"}
# 두 저자 채점기의 항목 (SWE-QA-Pro 1~10, SWE-QA 1~20)
CRITERIA = ["correctness", "completeness", "relevance", "clarity", "reasoning"]


def load_results(run_dir: Path) -> list[Result]:
    results = []
    for path in sorted((run_dir / "results").glob("*.json")):
        results.append(Result.model_validate_json(path.read_text()))
    return results


def count_calls(result: Result) -> int:
    if result.status == "success":
        return sum(len(turn.tool_calls) for turn in result.trajectory)
    return result.max_iterations - 1


def compute_row(results: list[Result]) -> dict:
    scores = []
    calls = []
    for result in results:
        if result.total is not None:
            scores.append(result.total)
        calls.append(count_calls(result))

    row = {"num_done": len(results), "num_judged": len(scores), "score": None, "calls": None}
    if scores:
        row["score"] = statistics.mean(scores)
    if calls:
        row["calls"] = statistics.mean(calls)
    return row


def format_number(value: float | None, digits: int) -> str:
    """원고 규칙: 정확한 평균에서 반올림 (ROUND_HALF_UP)."""
    if value is None:
        return "-"
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def find_model(run_dir: Path, benchmarks: list[str]) -> str:
    """실행분의 모델 키 (models.yaml 의 키). 설정 파일이 없으면 run-id."""
    for benchmark in benchmarks:
        spec_path = run_dir / benchmark / "run_agent_spec.json"
        if spec_path.exists():
            return json.loads(spec_path.read_text())["name"]
    return run_dir.name


def sort_key(run_id: str, model: str) -> tuple:
    if run_id == model:
        order = 0
    elif run_id == f"{model}-sft":
        order = 1
    else:
        order = 2
    return (model, order, run_id)


def format_cells(row: dict, num_questions: int) -> str:
    """한 벤치마크의 칸 두 개: Score, Calls."""
    if row["num_done"] == 0:
        return f"{'-':>18}{'-':>8}"
    score = format_number(row["score"], 2)
    if row["num_judged"] < num_questions:
        score += f" ({row['num_judged']}/{num_questions})"
    calls = format_number(row["calls"], 2)
    if row["num_done"] < num_questions:
        calls += "*"
    return f"{score:>18}{calls:>8}"


def compute_criteria(results: list[Result]) -> dict:
    """채점 항목별 평균과 답의 평균 단어 수 (채점된 문항만)."""
    sums = {}
    for name in CRITERIA:
        sums[name] = []
    words = []
    for result in results:
        if result.score is None:
            continue
        for name in CRITERIA:
            sums[name].append(result.score[name])
        words.append(len(result.answer.split()))
    row = {"num_judged": len(words)}
    for name in CRITERIA:
        row[name] = statistics.mean(sums[name]) if sums[name] else None
    row["words"] = statistics.mean(words) if words else None
    return row


def print_criteria(ordered: list[str], run_dirs: dict[str, Path], benchmarks: list[str]) -> None:
    for benchmark in benchmarks:
        print(f"\n== {benchmark}")
        header = "".join(f"{name[:12]:>13}" for name in CRITERIA)
        print(f"{'run':36s}{header}{'total':>8}{'words':>7}{'judged':>8}")
        for run_id in ordered:
            row = compute_criteria(load_results(run_dirs[run_id] / benchmark))
            if row["num_judged"] == 0:
                continue
            cells = "".join(f"{format_number(row[name], 2):>13}" for name in CRITERIA)
            total = sum(row[name] for name in CRITERIA)
            print(
                f"{run_id:36s}{cells}{format_number(total, 2):>8}{format_number(row['words'], 0):>7}{row['num_judged']:8d}"
            )


def default_output_dirs() -> list[str]:
    """evaluate-agent/out/, 기준선 폴더의 out/ (budget_tracker/out, autorater/out), 절제 폴더마다의 out/ (절제 판은 평가 결과도 절제 폴더 밑에 둔다)."""
    here = Path(__file__).resolve().parent.parent
    output_dirs = [str(here / "out"), str(here / "budget_tracker" / "out"), str(here / "autorater" / "out")]
    for path in sorted((here.parent / "ablation").glob("*/out")):
        output_dirs.append(str(path))
    return output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print score and tool calls of every run on both benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dirs",
        type=str,
        nargs="+",
        default=default_output_dirs(),
        help="output dirs of run_agent.py",
    )
    parser.add_argument(
        "-r",
        "--run-ids",
        type=str,
        nargs="+",
        default=None,
        help="run ids to show; default: every run under --output-dirs",
    )
    parser.add_argument(
        "-b",
        "--benchmarks",
        type=str,
        nargs="+",
        default=BENCHMARKS,
        choices=BENCHMARKS + [VALIDATION_SET],
        help="benchmarks to show as column groups (valid: validation questions for hyperparameter selection)",
    )
    parser.add_argument(
        "--criteria", action="store_true", help="also print the mean of each judge criterion and answer length"
    )
    return parser.parse_args()


def find_run_dirs(output_dirs: list[str], benchmarks: list[str]) -> dict[str, Path]:
    """run-id → 실행분 폴더. 같은 run-id 가 두 곳에 있으면 멈춘다 (표에서 구분이 안 된다)."""
    run_dirs = {}
    for output_dir in output_dirs:
        output_dir = Path(output_dir).expanduser()
        if not output_dir.exists():
            continue
        for run_dir in sorted(output_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not any((run_dir / benchmark / "results").exists() for benchmark in benchmarks):
                continue
            if run_dir.name in run_dirs:
                raise ValueError(f"Run id {run_dir.name} is in both {run_dirs[run_dir.name]} and {run_dir}")
            run_dirs[run_dir.name] = run_dir
    return run_dirs


def main() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.output_dirs, args.benchmarks)
    run_ids = args.run_ids
    if run_ids is None:
        run_ids = list(run_dirs)
    for run_id in run_ids:
        if run_id not in run_dirs:
            raise ValueError(f"Run id {run_id} has no results for {args.benchmarks} under {args.output_dirs}")

    models = {}
    keyed = []
    for run_id in run_ids:
        models[run_id] = find_model(run_dirs[run_id], args.benchmarks)
        keyed.append((sort_key(run_id, models[run_id]), run_id))
    ordered = []
    for _, run_id in sorted(keyed):
        ordered.append(run_id)

    rows = {}
    for run_id in ordered:
        for benchmark in args.benchmarks:
            rows[(run_id, benchmark)] = compute_row(load_results(run_dirs[run_id] / benchmark))

    header = f"{'Score':>18}{'Calls':>8}"
    print(f"{'':36s}" + "".join(f"{COLUMN_NAMES[benchmark]:>26}" for benchmark in args.benchmarks))
    print(f"{'run':36s}{header}{header}")
    previous_model = None
    for run_id in ordered:
        model = models[run_id]
        if previous_model is not None and model != previous_model:
            print()
        previous_model = model
        line = f"{run_id:36s}"
        for benchmark in args.benchmarks:
            line += format_cells(rows[(run_id, benchmark)], NUM_QUESTIONS[benchmark])
        print(line)
    print("\n* 실주행 도중 (끝난 문항만으로 셈)")
    if args.criteria:
        print_criteria(ordered, run_dirs, args.benchmarks)


if __name__ == "__main__":
    main()
