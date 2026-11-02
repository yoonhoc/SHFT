"""RL 문항을 학습용과 validation 용으로 나눈다 (hyperparameter 선택은 validation 으로만, 평가 벤치마크는 선택에 쓰지 않는다).

문항마다 저장소가 하나라 문항 단위로 나누되, 같은 저장소의 문항이 양쪽에 걸치지 않게 저장소 이름으로 묶어 뽑는다.
저장소를 못 받은 문항(삭제된 저장소)과 평가 벤치마크(SWE-QA-Pro, SWE-QA)에 같은 이름의 저장소가 있는 문항은 어느 쪽에도 넣지 않는다
(RL 문항은 SWE-QA-Pro 와는 겹치지 않고 SWE-QA 와 2개 겹친다). seed 를 고정해 한 번 만든 뒤에는 바꾸지 않는다.

읽는 것: questions/load.py 의 RL 문항, --repo-root (받아 둔 저장소 폴더; 없는 문항은 뺀다), --benchmark-repos (벤치마크 저장소 폴더들)
쓰는 것: --output (split.json: {"seed", "validation": [문항 번호], "train": [문항 번호], "excluded": [문항 번호]})
    evaluate-agent 의 "valid" 벤치마크와 rationale.assemble·das.pairs 의 --split 이 이 파일을 읽는다.

    cd code/build-dataset
    python -m questions.split
"""

import argparse
import json
import random
from pathlib import Path

from questions.load import Question, load_questions

HERE = Path(__file__).resolve().parent
THIRD_PARTY = HERE.parents[1] / "third_party"
BENCHMARK_REPOS = [THIRD_PARTY / "SWE-QA-Pro" / "eval" / "repos", THIRD_PARTY / "SWE-QA-Bench" / "datas" / "repos"]


def split_questions(questions: list[Question], num_validation: int, seed: int) -> tuple[list[int], list[int]]:
    """저장소 단위로 섞어 앞에서부터 validation 문항을 num_validation 개 채운다. 나머지가 학습용."""
    by_repo: dict[str, list[int]] = {}
    for question in questions:
        by_repo.setdefault(question.repo_name, []).append(question.index)
    repos = sorted(by_repo)
    random.Random(seed).shuffle(repos)
    validation = []
    for repo in repos:
        if len(validation) >= num_validation:
            break
        validation.extend(by_repo[repo])
    validation_set = set(validation)
    train = [q.index for q in questions if q.index not in validation_set]
    return sorted(validation), sorted(train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split the RL questions into train and validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(HERE / "out" / "split.json"),
        help="where to write the split",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(HERE / "out" / "repos-rl"),
        help="cloned repositories; questions whose repository is missing are left out",
    )
    parser.add_argument(
        "--benchmark-repos",
        type=str,
        nargs="+",
        default=[str(path) for path in BENCHMARK_REPOS],
        help="repository folders of the evaluation benchmarks; questions on a repository of the same name are excluded",
    )
    parser.add_argument("--num-validation", type=int, default=60, help="validation questions")
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser()
    benchmark_names = set()
    for folder in args.benchmark_repos:
        for path in Path(folder).expanduser().iterdir():
            benchmark_names.add(path.name.lower())
    questions, excluded = [], []
    for question in load_questions():
        if not (repo_root / question.repo_dir).exists():
            continue
        if question.repo_name.split("/")[-1].lower() in benchmark_names:
            excluded.append(question.index)
            continue
        questions.append(question)
    validation, train = split_questions(questions, args.num_validation, args.seed)
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    split = {"seed": args.seed, "validation": validation, "train": train, "excluded": sorted(excluded)}
    out_path.write_text(json.dumps(split, indent=2))
    print(
        f"validation {len(validation)}, train {len(train)}, excluded (benchmark repository) {len(excluded)} -> {out_path}"
    )


if __name__ == "__main__":
    main()
