"""RL 학습 문항의 저장소 464개를 GitHub 에서 받아 문항의 커밋으로 맞춘다 (저자 tool server 의 clone_and_checkout 과 같은 순서).

이미 받은 저장소는 건너뛰고, 못 받은 저장소는 끝에 찍는다 (그 문항은 뒤 단계에서 빠진다).

읽는 것: questions/load.py 의 문항 (repo_name, commit_id, repo_dir)
쓰는 것: <repo-root>/<repo_dir>/ (기본 questions/out/repos-rl/; 평가 환경 eval/ 밑에 두지 않는다 — 에이전트의 path "." 검색이 cwd 전체를 걷는다)

    cd code/build-dataset
    python -m questions.clone
"""

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from questions.load import Question, load_questions


def clone(question: Question, repo_root: Path) -> str | None:
    """저장소를 받고 커밋으로 맞춘다. 실패하면 이유를 돌려준다."""
    repo_dir = repo_root / question.repo_dir
    if (repo_dir / ".git").exists():
        return None
    shutil.rmtree(repo_dir, ignore_errors=True)
    url = f"https://github.com/{question.repo_name}.git"
    commands = [
        (["git", "clone", "-q", url, question.repo_dir], repo_root),
        (["git", "checkout", "-q", "-f", question.commit_id], repo_dir),
    ]
    for command, cwd in commands:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        )
        if completed.returncode != 0:
            shutil.rmtree(repo_dir, ignore_errors=True)
            return completed.stderr.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone the repositories of the RL questions at their commits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "repos-rl"),
        help="where to clone the repositories",
    )
    parser.add_argument("--workers", type=int, default=8, help="parallel clones")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser()
    repo_root.mkdir(parents=True, exist_ok=True)
    questions = load_questions()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(clone, question, repo_root) for question in questions]
        errors = [future.result() for future in tqdm(futures)]
    for question, error in zip(questions, errors):
        if error is not None:
            print(f"failed {question.index} {question.repo_name}: {error}")
    print(f"{len(questions) - errors.count(None)} failed of {len(questions)}")


if __name__ == "__main__":
    main()
