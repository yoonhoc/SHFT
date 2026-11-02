"""평가 벤치마크 두 개와 롤아웃용 문항 집합 하나를 문항 기록(`Item`)으로 불러온다.

- SWE-QA-Pro (Cai et al. 2026): 저자 러너와 같은 HF 평가 세트 260문항 (`sweqapro.data.load_benchmark`), 저장소 26개.
- SWE-QA (Peng et al. 2026): `third_party/SWE-QA-Bench/Benchmark/<저장소>.jsonl` 15개 × 48 = 720문항, 파일 이름 순.
- sweqapro-rl (롤아웃용, 평가 아님): SWE-QA-Pro 저자의 RL 학습 문항 `train/RL/verl-tool/dataset/train.parquet` 464문항.
  문항마다 저장소·커밋·ground truth 답이 있다. 저장소는 build-dataset/questions/clone.py 가 `build-dataset/questions/out/repos-rl/` 에 받는다 (평가 환경 eval/ 밑에 두면 path "." 검색이 그 40GB 까지 걷는다), 못 받은
  저장소(삭제된 것)의 문항은 뺀다. 자기 궤적으로 학습 데이터를 만들 때 쓰고, score.py 의 표에는 안 들어간다.
- valid (hyperparameter 선택용): sweqapro-rl 중 build-dataset/questions/split.py 가 validation 으로 뽑은 60문항. 학습 데이터에서는 뺀다
  (rationale.assemble·das.pairs 의 --split). 평가 벤치마크는 선택에 쓰지 않는다.

문항 번호(`index`)는 이 순서의 번호(sweqapro-rl 은 parquet 의 index)이고, 결과 파일 이름으로 쓴다.
"""

import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

THIRD_PARTY_DIR = Path(__file__).resolve().parents[1] / "third_party"
SWEQA_DIR = THIRD_PARTY_DIR / "SWE-QA-Bench" / "Benchmark"

BENCHMARKS = ["sweqapro", "sweqa"]
# RL 문항 중 validation (hyperparameter 선택용, 표에는 안 들어간다)
VALIDATION_SET = "valid"
NUM_QUESTIONS = {"sweqapro": 260, "sweqa": 720, VALIDATION_SET: 60}
# 평가 벤치마크가 아니라 자기 궤적 롤아웃용 문항 집합 (run_agent 만 받는다)
ROLLOUT_SETS = ["sweqapro-rl"]
RL_PARQUET = THIRD_PARTY_DIR / "SWE-QA-Pro" / "train" / "RL" / "verl-tool" / "dataset" / "train.parquet"
SPLIT_PATH = THIRD_PARTY_DIR.parent / "build-dataset" / "questions" / "out" / "split.json"
# 각 저자의 clone_repos.sh 가 저장소를 받는 곳 (README 참고). sweqapro-rl 은 build-dataset/questions/clone.py
REPO_ROOTS = {
    "sweqapro": THIRD_PARTY_DIR / "SWE-QA-Pro" / "eval" / "repos",
    "sweqa": THIRD_PARTY_DIR / "SWE-QA-Bench" / "datas" / "repos",
    "sweqapro-rl": THIRD_PARTY_DIR.parent / "build-dataset" / "questions" / "out" / "repos-rl",
    VALIDATION_SET: THIRD_PARTY_DIR.parent / "build-dataset" / "questions" / "out" / "repos-rl",
}


class Item(BaseModel):
    index: int
    # 저장소 이름. SWE-QA-Pro 는 "<owner>/<name>", SWE-QA 는 파일 이름 ("flask")
    repo: str
    question: str
    # 벤치마크 정답 (채점 기준 답)
    answer: str
    # SWE-QA-Pro 의 나머지 열 (commit_id, cluster, qa_type), sweqapro-rl 은 commit_id·repo_name. SWE-QA 는 비어 있다
    meta: dict = {}


def load_items(benchmark: str) -> list[Item]:
    if benchmark == "sweqapro":
        return load_sweqapro()
    if benchmark == "sweqa":
        return load_sweqa()
    if benchmark == "sweqapro-rl":
        return load_sweqapro_rl()
    if benchmark == VALIDATION_SET:
        return load_validation()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def load_split() -> dict:
    return json.loads(SPLIT_PATH.read_text())


def load_validation() -> list[Item]:
    """RL 문항 중 split.json 의 validation 문항."""
    validation = set(load_split()["validation"])
    return [item for item in load_sweqapro_rl() if item.index in validation]


def load_sweqapro() -> list[Item]:
    # 저자 코드(와 datasets)는 여기서만 부른다. score 처럼 문항 수만 쓰는 곳은 아무 파이썬으로 돈다
    from sweqapro.data import load_benchmark

    items = []
    for row in load_benchmark(split="test"):
        meta = {"commit_id": row["commit_id"], "cluster": row["cluster"], "qa_type": row["qa_type"]}
        item = Item(index=len(items), repo=row["repo"], question=row["question"], answer=row["answer"], meta=meta)
        items.append(item)
    return items


def load_sweqa() -> list[Item]:
    items = []
    for path in sorted(SWEQA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(Item(index=len(items), repo=path.stem, question=row["question"], answer=row["answer"]))
    return items


def load_sweqapro_rl() -> list[Item]:
    import pyarrow.parquet as pq

    rows = pq.read_table(RL_PARQUET).to_pylist()
    names = []
    for row in rows:
        names.append(row["extra_info"]["repo_name"].split("/")[-1])
    items = []
    for row in rows:
        info = row["extra_info"]
        owner, name = info["repo_name"].split("/")
        # 저장소 폴더 이름: build-dataset/questions/load.py 와 같은 규칙 (이름이 겹치면 <owner>__<name>)
        repo = name
        if names.count(name) > 1:
            repo = f"{owner}__{name}"
        if not (REPO_ROOTS["sweqapro-rl"] / repo).exists():
            logger.warning("Skip question %s: repository %s not cloned", info["index"], info["repo_name"])
            continue
        meta = {"commit_id": info["commit_id"], "repo_name": info["repo_name"]}
        item = Item(
            index=int(info["index"]),
            repo=repo,
            question=info["question"],
            answer=row["reward_model"]["ground_truth"],
            meta=meta,
        )
        items.append(item)
    return items
