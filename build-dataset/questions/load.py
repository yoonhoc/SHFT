"""SWE-QA-Pro 저자 RL 학습 문항 (verl-tool 의 train.parquet, 464문항) 을 읽는다.

SFT 궤적 1,000개와 달리 이 문항들은 commit_id 와 ground truth 답이 있어, 저장소를 그 커밋으로 받아 에이전트를 직접 돌리고
(자기 롤아웃) 강제 답변을 ground truth 로 채점할 수 있다. questions/ 의 다른 단계가 모두 이 모듈로 문항을 읽는다.

읽는 것: train.parquet (load_questions 의 인자; 행마다 extra_info 의 question·repo_name·commit_id·answer)
쓰는 것: 없음 (다른 모듈이 import 한다)
"""

from pathlib import Path

import pyarrow.parquet as pq
from pydantic import BaseModel

PARQUET = Path(__file__).resolve().parents[2] / "third_party/SWE-QA-Pro/train/RL/verl-tool/dataset/train.parquet"


class Question(BaseModel):
    index: int
    question: str
    # "<owner>/<name>"
    repo_name: str
    commit_id: str
    # ground truth 답 (reward_model.ground_truth 와 같다)
    answer: str
    # 저장소를 받아 둘 폴더 이름. 보통 <name>, 이름이 겹치는 저장소만 <owner>__<name>
    repo_dir: str


def load_questions(parquet: Path = PARQUET) -> list[Question]:
    table = pq.read_table(parquet).to_pylist()
    names = []
    for row in table:
        names.append(row["extra_info"]["repo_name"].split("/")[-1])
    questions = []
    for row in table:
        info = row["extra_info"]
        owner, name = info["repo_name"].split("/")
        repo_dir = name
        if names.count(name) > 1:
            repo_dir = f"{owner}__{name}"
        question = Question(
            index=int(info["index"]),
            question=info["question"],
            repo_name=info["repo_name"],
            commit_id=info["commit_id"],
            answer=row["reward_model"]["ground_truth"],
            repo_dir=repo_dir,
        )
        questions.append(question)
    return questions
