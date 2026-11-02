"""학습 데이터의 바탕이 되는 궤적을 궤적 기록(`Trajectory`)으로 불러오고, 학습·측정에 쓰는 대화(OpenAI 형식 messages)로 되돌린다.

궤적 출처는 둘이다 (`load_trajectories(source)`):
- "sft": SWE-QA-Pro SFT 궤적 (Claude Sonnet 4.5 가 만든 1,000개). system·user 메시지와 도구 스키마는 공개 데이터 그대로 쓴다.
  바꾸는 것은 도구 호출·결과의 표기뿐이다: 공개 데이터는 `tool_call` / `tool_response` 역할의 JSON 글로 적어 두어
  채팅 템플릿이 읽지 못하므로, assistant 메시지의 `tool_calls` 와 `tool` 역할 메시지로 바꾼다.
- evaluate-agent/run_agent.py 의 results 폴더: 에이전트 모델 자신의 롤아웃 (예: RL 문항 집합 sweqapro-rl 의 자기 궤적).
  system·user 메시지와 도구 스키마는 그 실주행이 쓴 저자 에이전트의 것으로 다시 만들고, 문항의 ground truth 답을
  `reference` 에 둔다 (golden_step/judge.py 가 궤적의 최종 답 대신 이것으로 채점한다).
"""

import json
import logging
from pathlib import Path

from datasets import load_dataset
from pydantic import BaseModel

logger = logging.getLogger(__name__)

SFT_DATASET = "TIGER-Lab/SWE-QA-Pro-SFT-Trajectories"
# run_agent.py 가 --repo-root 없이 쓰는 저장소 폴더 이름 (evaluate-agent/dataset.py 의 REPO_ROOTS 와 같게)
REPO_ROOT_NAMES = {"sweqapro": "repos", "sweqa": "repos", "sweqapro-rl": "repos-rl", "valid": "repos-rl"}


class ToolCall(BaseModel):
    tool: str
    args: dict
    result: str


class Step(BaseModel):
    """assistant 턴 하나와 그 턴의 도구 호출·결과."""

    response: str
    tool_calls: list[ToolCall]


class Trajectory(BaseModel):
    trajectory_id: int
    # 공개 데이터 그대로의 system·user 메시지와 도구 스키마
    system_prompt: str
    user_prompt: str
    tools: list[dict]
    repo_path: str
    question: str
    steps: list[Step]
    # 도구 호출 없이 끝나는 마지막 assistant 턴 (<finish> 블록 포함)
    final_response: str
    # 채점 기준 답. SFT 궤적은 없고(궤적의 최종 답을 쓴다), 자기 롤아웃은 문항의 ground truth
    reference: str | None = None

    @property
    def n(self) -> int:
        """궤적 스스로의 정지 스텝 = 최종 응답 전까지의 스텝 수."""
        return len(self.steps)

    @property
    def answer(self) -> str:
        """최종 응답의 <finish> 안쪽. 질문에 대한 답 r 로 쓴다."""
        return extract_answer(self.final_response)

    def prefix_messages(self, i: int) -> list[dict]:
        """system + user + 스텝 0..i-1. 스텝 i 를 앞둔 문맥 s_i."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]
        for step_idx, step in enumerate(self.steps[:i]):
            messages.extend(to_messages(step_idx, step))
        return messages

    def messages(self) -> list[dict]:
        """궤적 전체 (최종 응답까지)."""
        return self.prefix_messages(self.n) + [{"role": "assistant", "content": self.final_response}]


def to_messages(step_idx: int, step: Step) -> list[dict]:
    """스텝 하나 → assistant(본문 + 도구 호출) 메시지와 도구 결과 메시지들.

    tool_call id 는 새로 붙인다. 원본 궤적에 id 가 없고, assistant 와 tool 메시지의 짝만 맞으면 된다.
    """
    call_ids = [f"call_{step_idx}_{call_idx}" for call_idx in range(len(step.tool_calls))]
    assistant = {
        "role": "assistant",
        "content": step.response,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": call.tool, "arguments": json.dumps(call.args)},
            }
            for call, call_id in zip(step.tool_calls, call_ids)
        ],
    }
    tools = [
        {"role": "tool", "content": call.result, "tool_call_id": call_id}
        for call, call_id in zip(step.tool_calls, call_ids)
    ]
    return [assistant] + tools


def extract_answer(text: str) -> str:
    """<finish>…</finish> 안쪽. 태그가 없으면 원문 그대로."""
    if "<finish>" in text:
        return text.split("<finish>", 1)[1].split("</finish>", 1)[0].strip()
    return text.strip()


def parse_user_prompt(content: str) -> tuple[str, str]:
    """첫 user 메시지(저자 프롬프트)에서 (repo_path, question) 을 꺼낸다."""
    repo_path = content.split("Repository Path:")[1].splitlines()[0].strip()
    question = content.split("Question:")[1].split("\n\nPlease")[0].strip()
    return repo_path, question


def parse_steps(messages: list[dict]) -> tuple[list[Step], str | None]:
    """원본 messages(assistant / tool_call / tool_response 가 한 줄씩) → (스텝 목록, 최종 응답)."""
    steps: list[Step] = []
    current: Step | None = None
    for message in messages:
        role = message["role"]
        content = message.get("content") or ""
        if role == "assistant":
            if current is not None and current.tool_calls:
                steps.append(current)
            current = Step(response=content, tool_calls=[])
        elif role == "tool_call":
            call = json.loads(content)
            if current is None:
                current = Step(response="", tool_calls=[])
            current.tool_calls.append(ToolCall(tool=call["name"], args=call.get("arguments") or {}, result=""))
        elif role == "tool_response":
            if current is not None and current.tool_calls:
                current.tool_calls[-1].result = content
    # 마지막 assistant 턴: 도구 호출이 없으면 최종 응답, 있으면 마지막 스텝
    if current is None:
        return steps, None
    if current.tool_calls:
        steps.append(current)
        return steps, None
    return steps, current.response


def load_sft_trajectories() -> list[Trajectory]:
    trajectories = []
    dataset = load_dataset(SFT_DATASET, split="train")
    for idx, row in enumerate(dataset):
        try:
            repo_path, question = parse_user_prompt(row["messages"][1]["content"])
            steps, final_response = parse_steps(row["messages"])
        except (IndexError, KeyError, json.JSONDecodeError):
            logger.warning("Skip trajectory %d: failed to parse", idx)
            continue
        if not steps or not final_response:
            logger.warning("Skip trajectory %d: no tool call or no final response", idx)
            continue
        trajectory = Trajectory(
            trajectory_id=idx,
            system_prompt=row["messages"][0]["content"],
            user_prompt=row["messages"][1]["content"],
            tools=row["tools"],
            repo_path=repo_path,
            question=question,
            steps=steps,
            final_response=final_response,
        )
        trajectories.append(trajectory)
    return trajectories


def strip_open_think(response: str) -> str:
    """생각 없이 도구를 부를 때 Qwen3 가 쓰는 닫히지 않은 빈 "<think>\n" 을 뗀다.

    학습 템플릿은 </think> 가 없으면 본문으로 보고 앞에 빈 생각 블록을 또 붙여, 모델이 낸 적 없는 이중 think 형식이 된다
    (2026-10-10 own SIS 첫 판: 탐색 턴 79% 가 이 형식으로 학습돼 탐색 행동이 바뀜). 떼면 정식 "생각 없음" 형식으로 렌더된다.
    """
    if "<think>" in response and "</think>" not in response and not response.split("<think>", 1)[1].strip():
        return ""
    return response


def load_own_trajectories(results_dir: Path) -> list[Trajectory]:
    """evaluate-agent/run_agent.py 의 results/<index>.json (자기 롤아웃) → Trajectory. 롤아웃이 실패해 답이 없는 문항만 뺀다.

    system·user 메시지는 저자 에이전트가 그 실주행에서 만든 것과 같게 다시 만든다 (저장소 경로 = 저장소 폴더 이름 + 저장소 이름,
    run_agent.py 의 규칙. 폴더 이름은 results 의 부모 폴더에 있는 run_agent_args.json 의 repo_root, 없으면 벤치마크의 기본 폴더).
    """
    from sweqapro.agent import ToolCallingAgent
    from sweqapro.tool_schemas import TOOL_SCHEMAS

    results_dir = Path(results_dir).expanduser()
    args = json.loads((results_dir.parent / "run_agent_args.json").read_text())
    if args.get("repo_root"):
        repo_root_name = Path(args["repo_root"]).name
    else:
        repo_root_name = REPO_ROOT_NAMES[args["benchmark"]]
    system_template = ToolCallingAgent._load_system_prompt(None)
    trajectories = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        turns = result["trajectory"]
        if (
            result["status"] != "success"
            or not turns
            or turns[-1]["tool_calls"]
            or "<finish>" not in turns[-1]["response"]
        ):
            logger.warning("Skip question %s: no completed trajectory", result["item"]["index"])
            continue
        # 마지막 답 전의 턴은 있는 그대로 스텝으로 둔다. 도구 없이 글만 쓴 턴, 퇴화한 턴, 받아들여지지 않은 답 턴도
        # 모델의 행동이고 에이전트가 실제로 본 대화이므로 고치거나 빼지 않는다
        steps = []
        for turn in turns[:-1]:
            tool_calls = [ToolCall(tool=c["tool"], args=c["args"], result=c["result"]) for c in turn["tool_calls"]]
            steps.append(Step(response=strip_open_think(turn["response"]), tool_calls=tool_calls))
        item = result["item"]
        repo_path = f"{repo_root_name}/{item['repo'].split('/')[-1]}"
        trajectory = Trajectory(
            trajectory_id=item["index"],
            system_prompt=system_template.replace("{repo_path}", repo_path),
            user_prompt=ToolCallingAgent._initial_human_prompt(repo_path, item["question"]),
            tools=TOOL_SCHEMAS,
            repo_path=repo_path,
            question=item["question"],
            steps=steps,
            final_response=turns[-1]["response"],
            reference=item["answer"],
        )
        trajectories.append(trajectory)
    return trajectories


def load_trajectories(source: str) -> list[Trajectory]:
    """ "sft" 면 SFT 궤적, 아니면 run_agent.py 의 results 폴더 경로."""
    if source == "sft":
        return load_sft_trajectories()
    return load_own_trajectories(Path(source))
