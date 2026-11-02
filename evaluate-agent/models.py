from pydantic import BaseModel

from dataset import Item


class ToolCall(BaseModel):
    tool: str
    args: dict
    result: str
    execution_time: float


class Turn(BaseModel):
    """에이전트의 LLM 호출 하나. 도구를 부른 턴은 저자 궤적 그대로, 도구 없이 끝낸 턴은 run_agent.py 가 더한 것."""

    step: int
    # 모델 출력 원문 (정지 턴이면 <finish> 블록까지)
    response: str
    tool_calls: list[ToolCall]
    # 이 호출의 prompt_tokens, completion_tokens, total_tokens
    token_usage: dict[str, int]


class Result(BaseModel):
    item: Item
    # 이하 저자 ToolCallingAgent.query 의 결과. 끝내 실패하면 answer 가 "Error: <이유>", status 가 "failed"
    answer: str
    status: str
    stop_reason: str | None = None
    # 마지막 답 턴까지 센 LLM 호출 수. 실패하면 0
    steps_completed: int = 0
    max_iterations: int
    retry_attempts: int | None = None
    total_time: float | None = None
    error: str | None = None
    trajectory: list[Turn] = []
    # oracle/run.py 만 채운다: 정지를 억제한 실주행에서 모델이 처음 답하려 한 스텝 (끝까지 안 그랬으면 None)
    natural_stop: int | None = None
    # judge.py 가 채우는 항목별 점수 (SWE-QA-Pro 5 항목 × 1~10, SWE-QA 5 항목 × 1~20). 채점 전이면 None
    score: dict[str, int] | None = None

    @property
    def total(self) -> int | None:
        if self.score is None:
            return None
        return sum(self.score.values())
