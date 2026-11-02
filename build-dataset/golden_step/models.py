from pydantic import BaseModel


class Answer(BaseModel):
    # 모델 출력에서 <finish> 앞부분 그대로 (생각 블록이 올 수도, 답 앞 서술이 올 수도, 비어 있을 수도 있다)
    reasoning: str | None = None
    # 학습 전 에이전트 모델의 강제 답변 (<finish> 안쪽). 저자 코드가 실패로 판정하면 None
    answer: str | None
    # answer 가 None 인 이유: 저자 코드의 실패 판정(퇴화·<finish> 없음·도구 호출) 또는 문맥 한도 초과
    error: str | None = None
    # judge.py 가 채우는 항목별 점수 (저자 채점 프롬프트: 5 항목 × 1~10). 채점 전이거나 결과를 못 읽으면 None.
    # answer 가 None 이면 저자 러너처럼 "Error: <error>" 를 채점한 점수 (문맥 한도 초과는 채점하지 않음)
    score: dict[str, int] | None = None

    @property
    def total(self) -> int | None:
        if self.score is None:
            return None
        return sum(self.score.values())


class Step(BaseModel):
    # 지점 번호 i: 궤적의 스텝 0..i-1 을 본 뒤 (0 이면 도구를 쓰기 전, n 이면 궤적을 끝까지 본 뒤)
    step: int
    # 이 지점에서 받은 강제 답변들
    answers: list[Answer]


class Run(BaseModel):
    trajectory_id: int
    # 지점 0..n 의 강제 답변들 (steps[i].step == i)
    steps: list[Step]
    # label.py 가 채운다: 정지 스텝 i* 와 그 자리의 정지 응답에 넣을 답
    halting_step: int | None = None
    halting_answer: str | None = None
