"""Sufficient Context 기준선 실주행: 도구 호출 턴마다 "지금까지의 context 가 질문에 답하기에 충분한가" 를 autorater 에게 묻고
그 판정을 기록하면서 저자 에이전트를 평소대로 돌린다.

Sufficient Context (Joren et al., ICLR 2025) 의 autorater prompt (부록 C.1, 1-shot) 를 그대로 쓰되, 예시 하나만 우리 도메인의 학습 문항
롤아웃으로 바꾼다 (autorater/example.py; 논문도 자기 데이터의 held-out 사례를 예시로 썼다). 그 논문은 autorater 를 단발 RAG 의
답변/abstain 결정에 썼고, "autorater 로 더 검색할지 답할지를 반복 판정하는 것" 을 future work 로 남겼다 — 그것이 이 기준선이다.
- rater 는 우리 표에서는 judge 와 같은 GPT-6 Luna 다 (`--rater-openai`; judge.py 와 같은 reasoning effort·Flex). 원 논문의 autorater 도
  frontier 모델(Gemini 1.5 Pro)이었다. "큰 모델을 inference 마다 부를 것인가(이 기준선), 학습 때 한 번만 쓸 것인가(SIS)" 의 비교다.
  `--rater-base-url` 을 주면 그 vLLM 서버의 모델(생각 모드 끔, 온도 0)이 rater 고, 둘 다 없으면 에이전트 모델 자신이 rater 다 (부록: 약한 rater).
- prompt 의 <question> 은 문항, <context> 는 지금까지의 대화 기록(저자 `_force_finish` 와 같은 flatten). 원 prompt 의
  timestamp 줄은 우리 질문에 해당이 없어 뺐다. 판정은 출력의 {"Sufficient Context": 0/1} 로 읽는다.
- 처음 충분하다고 판정한 스텝에서 저자 `_force_finish` 를 state 복사본에 그대로 적용해(학습 데이터의 강제 답변 golden_step/answer.py 와 같은 호출,
  같은 사유 문구, 저자의 재시도 경로 포함) 그 답으로 끝낸다 (정지 = 처음 충분하다고 한 스텝). 우리 결과 파일에는 stop_reason "sufficient context" 로 표시한다 (저자 query 는 Natural completion).
  저자 코드가 실패로 보는 출력(퇴화·<finish> 없음·도구 호출, RuntimeError)이면 원래 state 로 계속 탐색하며 다음 스텝을 다시 판정한다.
에이전트는 SWE-QA-Pro 저자 ToolCallingAgent 를 바로 상속하고, 궤적은 run_agent.py 와 같게 도구 없이 끝낸 턴도 더한다.
나머지(모델 설정, vLLM 서버, 저장소 경로, 저장 형식, 인자)는 run_agent.py 그대로다.

읽는 것: dataset.load_items(--benchmark), --repo-root 밑의 저장소
쓰는 것: autorater/out/<run-id>/<benchmark>/results/<index>.json (Result, 멈춘 결과), ratings/<index>.json (Ratings, 스텝별 판정)

    cd code/evaluate-agent
    CUDA_VISIBLE_DEVICES=0 python -m autorater.run -r qwen3-8b-sufficiency -b sweqapro -m qwen3-8b --rater-openai
    # 로컬 rater: CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3.8-27B --port 8102 --max-model-len 40960 --max-num-seqs 256 을 띄우고
    #   --rater-base-url http://localhost:8102/v1 --rater-model Qwen/Qwen3.8-27B
"""

import argparse
import copy
import logging
import re
import threading
import time
from pathlib import Path

import openai
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sweqapro.agent import AgentState, ToolCallingAgent

from dataset import Item
from score.judge import JUDGE_MODEL, REASONING_EFFORT, SERVICE_TIER, TIMEOUT
from run_agent import AgentRunner, SandboxedTools, build_parser, record_response_turn, run_jobs

# Joren et al. (2025) 부록 C.1 의 Sufficient Context autorater prompt. timestamp 줄만 뺐다
RATER_PROMPT = """You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES.
Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION. Include questions about any mathematical calculations or arithmetic that would be required. Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)
EXAMPLE:
{example}
Remember the instructions: You are an expert LLM evaluator that excels at evaluating a QUESTION and REFERENCES. Consider the following criteria:
Sufficient Context: 1 IF the CONTEXT is sufficient to infer the answer to the question and 0 IF the CONTEXT cannot be used to infer the answer to the question
First, output a list of step-by-step questions that would be used to arrive at a label for the criteria. Make sure to include questions about assumptions implicit in the QUESTION Include questions about any mathematical calculations or arithmetic that would be required. Next, answer each of the questions. Make sure to work step by step through any required mathematical calculations or arithmetic. Finally, use these answers to evaluate the criteria.
Output the ### EXPLANATION (Text). Then, use the EXPLANATION to output the ### EVALUATION (JSON)
### QUESTION
{question}
### REFERENCES
{context}
"""
# 1-shot 예시는 우리 도메인의 학습 문항 롤아웃으로 (autorater/example.py; 원 논문은 자기 데이터의 백과사전 문단 예시).
# format 의 중괄호와 겹치지 않게 예시의 중괄호는 두 겹으로 만든다
EXAMPLE = (Path(__file__).resolve().parent / "example.txt").read_text(encoding="utf-8").strip()
RATER_PROMPT = RATER_PROMPT.replace("{example}", EXAMPLE.replace("{", "{{").replace("}", "}}"))
VERDICT_PATTERN = re.compile(r'"Sufficient Context"\s*:\s*([01])')
# 강제 답변의 사유 문구. 학습 데이터의 강제 답변(golden_step/answer.py)과 같다
FORCE_REASON = "STOP AND ANSWER NOW"
RATER_ATTEMPTS = 60  # 429 가 한 시간 넘게 이어지면 포기

logger = logging.getLogger(__name__)


class Rating(BaseModel):
    # 판정까지 한 도구 호출 턴 수
    step: int
    # autorater 판정. 출력에서 JSON 을 못 읽으면 None
    sufficient: bool | None
    rater_output: str
    # 충분하다고 한 스텝의 강제 답변 (저자 _force_finish 가 실패하면 None). 불충분이면 받지 않아 None
    answer: str | None = None
    token_usage: dict[str, int]
    # rater 호출 한 번의 벽시계 시간(초). 재시도가 있었으면 마지막(성공한) 호출만
    seconds: float | None = None


class Ratings(BaseModel):
    index: int
    ratings: list[Rating]


def force_finish(agent: ToolCallingAgent, state: AgentState) -> AgentState | None:
    """저자 _force_finish 를 state 의 복사본에 그대로 적용한다 (golden_step/answer.py 의 강제 답변과 같은 호출, 재시도 경로 포함).

    저자 코드가 실패로 보는 출력(퇴화, <finish> 없음, 도구 호출)이면 RuntimeError 가 나므로 None 을 돌려주고, 원래 state 는 그대로다.
    """
    trial = copy.deepcopy(state)
    try:
        return agent._force_finish(trial, FORCE_REASON)
    except RuntimeError as error:
        logger.info("Forced finish failed at step %d: %s", state["current_step"], error)
        return None


def should_retry(error: openai.APIError) -> bool:
    """잠시 뒤 풀리는 오류만 다시 시도한다: 연결·시간 초과, 429, 5xx. 다른 4xx(프롬프트가 너무 긺 등)는 바로 실패."""
    if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(error, openai.APIStatusError):
        return error.status_code == 429 or error.status_code >= 500
    return False


def parse_verdict(text: str) -> bool | None:
    """출력의 마지막 {"Sufficient Context": 0/1} 을 읽는다 (생각 블록 안의 것보다 뒤의 최종 JSON 을 택한다)."""
    matches = VERDICT_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1] == "1"


class SufficiencyAgent(SandboxedTools, ToolCallingAgent):
    """저자 에이전트 그대로 돌리면서, 도구 호출 턴마다 autorater 판정을 문항(질문 글)별로 모은다."""

    def __init__(self, rater, **kwargs) -> None:
        super().__init__(**kwargs)
        # 충분성을 판정하는 LLM. 별도 서버의 큰 모델이거나 (기본) 에이전트 자신(llm_no_tools)
        self.rater = rater
        self.ratings: dict[str, list[Rating]] = {}
        self.lock = threading.Lock()

    def rate(self, state: AgentState) -> Rating:
        """rater 로 지금까지의 context 가 충분한지 묻는다. 상태는 바꾸지 않는다.
        429(Flex 자원 부족)·5xx·연결 오류는 문항을 잃지 않도록 기다렸다가 다시 묻는다 (최대 RATER_ATTEMPTS 번, 1시간 남짓)."""
        prompt = RATER_PROMPT.format(question=state["question"], context=state["history_manager"].flatten())
        messages = [HumanMessage(content=prompt)]
        response = None
        for attempt in range(RATER_ATTEMPTS):
            try:
                started = time.perf_counter()
                response = self.rater.invoke(messages)
                seconds = time.perf_counter() - started
                break
            except openai.APIError as error:
                if not should_retry(error):
                    raise
                logger.warning(
                    "Rater error at step %d (attempt %d/%d): %s",
                    state["current_step"],
                    attempt + 1,
                    RATER_ATTEMPTS,
                    error,
                )
                time.sleep(min(60, 2**attempt))
        if response is None:
            raise ValueError(f"Rater failed after {RATER_ATTEMPTS} attempts")
        output = self._content_text(response.content)
        return Rating(
            step=state["current_step"],
            sufficient=parse_verdict(output),
            rater_output=output,
            token_usage=self._extract_token_usage(response) or {},
            seconds=seconds,
        )

    def _agent_step(self, state: AgentState) -> AgentState:
        step = state.get("current_step", 0)
        num_turns = len(state["trajectory"])
        # 도구를 한 번 이상 쓴 뒤부터, 저자가 답을 강제하는 마지막 턴 전까지
        if 1 <= step < self.max_iterations - 1:
            rating = self.rate(state)
            finished = None
            if rating.sufficient:
                finished = force_finish(self, state)
            if finished is not None:
                rating.answer = finished["final_answer"]
            with self.lock:
                self.ratings.setdefault(state["question"], []).append(rating)
            if finished is not None:
                record_response_turn(finished, step, num_turns)
                return finished
        state = super()._agent_step(state)
        record_response_turn(state, step, num_turns)
        return state

    def pop_ratings(self, question: str) -> list[Rating]:
        """문항의 판정들. 저자 query 가 다시 시도해 스텝이 처음부터 다시 돌았으면 마지막 시도의 것만 남긴다."""
        with self.lock:
            ratings = self.ratings.pop(question, [])
        last_start = 0
        for idx in range(1, len(ratings)):
            if ratings[idx].step <= ratings[idx - 1].step:
                last_start = idx
        return ratings[last_start:]


class SufficiencyRunner(AgentRunner):
    def build_rater(self):
        """--rater-openai 면 OpenAI 의 모델(judge 와 같은 reasoning effort·Flex·제한 시간), --rater-base-url 이 있으면 그 vLLM 서버의
        모델(생각 모드 끔, 온도 0), 둘 다 없으면 에이전트 자신."""
        if self.args.rater_openai:
            return ChatOpenAI(
                model=self.args.rater_model or JUDGE_MODEL,
                reasoning_effort=REASONING_EFFORT,
                service_tier=SERVICE_TIER,
                timeout=TIMEOUT,
                max_retries=10,
            )
        if self.args.rater_base_url is None:
            return self.llm_no_tools
        return ChatOpenAI(
            api_key="EMPTY",
            base_url=self.args.rater_base_url,
            model=self.args.rater_model or "Qwen/Qwen3.8-27B",
            temperature=0.0,
            max_tokens=self.args.rater_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    def build_agent(self, base_url: str) -> None:
        self.build_llms(base_url)
        self.agent = SufficiencyAgent(rater=self.build_rater(), **self.agent_kwargs())
        (self.run_dir / "ratings").mkdir(exist_ok=True)

    def process_job(self, item: Item) -> None:
        output = self.agent.query(question=item.question, repo_path=str(self.repo_path(item)))
        result = self.build_result(item, output)
        ratings = Ratings(index=item.index, ratings=self.agent.pop_ratings(item.question))
        # 마지막 판정이 "충분" 이고 거기서 끝났으면 rater 가 멈춘 것이다 (저자 query 는 Natural completion 으로 본다).
        # 우리 결과 파일의 표시일 뿐이고 저자 에이전트의 동작·답은 그대로다
        if (
            ratings.ratings
            and ratings.ratings[-1].answer is not None
            and result.steps_completed == ratings.ratings[-1].step + 1
        ):
            result = result.model_copy(update={"stop_reason": "sufficient context"})
        (self.run_dir / "ratings" / f"{item.index:03d}.json").write_text(
            ratings.model_dump_json(indent=2), encoding="utf-8"
        )
        (self.result_dir / f"{item.index:03d}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.set_defaults(output_dir=str(Path(__file__).resolve().parent / "out"))
    parser.add_argument(
        "--rater-openai",
        action="store_true",
        help=f"rate with an OpenAI model (default {JUDGE_MODEL}, the judge; OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--rater-base-url",
        type=str,
        default=None,
        help="vLLM server of the sufficiency rater; if neither this nor --rater-openai is set, the agent model itself rates",
    )
    parser.add_argument(
        "--rater-model",
        type=str,
        default=None,
        help=f"rater model name (default: {JUDGE_MODEL} with --rater-openai, Qwen/Qwen3.8-27B with --rater-base-url)",
    )
    parser.add_argument("--rater-max-tokens", type=int, default=2048, help="max tokens of a rating (vLLM rater only)")
    return parser.parse_args()


def main() -> None:
    run_jobs(SufficiencyRunner(parse_args()), "autorater")


if __name__ == "__main__":
    main()
