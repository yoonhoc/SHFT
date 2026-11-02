"""Sufficiency Rater 의 1-shot 예시를 우리 도메인(저장소 QA)의 학습 문항에서 만든다.

Sufficient Context (Joren et al., ICLR 2025) 의 autorater prompt 는 지시문 + 예시 하나(1-shot) 이고, 그 예시는 저자 데이터에서 뗀 사례다
(백과사전 문단). 우리 rater 가 보는 context 는 도구 호출 대화 기록이라, 예시만 같은 형식의 사례로 바꾼다 (지시문은 원문 그대로).
예시는 손으로 짓지 않고 RL 학습 문항(평가셋 아님)의 Base 롤아웃 prefix 를 그대로 쓴다: 질문 420 (estimagic), 도구 호출 4개
(find 둘 → semantic_search "AVAILABLE_ALGORITHMS" → view_codebase optimization/__init__.py) 뒤가 "충분(1)" 인 지점.
강제 답변 점수가 3스텝 29.7 → 4스텝 42 (50점 만점) 로 뛰는 자리라, 몇 번 만에 충분해지는지가 아니라 핵심 파일을 봤는지가 기준임을 보인다
(너무 이른 지점의 예시는 조기 정지를 유도할 수 있어 피했다).
context 글은 rater 가 실제로 받는 형식(대화 기록의 LangChain 메시지 목록 repr, 저자 _force_finish 와 같음)으로 만든다.

읽는 것: --results 의 <question>.json (run_agent.py 의 Result)
쓰는 것: autorater/example.txt (run.py 가 import 때 읽어 RATER_PROMPT 에 넣는다)

    cd code/evaluate-agent
    python -m autorater.example
"""

import argparse
import json
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

HERE = Path(__file__).resolve().parent
QUESTION_ID = 420
HALTING_STEP = 4
# 논문 예시의 설명 문체 (무엇이 쓰여 있고, 그래서 답을 추론할 수 있다)
EXPLANATION = (
    "The first two commands locate the optimizer modules, the search for `AVAILABLE_ALGORITHMS` points to "
    "`estimagic/optimization/__init__.py`, and the view of that file shows the mechanism: `COLLECTED_FUNCTIONS` is built "
    "with `inspect.getmembers(..., inspect.isfunction)` from `scipy_optimizers`, and the other optimizer modules are added "
    "under the `IS_*_INSTALLED` flags imported from `estimagic.config`. The question asks how the dictionary is built from "
    "multiple modules with conditional dependency checking, and this file shows exactly that. Therefore, the context does "
    "provide a precise answer to the question."
)


def build_context(result: dict, halting_step: int) -> str:
    """스텝 0..halting_step-1 의 assistant·tool 메시지 목록을 rater 가 받는 형식(repr)으로."""
    messages = []
    for turn in result["trajectory"][:halting_step]:
        calls = []
        for idx, call in enumerate(turn["tool_calls"]):
            calls.append(
                {"name": call["tool"], "args": call["args"], "id": f"call_{turn['step']}_{idx}", "type": "tool_call"}
            )
        messages.append(AIMessage(content=turn["response"], tool_calls=calls))
        for idx, call in enumerate(turn["tool_calls"]):
            messages.append(ToolMessage(content=call["result"], tool_call_id=f"call_{turn['step']}_{idx}"))
    return str(messages)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the one-shot example of the Sufficiency Rater from a training-question rollout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(HERE.parent / "out" / "qwen3-8b-nothink-own" / "sweqapro-rl" / "results"),
        help="results dir of the rollout the example is taken from",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(HERE / "example.txt"), help="where to write the example"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads((Path(args.results).expanduser() / f"{QUESTION_ID:03d}.json").read_text())
    example = (
        "### QUESTION\n"
        f"{result['item']['question']}\n"
        "### References\n"
        f"{build_context(result, HALTING_STEP)}\n"
        "### EXPLANATION\n"
        f"{EXPLANATION}\n"
        "### JSON\n"
        '{"Sufficient Context": 1}'
    )
    Path(args.output).expanduser().write_text(example + "\n", encoding="utf-8")
    print(f"saved {args.output} ({len(example)} chars)")


if __name__ == "__main__":
    main()
