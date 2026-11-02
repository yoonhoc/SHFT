"""BATS 기준선 (Budget Tracker): 저자 에이전트에 남은 도구 호출 예산을 매 턴 알려 주는 것만 더해 실주행한다 (BATS 의 ReAct + Budget Tracker).

원 논문(liu2026budget) 저자 코드(third_party/budget-aware-agent, src/agent_budget_tracker.py 와 src/prompts_budget_tracker.py 의
prompt_v011_oct, 논문 부록 D.1 그림 11 과 같음)의 Budget Tracker 를 저자 에이전트에 옮긴다. 원 논문의 코딩 에이전트 실험(부록 C.3)처럼
도구마다가 아닌 전체 예산 하나를 두고, 검색·브라우즈 지시를 우리 도구(semantic_search, view_codebase, execute_readonly_command)에
맞게 바꾼 것 말고는 문구 그대로다.
- 예산 = 도구를 부를 수 있는 턴 수 (max_iterations - 1). 실행한 도구 호출 하나가 1 을 쓴다 (원 저자 코드도 query·url 하나마다 1;
  저자 scaffold 는 턴마다 호출 하나라 턴 수와 같고, 표의 Calls 지표와 같은 단위). 도구 없이 지나간 턴은 예산을 안 쓴다.
- system prompt 끝에 예산 지침(## Budget)을 붙인다 (원 저자 코드: system instruction 안의 ## Budget 절).
- 첫 질문 끝과 답하지 않은 턴마다 <budget> 블록을 붙인다: 도구를 부른 턴은 그 턴의 마지막 도구 결과 뒤에, 도구 없이 글만 쓴 턴은
  user 메시지 "No tool response." 뒤에 (원 저자 코드 agent_budget_tracker.py 443~458줄: 첫 질문 끝에 Used 0, 턴마다 <tool_response>
  또는 "No tool response." 뒤에 budget_template).
- 예산이 다 되면 저자 강제 답변 그대로 (원 저자 코드: 예산이 다 되면 멈춤). 원 저자 코드의 다시 탐색시키기(--flag_cont_gen)는
  Budget Tracker 실행 스크립트(scripts/run_budget_tracker.sh)에서 끄므로 쓰지 않는다. 디코딩은 다른 행과 같게 우리 설정(온도 0)을 쓴다.
에이전트는 SWE-QA-Pro 저자 ToolCallingAgent 를 바로 상속하고, 궤적은 run_agent.py 와 같게 도구 없이 끝낸 턴도 더한다.
나머지(모델 설정, vLLM 서버, 저장소 경로, 저장 형식, 인자)는 run_agent.py 그대로다.

읽는 것: dataset.load_items(--benchmark), --repo-root 밑의 저장소
쓰는 것: budget_tracker/out/<run-id>/<benchmark>/results/<index>.json (Result), 로그는 run_agent.py 와 같은 이름

    cd code/evaluate-agent
    CUDA_VISIBLE_DEVICES=0 python -m budget_tracker.run -r qwen3-8b-bats -b sweqapro -m qwen3-8b
"""

import argparse
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from sweqapro.agent import AgentState, ToolCallingAgent

from run_agent import AgentRunner, SandboxedTools, build_parser, record_response_turn, run_jobs

# 원 저자 prompt_v011_oct 의 "## Budget" 절. 예산 둘(Query, URL)을 하나로, Search·Browse 를 우리 도구로, <answer>None</answer> 을 <finish> 로
BUDGET_GUIDE = """

## Budget

You have one tool-call budget of {budget} tool calls.
Each tool call consumes 1 unit.

After each tool response, a <budget> tag shows remaining units.
You must ADAPT your strategy dynamically to the current budget state.

### HIGH Budget (>=70% remaining)
- Search: 3-5 diverse semantic_search or grep queries in one turn.
- View: up to 2-3 high-value files or line ranges.
- Goal: Broad exploration, build context fast.

### MEDIUM Budget (30%-70%)
- Search: 2-3 precise, refined queries per turn.
- View: 1-2 files or line ranges that close key knowledge gaps.
- Goal: Converge; eliminate uncertainty efficiently.

### LOW Budget (10%-30%)
- Search: 1 tightly focused query.
- View: at most 1 most promising file or line range.
- Goal: Verify a single critical fact or finalize answer.

### CRITICAL (<10% remaining)
- Only perform 1 minimal-cost search or view if absolutely essential.
- If uncertainty remains and no tool use is possible, give your best answer in the <finish> block.

Always justify tool choices based on the remaining budget.
"""
# 원 저자 agent_budget_tracker.py 의 budget_template (논문 3.1 절)
BUDGET_BLOCK = """

<budget>
Tool Call Budget Used: {used}, Tool Call Budget Remaining: {remaining}
Make the best use of the available resources.
</budget>"""


class BATSAgent(SandboxedTools, ToolCallingAgent):
    """SWE-QA-Pro 저자 에이전트에 Budget Tracker 를 붙인다. 예산은 state 의 tool_calls 로만 세서 문항끼리 섞이지 않는다."""

    @property
    def budget(self) -> int:
        # 저자 에이전트는 스텝 max_iterations - 1 에서 답을 강제하므로 도구를 부를 수 있는 턴은 그보다 하나 적다
        return self.max_iterations - 1

    def _system_prompt(self, repo_path: str) -> str:
        return super()._system_prompt(repo_path) + BUDGET_GUIDE.format(budget=self.budget)

    def _initial_human_prompt(self, repo_path: str, question: str) -> str:
        return super()._initial_human_prompt(repo_path, question) + BUDGET_BLOCK.format(used=0, remaining=self.budget)

    def _agent_step(self, state: AgentState) -> AgentState:
        step = state.get("current_step", 0)
        num_turns = len(state["trajectory"])
        state = super()._agent_step(state)
        record_response_turn(state, step, num_turns)
        if state.get("final_answer"):
            return state
        # 답하지 않은 턴마다 예산 블록을 붙인다 (대화 기록만 바꾸고 궤적 저장은 그대로): 도구를 부른 턴은 마지막 도구 결과 뒤에,
        # 도구 없이 글만 쓴 턴은 원 저자 코드처럼 user 메시지 "No tool response." 뒤에
        used = len(state["tool_calls"])
        block = BUDGET_BLOCK.format(used=used, remaining=max(self.budget - used, 0))
        interaction = state["history_manager"].history[-1]
        last = interaction[-1]
        if isinstance(last, ToolMessage):
            interaction[-1] = ToolMessage(content=str(last.content) + block, tool_call_id=last.tool_call_id)
        else:
            interaction.append(HumanMessage(content="No tool response." + block))
        return state


class BATSRunner(AgentRunner):
    def build_agent(self, base_url: str) -> None:
        self.build_llms(base_url)
        self.agent = BATSAgent(**self.agent_kwargs())


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    parser.set_defaults(output_dir=str(Path(__file__).resolve().parent / "out"))
    return parser.parse_args()


def main() -> None:
    run_jobs(BATSRunner(parse_args()), "budget_tracker")


if __name__ == "__main__":
    main()
