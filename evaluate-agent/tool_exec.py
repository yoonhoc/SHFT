"""샌드박스 안에서 도구 호출 하나를 실행한다 (tool_sandbox.py 가 이 글을 표준 입력으로 bwrap 안의 파이썬에 넣는다).

첫 인자로 {"call": {"name", "args"}, "repo_path"} JSON 을 받아 저자 도구를 그대로 부르고, 결과 글을 표준 출력으로 돌려준다.
분기는 저자 ToolCallingAgent._run_tool 과 같다 (저자 agent 모듈은 langchain 까지 끌어와 import 가 무거워 도구만 import 한다).
"""

import json
import sys

from sweqapro.tools import execute_readonly_command, semantic_search, view_codebase


def run_tool(call: dict, repo_path: str) -> str:
    name = call.get("name")
    args = call.get("args", {}) or {}
    try:
        if name == "semantic_search":
            result = semantic_search(
                term=args["term"],
                path=args.get("path", repo_path),
                python_only=args.get("python_only", False),
                max_files=args.get("max_files", 100),
                include_lines=args.get("include_lines", False),
            )
            return json.dumps(result)
        if name == "view_codebase":
            return view_codebase(
                path=args["path"],
                view_range=args.get("view_range"),
                concise=args.get("concise", False),
                python_only=args.get("python_only", False),
            )
        if name == "execute_readonly_command":
            return execute_readonly_command(args["command"])
        return f"ERROR: Unknown tool {name}"
    except Exception as e:
        return f"ERROR: Tool failed: {e}"


def main() -> None:
    request = json.loads(sys.argv[1])
    sys.stdout.write(run_tool(request["call"], request["repo_path"]))


if __name__ == "__main__":
    main()
