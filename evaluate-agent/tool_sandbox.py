"""도구 호출을 그 문항의 저장소 하나만 보이는 샌드박스(bubblewrap) 안에서 실행한다.

저자 도구는 모델이 준 경로 아래를 전부 걷고(semantic_search) 셸 명령도 돌린다(execute_readonly_command). 호스트에서 그대로 돌리면
"/" 나 "." 으로 서버 디스크 전체·다른 저장소·벤치마크 정답까지 닿는다. 그래서 도구 호출마다 bwrap 으로 가짜 루트를 만들어
시스템 바이너리(/usr 등), 파이썬, 저자 도구 패키지, 그리고 **그 문항의 저장소 하나**만 보이게 하고 그 안에서 tool_exec.py 를 띄운다.
에이전트 프로세스(LLM 호출·궤적 기록)는 밖에 있고, 도구 코드는 저자 것 그대로다. 작업 폴더와 저장소의 상대 경로("repos/<이름>")는
밖과 같게 두므로 결과 글의 경로도 같다. run_agent.py 의 Agent 가 저자 _run_tool 대신 이것을 쓴다.

    run_tool({"name": "semantic_search", "args": {...}}, "repos/qibo") -> 결과 글
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 샌드박스 안에서 돌릴 스크립트 (파일을 붙이는 대신 표준 입력으로 넣는다)
TOOL_EXEC = (HERE / "tool_exec.py").read_text(encoding="utf-8")
SWEQAPRO_DIR = HERE.parent / "third_party" / "SWE-QA-Pro" / "eval"
# 이 프로세스의 인터프리터 (venv 의 python 은 uv 가 설치한 인터프리터로의 심볼릭 링크라 실체를 쓴다)
PYTHON = Path(sys.executable).resolve()
HOME = Path.home()


def sandbox_args(repo: Path, cwd: Path) -> list[str]:
    args = [
        "bwrap",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/sbin",
        "/sbin",
        "--ro-bind",
        "/etc",
        "/etc",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--tmpfs",
        "/tmp",
        # 파이썬 인터프리터 (저자 도구는 표준 라이브러리만 쓴다)
        "--ro-bind",
        str(PYTHON.parents[1]),
        str(PYTHON.parents[1]),
        # 저자 도구 패키지 (tool_exec.py 는 표준 입력으로 넣어 안에 파일을 두지 않는다)
        "--ro-bind",
        str(SWEQAPRO_DIR / "sweqapro"),
        str(SWEQAPRO_DIR / "sweqapro"),
        # 작업 폴더(빈 폴더)와 그 안의 이 저장소 하나
        "--dir",
        str(cwd),
        "--ro-bind",
        str(repo),
        str(repo),
        "--chdir",
        str(cwd),
        "--unshare-net",
        "--die-with-parent",
    ]
    return args


def run_tool(call: dict, repo_path: str) -> str:
    """도구 호출 하나를 저장소 repo_path 만 보이는 샌드박스에서 실행하고 결과 글을 돌려준다."""
    cwd = Path.cwd()
    repo = (cwd / repo_path).resolve()
    if not repo.is_dir():
        return f"ERROR: Tool failed: repository not found: {repo_path}"
    request = json.dumps({"call": call, "repo_path": repo_path})
    command = sandbox_args(repo, cwd) + [str(PYTHON), "-", request]
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(SWEQAPRO_DIR),
        "HOME": str(HOME),
        "LANG": "C.UTF-8",
    }
    try:
        completed = subprocess.run(command, input=TOOL_EXEC, capture_output=True, text=True, env=env)
    except OSError as e:
        return f"ERROR: Tool failed: {e}"
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-1:] or ["no output"]
        return f"ERROR: Tool failed: {tail[0]}"
    return completed.stdout
