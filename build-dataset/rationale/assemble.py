"""golden_step/ 의 정지 스텝·답과 generate.py 의 정지 서술을 모아 학습 데이터(TRL 대화 형식)를 만든다.

golden_step/ 이 답을 받은 궤적마다 학습 예시 하나:
    h == n  궤적 그대로 (Claude 의 최종 응답까지)
    h <  n  스텝 0..h-1 의 Claude 도구 호출 + 정지 응답 "z_h\\n\\n<finish>\\ny(s_h)\\n</finish>"
            (--no-teacher 이면 teacher 서술 대신, 그 답을 낸 강제 답변의 출력 그대로 "<finish> 앞 글 + <finish>…</finish>":
             teacher 서술이 필요한지 보는 비교 판)
y(s_h) 는 h 지점의 강제 답변 중 총점이 가장 높은 것이다 (golden_step/label.py).
정지 스텝·답·서술이 없는 궤적은 빼고, 학습 최대 길이를 넘는 예시도 뺀다.
결과는 `<output-dir>/<run-id>/datasets/<teacher 또는 no-teacher>/train.jsonl` (행마다 messages, tools) 이고,
train/ 가 그대로 읽는다.
"""

import argparse
import copy
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from dataset import Trajectory, load_trajectories
from golden_step.models import Run


def build_messages(run: Run, trajectory: Trajectory, rationale_dir: Path | None) -> list[dict] | None:
    """rationale_dir 가 None 이면 teacher 서술 대신 강제 답변의 <finish> 앞 글(reasoning)을 그대로."""
    h = run.halting_step
    if h is None:
        return None
    if h == trajectory.n:
        return trajectory.messages()
    if run.halting_answer is None:
        return None
    if rationale_dir is None:
        preface = find_reasoning(run)
    else:
        rationale_path = rationale_dir / f"{trajectory.trajectory_id:04d}-{h:02d}.txt"
        if not rationale_path.exists():
            return None
        preface = rationale_path.read_text()
    halting_response = "<finish>\n" + run.halting_answer + "\n</finish>"
    if preface:
        halting_response = preface + "\n\n" + halting_response
    return trajectory.prefix_messages(h) + [{"role": "assistant", "content": halting_response}]


def count_tokens(tokenizer, messages: list[dict], tools: list[dict]) -> int:
    """대화를 모델 템플릿으로 렌더한 토큰 수 (학습 최대 길이 거르기용).

    도구 호출 인자는 학습 때처럼 사전으로 풀어 렌더한다 (Gemma 4 템플릿은 사전만 받고, Qwen3 는 글과 같은 결과).
    """
    parsed = []
    for message in messages:
        if message.get("tool_calls"):
            message = copy.deepcopy(message)
            for tool_call in message["tool_calls"]:
                tool_call["function"]["arguments"] = json.loads(tool_call["function"]["arguments"])
        parsed.append(message)
    text = tokenizer.apply_chat_template(parsed, tools=tools, tokenize=False)
    return len(tokenizer.encode(text, add_special_tokens=False))


def find_reasoning(run: Run) -> str:
    """정지 응답에 넣을 답(golden_step/label.py 가 고른 답)을 낸 강제 답변의 <finish> 앞 글."""
    for answer in run.steps[run.halting_step].answers:
        if answer.answer == run.halting_answer:
            return answer.reasoning or ""
    raise ValueError(f"Trajectory {run.trajectory_id}: halting answer not found at step {run.halting_step}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble the stop training data from the golden step labels and the rationales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out"),
        help="output dir of generate.py",
    )
    parser.add_argument(
        "-r",
        "--run-id",
        type=str,
        required=True,
        help="run id (same as golden_step/)",
    )
    parser.add_argument(
        "--golden-step-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "golden_step" / "out"),
        help="output dir of golden_step/ (answers with halting steps)",
    )
    parser.add_argument(
        "--golden-step-run-id",
        type=str,
        default=None,
        help="run id of golden_step/ to read; default: --run-id (set it when an ablation writes under its own run id)",
    )
    parser.add_argument(
        "--no-teacher",
        action="store_true",
        help="halting turn = the forced answer output as is, without the teacher rationale (ablation)",
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default="sft",
        help="trajectories: sft, or a results dir of evaluate-agent/run_agent.py (the agent model's own rollouts)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="agent model whose tokenizer measures the example length",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=32768,
        help="drop examples longer than this many tokens (same limit as training)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "questions" / "out" / "split.json"),
        help="questions/split.py's split; own trajectories (--trajectories not sft) keep only its train questions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_dir = Path(args.output_dir).expanduser() / args.run_id
    answer_dir = Path(args.golden_step_dir).expanduser() / (args.golden_step_run_id or args.run_id) / "answers"
    rationale_dir = None if args.no_teacher else run_dir / "rationales"
    trajectories = {t.trajectory_id: t for t in load_trajectories(args.trajectories)}
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # expert(SFT) 궤적은 RL 문항이 아니라 split 을 적용하지 않는다
    train_ids = None
    if args.trajectories != "sft":
        train_ids = set(json.loads(Path(args.split).expanduser().read_text())["train"])

    rows = []
    num_missing, num_too_long, num_validation = 0, 0, 0
    lengths, halting_steps = [], []
    for path in sorted(answer_dir.glob("*.json")):
        run = Run.model_validate_json(path.read_text())
        #train_ids가 none이 아니고, traj이 train이 아니면 스킵
        if train_ids is not None and run.trajectory_id not in train_ids:
            num_validation += 1
            continue
        trajectory = trajectories[run.trajectory_id]
        messages = build_messages(run, trajectory, rationale_dir)
        if messages is None:
            num_missing += 1
            continue
        length = count_tokens(tokenizer, messages, trajectory.tools)
        if length > args.max_length:
            num_too_long += 1
            continue
        rows.append({"messages": messages, "tools": trajectory.tools, "trajectory_id": trajectory.trajectory_id})
        lengths.append(length)
        halting_steps.append(run.halting_step)

    out_path = run_dir / "datasets" / ("no-teacher" if args.no_teacher else "teacher") / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} examples to {out_path}")
    print(
        f"dropped: missing halting step, answer, or rationale = {num_missing}, too long = {num_too_long}, not in the train split = {num_validation}"
    )
    if rows:
        print(f"mean h = {statistics.mean(halting_steps):.2f}, median tokens = {int(statistics.median(lengths)):,}")


if __name__ == "__main__":
    main()
