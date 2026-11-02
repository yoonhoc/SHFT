"""SFT 기준선의 학습 데이터: SWE-QA-Pro SFT 궤적을 통째로 TRL 대화 형식(messages, tools) 행으로 바꾼다.

공개 데이터는 도구 호출과 결과를 `tool_call` / `tool_response` 역할의 JSON 글로 적어 두어 채팅 템플릿이 읽지 못한다.
assistant 메시지의 `tool_calls` 와 `tool` 역할 메시지로 바꾼다. 그 밖(system·user 메시지, 도구 스키마, 최종 응답)은 원본 그대로다.

궤적을 읽고 대화로 되돌리는 코드는 build-dataset 과 같은 것을 쓴다 (`code/build-dataset/dataset.py`).
학습 최대 길이를 넘는 행은 모델 토크나이저마다 다르므로 여기서는 거르지 않고, 학습 파일이 `utils.drop_long_rows` 로 뺀다.

    cd code/train
    python -m sft.convert_data               # → train/sft/sftdata/train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "build-dataset"))

from dataset import load_sft_trajectories  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the SWE-QA-Pro SFT trajectories into training rows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "sftdata"),
        help="output dir (train.jsonl is written here)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_sft_trajectories()
    with (out_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        for trajectory in trajectories:
            row = {
                "messages": trajectory.messages(),
                "tools": trajectory.tools,
                "trajectory_id": trajectory.trajectory_id,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(trajectories)} rows to {out_dir / 'train.jsonl'}")


if __name__ == "__main__":
    main()
