"""Ministral 3 tokenizer 를 transformers·vLLM 이 제대로 읽는 형태로 저장한다.

공식 저장소의 tokenizer.json(tekken, byte-level BPE)은 그 자체로는 맞지만, tokenizer_config.json 이 `LlamaTokenizerFast` 를 가리켜
transformers 가 sentencepiece 식으로 열어 글을 잘못 자르고(“Where does” → W|her|ed|oes) 공백·줄바꿈을 잃은 채 되돌린다
(transformers 5.8, 2026-10-11 확인; `fix_mistral_regex=True` 는 자르기만 고치고 되돌리기는 못 고친다). vLLM 의 HF tokenizer 모드도
같은 경로를 쓴다. tokenizer_class 를 `PreTrainedTokenizerFast` 로 바꾸면 tokenizer.json 을 그대로 써서 공식 tokenizers 결과와
글 176개에서 자르기·되돌리기 모두 일치한다. 평가(vLLM `--tokenizer`)와 학습(`--tokenizer`)이 같은 폴더를 쓴다.

읽는 것: --model 의 tokenizer.json, tokenizer_config.json, chat_template.jinja, special_tokens_map.json
쓰는 것: --output-dir (기본 out/ministral-8b-tokenizer)

    cd code/train
    python -m mistral_tokenizer
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the Ministral 3 tokenizer as a PreTrainedTokenizerFast that transformers and vLLM read correctly",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="mistralai/Ministral-3-8B-Instruct-2512-BF16", help="HF model id")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "out" / "ministral-8b-tokenizer"),
        help="where to save the tokenizer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(snapshot_download(args.model, allow_patterns=FILES))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if name != "tokenizer_config.json" and (source / name).exists():
            shutil.copy(source / name, output_dir / name)
    config = json.loads((source / "tokenizer_config.json").read_text())
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    (output_dir / "tokenizer_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"saved {output_dir} (tokenizer_class PreTrainedTokenizerFast)")


if __name__ == "__main__":
    main()
