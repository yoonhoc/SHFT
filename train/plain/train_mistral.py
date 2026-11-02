"""Plain, Ministral 3 판: plain/train_qwen.py 와 같은 TRL 기본 SFT 손실이고 모델만 SHFT/train_mistral.py 처럼 올린다.

Ministral-3-8B-Instruct 는 `AutoModelForImageTextToText` 로 열고, 어댑터는 언어 모델의 선형층에만 붙인다.
이 모델 클래스는 `accepts_loss_kwargs` 가 없어 trainer 에서 플래그를 켠다 (SHFT/train_mistral.py 와 같다).

입력·출력·인자는 plain/train_qwen.py 와 같다.

    cd code/train
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m plain.train_mistral \\
        -r <run name> --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --tokenizer out/ministral-8b-tokenizer \\
        --chat-template templates/ministral3_training.jinja \\
        --dataset-dir ../build-dataset/rationale/out/<run-id>/datasets/no-teacher
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText
from trl import SFTTrainer

from plain.train_qwen import parse_args
from SHFT.train_mistral import LORA_TARGET_MODULES
from SHFT.train_qwen import prepare_output
from utils import (
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    save_args,
)


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model, peft_config = attach_adapter(model, args, LORA_TARGET_MODULES)
    dataset = load_dataset(args.dataset_dir, split="train").map(parse_tool_arguments)
    trainer = SFTTrainer(
        args=build_sft_config(args, loss_type="chunked_nll"),
        model=model,
        processing_class=tokenizer,
        train_dataset=drop_long_rows(dataset, tokenizer, args.max_length),
        peft_config=peft_config,
    )
    # Trainer 가 누적 배치의 토큰 수를 손실에 넘기게 한다 (모듈 docstring)
    trainer.model_accepts_loss_kwargs = True
    if trainer.is_world_process_zero():
        save_args(args, output_dir / "train_args.json")
    trainer.train()


if __name__ == "__main__":
    main()
