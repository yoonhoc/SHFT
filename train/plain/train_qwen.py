"""Plain (Qwen3): golden step 에서 자른 궤적을 TRL 기본 SFT 손실 그대로 학습한다 (서술·halting weight 없음).

SHFT/train_qwen.py 와 같은 모델·어댑터·설정에서 손실만 TRL 기본(`chunked_nll`, 모든 assistant 토큰 같은 무게)이다.
SHFT 의 teacher 서술과 λ 가 하는 일을 보는 절제용이다. `--init-adapter` 를 주면 그 어댑터에서 이어서 하고,
주지 않으면 원본에 새 어댑터를 붙여 처음부터 한다.

입력: `--dataset-dir` 의 train.jsonl (`../build-dataset/rationale/assemble.py --no-teacher` 가 쓴 행: messages, tools)
출력: `plain/out/<run-name>/` — 에포크마다 어댑터 checkpoint-<step>, 실행 인자 train_args.json, train_stdout.log, wandb 기록

    cd code/train
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m plain.train_qwen \\
        -r <run name> --model Qwen/Qwen3-8B --dataset-dir ../build-dataset/rationale/out/<run-id>/datasets/no-teacher
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import SFTTrainer

from SHFT.train_qwen import LORA_TARGET_MODULES, prepare_output
from utils import (
    add_adapter_args,
    add_common_args,
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    save_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an adapter on the golden-step trajectories with the plain SFT loss",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser)
    add_adapter_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model, peft_config = attach_adapter(model, args, LORA_TARGET_MODULES)
    dataset = load_dataset(args.dataset_dir, split="train").map(parse_tool_arguments)
    trainer = SFTTrainer(
        args=build_sft_config(args, loss_type="chunked_nll"),
        model=model,
        processing_class=tokenizer,
        train_dataset=drop_long_rows(dataset, tokenizer, args.max_length),
        peft_config=peft_config,
    )
    if trainer.is_world_process_zero():
        save_args(args, output_dir / "train_args.json")
    trainer.train()


if __name__ == "__main__":
    main()
