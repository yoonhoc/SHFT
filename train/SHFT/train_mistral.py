"""우리 방법, Ministral 3 판: SHFT/train_qwen.py 와 같은 손실(SHFT/halting.py, --stop-weight) 이고 모델만 다르게 올린다.

Ministral-3-8B-Instruct 는 vision 을 붙인 `Mistral3ForConditionalGeneration` 이라 `AutoModelForCausalLM` 으로는 안 열리고
`AutoModelForImageTextToText` 로 연다 (글만 넣으면 언어 모델만 돈다). 어댑터는 언어 모델의 선형층에만 붙인다 (vision·projector 제외).
이 모델 클래스는 `accepts_loss_kwargs` 가 없어 Trainer 가 누적 배치의 토큰 수를 안 넘기므로 (SHFT/train_gemma.py 와 같이) trainer 에서
플래그를 켠다. 어텐션은 헤드 128 이라 flash-attn 2 그대로다. TRL 에 학습용 템플릿이 없어 `--chat-template templates/ministral3_training.jinja`
(원본 템플릿에 {% generation %} 표식만 넣은 판) 를 준다. tokenizer 는 mistral_tokenizer.py 가 저장한 폴더를 --tokenizer 로 준다.
가중치는 Mistral 의 공식 BF16 배포본 (mistralai/Ministral-3-8B-Instruct-2512-BF16).

입력·출력·인자는 SHFT/train_qwen.py 와 같다.

    cd code/train
    CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m SHFT.train_mistral \\
        -r <run name> --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --tokenizer out/ministral-8b-tokenizer \\
        --chat-template templates/ministral3_training.jinja \\
        --dataset-dir ../build-dataset/rationale/out/<run-id>/datasets/teacher --learning-rate 5e-4 --stop-weight 5 \\
        --gradient-accumulation-steps 16
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText

from SHFT.halting import HaltingTrainer
from SHFT.train_qwen import parse_args, prepare_output
from utils import (
    attach_adapter,
    build_sft_config,
    drop_long_rows,
    load_tokenizer,
    parse_tool_arguments,
    save_args,
)

# 언어 모델의 선형층만 (vision tower 와 projector 에도 같은 이름의 층이 있다)
LORA_TARGET_MODULES = r"^(?!.*(vision|projector)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def main() -> None:
    args = parse_args()
    output_dir = prepare_output(args, __file__)

    tokenizer = load_tokenizer(args.tokenizer or args.model, args.chat_template)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn)
    model, peft_config = attach_adapter(model, args, LORA_TARGET_MODULES)
    dataset = load_dataset(args.dataset_dir, split="train").map(parse_tool_arguments)
    trainer = HaltingTrainer(
        stop_weight=args.stop_weight,
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
