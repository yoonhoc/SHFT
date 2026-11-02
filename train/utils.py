import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer
from trl import SFTConfig


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """sft/·SHFT/·das/ 의 학습 파일이 같이 쓰는 인자."""
    parser.add_argument("-r", "--run-name", type=str, required=True, help="run name (wandb)")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="output dir of the adapter checkpoints; default <sft|sis|das>/out/<run-name>",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="agent model (HF model id or path)")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="tokenizer to use instead of the model's (Ministral: mistral_tokenizer.py output)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="dir with train.jsonl (rows of messages and tools)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="training chat template with {%% generation %%} markers, for models TRL has none for",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--num-train-epochs", type=int, default=4, help="epochs")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="weight decay")
    parser.add_argument("--max-length", type=int, default=32768, help="max tokens of a training row")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="if set, stop after this many steps (for a quick test run)"
    )
    parser.add_argument("--attn", type=str, default="flash_attention_2", help="attention implementation")


def resolve_output_dir(args: argparse.Namespace, task_dir: Path) -> None:
    """출력 디렉터리를 안 주면 학습 폴더(sft/·SHFT/·das/) 밑 out/<run-name> 에 저장한다. 학습 로그도 여기에 둔다."""
    if args.output_dir is None:
        args.output_dir = str(task_dir / "out" / args.run_name)


def save_args(args: argparse.Namespace, path: Path) -> None:
    """실행 인자를 json 으로 남긴다 (같은 학습을 다시 돌릴 때 설정 확인용)."""
    path.write_text(json.dumps(vars(args), indent=2))


class Tee:
    """sys.stdout·sys.stderr 자리에 끼워, 쓰는 글을 원래 스트림과 파일에 같이 쓴다."""

    def __init__(self, stream: TextIO, file: TextIO):
        self.stream = stream
        self.file = file

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def tee_output(path: str | Path) -> None:
    """stdout·stderr (진행 막대, print, 오류) 를 화면과 path 에 같이 쓴다.

    nohup 으로 띄워도 실행 폴더에서 진행 상황을 볼 수 있게 한다. 실행 명령에서 따로 리다이렉트하지 않아도 된다.
    """
    file = open(path, "a", encoding="utf-8")
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, file)
    sys.stderr = Tee(sys.stderr, file)
    # import 때 이미 원래 stderr 를 잡은 로그 핸들러(transformers 등)도 파일에 쓰게 바꾼다
    loggers = [logging.getLogger()]
    for item in logging.root.manager.loggerDict.values():
        if isinstance(item, logging.Logger):
            loggers.append(item)
    for item in loggers:
        for handler in item.handlers:
            if type(handler) is logging.StreamHandler and handler.stream is original_stderr:
                handler.setStream(sys.stderr)


def set_local_device() -> None:
    # 각 rank 가 자기 GPU 를 먼저 잡는다 (peft 의 기본 적재 장치는 항상 0번 GPU)
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def add_adapter_args(parser: argparse.ArgumentParser) -> None:
    """SHFT/ 의 학습 파일이 같이 쓰는 어댑터 인자: SFT 기준선 어댑터에서 이어 학습하거나, 새 어댑터를 붙인다."""
    parser.add_argument(
        "--init-adapter",
        type=str,
        default=None,
        help="SFT baseline adapter to continue training; if not set, train a new adapter on the base model",
    )
    parser.add_argument("--lora-rank", type=int, default=32, help="LoRA rank (new adapter only)")
    parser.add_argument("--lora-alpha", type=float, default=64, help="LoRA alpha (new adapter only)")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout (new adapter only)")


def attach_adapter(model, args: argparse.Namespace, target_modules) -> tuple[object, LoraConfig | None]:
    """--init-adapter 가 있으면 그 어댑터를 올린 모델을 돌려주고 (peft_config 는 None), 없으면 새 LoraConfig 를 돌려준다."""
    if args.init_adapter is not None:
        model = PeftModel.from_pretrained(
            model,
            args.init_adapter,
            is_trainable=True,
            torch_device=f"cuda:{torch.cuda.current_device()}",
        )
        return model, None
    # SFT 기준선과 같은 새 어댑터 (sft/train_*.py)
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    return model, peft_config


def load_tokenizer(name: str, chat_template: str | None):
    """TRL 이 학습용 템플릿을 가진 모델(Qwen3 등)은 TRL 이 알아서 바꿔 끼우고, 없는 모델은 chat_template 파일을 쓴다.

    name 은 보통 모델 id 다. Ministral 3 은 transformers 가 공식 저장소의 tokenizer 를 잘못 열므로 mistral_tokenizer.py 가 저장한
    폴더를 --tokenizer 로 준다 (평가의 vLLM 도 같은 폴더).
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if chat_template is not None:
        with open(chat_template) as f:
            tokenizer.chat_template = f.read()
    return tokenizer


def parse_tool_arguments(row: dict) -> dict:
    """도구 호출 인자를 JSON 글에서 사전으로 푼다 (dataset.map 용).

    평가 때 vLLM 이 템플릿에 넘기기 전에 하는 것과 같다. Gemma 4 템플릿은 사전만 받고,
    Qwen3 템플릿은 글과 사전을 똑같이 렌더링한다 (SFT 궤적 1,000행에서 확인).
    """
    messages = row["messages"]
    for message in messages:
        for tool_call in message.get("tool_calls") or []:
            arguments = tool_call["function"]["arguments"]
            if isinstance(arguments, str):
                tool_call["function"]["arguments"] = json.loads(arguments)
    return {"messages": messages}


def drop_long_rows(dataset, tokenizer, max_length: int):
    """최대 길이를 넘는 행은 자르지 않고 뺀다 (자르면 최종 응답이 사라진다, 저자 조리법의 truncation_strategy delete)."""

    def fits(row: dict) -> bool:
        text = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
        return len(tokenizer.encode(text, add_special_tokens=False)) <= max_length

    kept = dataset.filter(fits)
    print(f"[train] {len(kept)} rows ({len(dataset) - len(kept)} longer than {max_length} tokens dropped)", flush=True)
    return kept


def build_sft_config(args: argparse.Namespace, loss_type: str) -> SFTConfig:
    return SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_length=args.max_length,
        assistant_only_loss=True,
        loss_type=loss_type,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        # transformers 5 는 1 미만 소수를 비율로 읽는다
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=True,
        use_liger_kernel=False,
        shuffle_dataset=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=1,
        save_only_model=True,
        save_strategy="epoch",
        report_to="wandb",
    )
