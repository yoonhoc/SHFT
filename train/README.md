# train

LoRA training of the agents. The SFT baseline, Plain, and SHFT use the same adapter and optimizer settings and differ
only in their training data and loss.

```
train/
  utils.py  deepspeed_zero2.yaml        shared arguments, SFT config, data loading
  templates/ministral3_training.jinja    Ministral chat template with {% generation %} markers
  mistral_tokenizer.py                   writes the Ministral tokenizer that transformers and vLLM read correctly
  sft/                                   SFT baseline on the full Expert trajectories
    convert_data.py  train_qwen.py  train_mistral.py  run.sh
  plain/                                 Plain: halted trajectories without the rationale, plain SFT loss
    train_qwen.py  train_mistral.py
  SHFT/                                  SHFT: halted trajectories with the rationale, halting weight λ
    halting.py  train_qwen.py  train_mistral.py
```

Run the modules from this folder. Each training run writes one adapter per epoch to
`<sft|plain|SHFT>/out/<run-name>/checkpoint-<step>`, with the arguments in `train_args.json`, the progress in
`train_stdout.log`, and the Weights & Biases log in `wandb/` (set `WANDB_MODE=offline` to skip the account).

## Shared settings

| Setting | Value |
|---|---|
| Adapter | LoRA r 32, α 64, dropout 0.1, on the linear layers of the language model (`LORA_TARGET_MODULES`) |
| Optimizer | learning rate 5e-4, cosine schedule, warmup 0.05, weight decay 0.05, bf16 |
| Epochs and batch | 4 epochs, effective batch 16 (1 row per GPU × GPUs × gradient accumulation) |
| Length | rows longer than 32,768 tokens are dropped, not truncated |
| Loss | assistant tokens only (TRL `SFTTrainer`, `assistant_only_loss`, default `chunked_nll` loss) |

Tool-call arguments are parsed from JSON strings into dictionaries before training, as vLLM does at evaluation time.
With `--gradient-accumulation-steps`, keep the effective batch at 16: for example 1 GPU with 16, 2 GPUs with 8, or
8 GPUs with 2. An epoch is then 63 steps on the Expert data and 25 steps on the Self data.

## Ministral-3-8B

Ministral-3-8B-Instruct is a vision-language model, so the trainers open it with `AutoModelForImageTextToText` and put
the adapter only on the language model. Its official `tokenizer_config.json` makes transformers split and decode text
wrongly, so write a fixed tokenizer once and use it for training (`--tokenizer`) and evaluation (`models.yaml`):

```bash
python -m mistral_tokenizer     # → out/ministral-8b-tokenizer
```

TRL has no training template for Ministral, so pass `--chat-template templates/ministral3_training.jinja`, the original
template with `{% generation %}` markers added. Qwen3 needs neither option.

## 1. SFT baseline (`sft/`)

```bash
python -m sft.convert_data      # Expert trajectories → sft/sftdata/train.jsonl
CUDA_VISIBLE_DEVICES=0,1 bash sft/run.sh qwen3 ministral
```

`sft/convert_data.py` turns the 1,000 Expert trajectories into chat rows (`messages`, `tools`). The released data
writes tool calls and results as JSON text in `tool_call` / `tool_response` roles, which chat templates cannot read,
so they become assistant `tool_calls` and `tool` messages. Everything else is kept as released.

`sft/run.sh` trains the given models one after another on the GPUs in `CUDA_VISIBLE_DEVICES`, setting gradient
accumulation to 16 divided by the number of GPUs. The run names are `<model>-sft-$SUFFIX` (default `rerun`). The single
command for one model is:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m sft.train_qwen \
    -r qwen3-sft --model Qwen/Qwen3-8B --dataset-dir sft/sftdata --learning-rate 5e-4 --gradient-accumulation-steps 16
```

## 2. Plain (`plain/`)

Plain trains on the halted trajectories without the rationale (`datasets/no-teacher` of `../build-dataset`) with the
same loss as SFT.

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m plain.train_qwen \
    -r qwen3-self-plain --dataset-dir ../build-dataset/rationale/out/qwen3-8b-nothink-own/datasets/no-teacher \
    --learning-rate 5e-4 --gradient-accumulation-steps 16
```

## 3. SHFT (`SHFT/`)

SHFT trains on the halted trajectories with the rationale (`datasets/teacher`). The loss (`SHFT/halting.py`) is
cross-entropy with token weights:

| Tokens | Weight |
|---|---|
| last assistant turn, from its start up to `<finish>` (the decision to answer and the rationale) | λ (`--stop-weight`) |
| other assistant tokens (tool calls, the answer inside `<finish>`) | 1 |
| system, user, and tool messages | 0 |

The weights of each row are rescaled so that they sum to its number of assistant tokens, and the loss is divided by the
number of assistant tokens in the accumulated batch. With λ = 1 the loss equals the SFT loss. We use λ = 5, chosen on
the validation set. The output layer is computed only at positions with a loss, which needs one row per GPU.

```bash
# Self or Expert: a new adapter on the base model
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m SHFT.train_qwen \
    -r qwen3-self-shft --dataset-dir ../build-dataset/rationale/out/qwen3-8b-nothink-own/datasets/teacher \
    --learning-rate 5e-4 --stop-weight 5 --gradient-accumulation-steps 16
# Ministral
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file deepspeed_zero2.yaml --num_processes 1 -m SHFT.train_mistral \
    -r ministral-self-shft --model mistralai/Ministral-3-8B-Instruct-2512-BF16 --tokenizer out/ministral-8b-tokenizer \
    --chat-template templates/ministral3_training.jinja \
    --dataset-dir ../build-dataset/rationale/out/ministral-8b-own/datasets/teacher \
    --learning-rate 5e-4 --stop-weight 5 --gradient-accumulation-steps 16
```

For Expert → Self, train on the data of `pipeline_two_stage.sh` (`<key>-expert-sis-online`) and continue from the
selected Expert SHFT checkpoint with `--init-adapter <checkpoint>`. Plain and SHFT both take this option.

## Data flow

```
sft/convert_data.py     →  sft/sftdata/train.jsonl                       full Expert trajectories
sft/train_<model>.py    →  sft/out/<run-name>/checkpoint-<step>           SFT baseline
plain/train_<model>.py  →  plain/out/<run-name>/checkpoint-<step>         Plain, from ../build-dataset datasets/no-teacher
SHFT/train_<model>.py   →  SHFT/out/<run-name>/checkpoint-<step>          SHFT, from ../build-dataset datasets/teacher
mistral_tokenizer.py    →  out/ministral-8b-tokenizer                     Ministral tokenizer for training and evaluation
```
