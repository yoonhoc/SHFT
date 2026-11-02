#!/usr/bin/env bash
# SFT 기준선을 학습한다: TRL SFTTrainer 기본 loss (chunked_nll, assistant token 만), Expert 궤적 1,000개 (sft/sftdata).
# batch 는 원래 학습(2026-10-14, H100 8장 × gradient accumulation 2)과 같은 16 이 되게 gradient accumulation 을 16 / GPU 수로 준다
# (epoch 당 63 step, checkpoint-63·126·189·252). 나머지는 utils.py 기본값 (lr 5e-4, epoch 4, LoRA r32·α64·dropout 0.1).
#
#   cd code/train && source ~/.secrets
#   CUDA_VISIBLE_DEVICES=<g1>,<g2> PY=/usr/bin/python3 setsid nohup bash sft/run.sh qwen3 ministral > sft/out/run.log 2>&1 < /dev/null &
#
# 인자: 학습할 모델 (qwen3, ministral), 여럿이면 차례로. run name 은 <모델>-sft-$SUFFIX (기본 rerun).
# GPU 는 CUDA_VISIBLE_DEVICES 의 장 수만큼 쓴다 (1, 2, 4, 8, 16 중 하나).
# 결과: sft/out/<run name>/ (checkpoint-<step>, train_args.json, train_stdout.log, wandb)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
SUFFIX=${SUFFIX:-rerun}
GPUS=$(echo "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
if [ $((16 % GPUS)) -ne 0 ]; then
  echo "GPU count $GPUS does not divide batch 16" >&2
  exit 1
fi
ACCUM=$((16 / GPUS))

for model in "$@"; do
  case $model in
    qwen3)
      args=(-m sft.train_qwen -r qwen3-8b-nothink-sft-$SUFFIX --model Qwen/Qwen3-8B)
      ;;
    ministral)
      args=(-m sft.train_mistral -r ministral-8b-sft-$SUFFIX --model mistralai/Ministral-3-8B-Instruct-2512-BF16
            --tokenizer out/ministral-8b-tokenizer --chat-template templates/ministral3_training.jinja)
      ;;
    *)
      echo "unknown model: $model (qwen3 | ministral)" >&2
      exit 1
      ;;
  esac
  echo "== $model on $GPUS GPU(s), gradient accumulation $ACCUM: ${args[*]}"
  HF_HUB_OFFLINE=1 "$PY" -m accelerate.commands.launch --config_file deepspeed_zero2.yaml --num_processes "$GPUS" "${args[@]}" \
    --dataset-dir sft/sftdata --learning-rate 5e-4 --gradient-accumulation-steps "$ACCUM"
done
