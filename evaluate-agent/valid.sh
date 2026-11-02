#!/usr/bin/env bash
# 학습 한 판의 checkpoint 들을 validation 60문항으로 차례로 평가한다 (실주행 + 채점). GPU 1장.
# 실행분 id 는 <학습 run-id>-e<epoch> (epoch = checkpoint 번호 / 한 epoch 의 step 수, 4 epoch 학습 전제).
#
#   cd evaluate-agent && set -a && source ../.env && set +a
#   CUDA_VISIBLE_DEVICES=1 bash valid.sh ministral-8b-own-sis-w5 ministral-8b 25 50 75 100
#
# 인자: <train/SHFT/out 의 run-id> <models.yaml 키> <checkpoint 번호…>. 이미 있는 결과는 run_agent·judge 가 건너뛴다.
set -euo pipefail
RUN=$1
KEY=$2
HERE=$(cd "$(dirname "$0")" && pwd)                          # evaluate-agent/
PY=${PY:-python}
export PYTHONPATH=$HERE/../third_party/SWE-QA-Pro/eval

cd "$HERE"
DIR=${TRAIN_DIR:-../train/SHFT/out}/$RUN       # Plain 판은 TRAIN_DIR=../train/plain/out
if [ -d "$DIR/train" ]; then DIR=$DIR/train; fi    # 베슬 판은 train/ 아래에 checkpoint 가 있다
LAST=$(ls -d "$DIR"/checkpoint-* | sed 's/.*-//' | sort -n | tail -1)
for ckpt in "${@:3}"; do
    epoch=$((ckpt * 4 / LAST))
    $PY -m run_agent -r "$RUN-e$epoch" -b valid -m "$KEY" --adapter "$DIR/checkpoint-$ckpt" ${RUN_AGENT_ARGS:-}
    $PY -m score.judge -r "$RUN-e$epoch" -b valid
done
