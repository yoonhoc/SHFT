#!/usr/bin/env bash
# SIS(expert) + SIS-online: expert 궤적으로 SIS 학습한 모델(표 1 의 <모델>-SIS)이 학습 문항을 직접 푼 궤적으로 SIS 를 한 번 더 한다.
# 순서는 pipeline_sft_self.sh 와 같다: 롤아웃 → 강제 답변 → 채점 → 라벨 → teacher 서술 → 조립 (학습은 따로, 시작 어댑터에서 이어서).
# 강제 답변도 시작 어댑터를 얹은 모델이 낸다 (golden step = 이 모델에게 충분해지는 지점).
#
#   cd build-dataset && set -a && source ../.env && set +a
#   CUDA_VISIBLE_DEVICES=<g> setsid nohup bash pipeline_two_stage.sh qwen3     > golden_step/out/qwen3-8b-nothink-expert-sis-online.log 2>&1 < /dev/null &
#   CUDA_VISIBLE_DEVICES=<g> setsid nohup bash pipeline_two_stage.sh ministral > golden_step/out/ministral-8b-expert-sis-online.log 2>&1 < /dev/null &
#
# GPU 한 장에 두 모델을 같이 올려 라벨까지만 먼저 하려면 (vLLM 메모리를 반씩, teacher 는 나중에 한 모델씩):
#   GPU_MEM=0.45 UNTIL=label CUDA_VISIBLE_DEVICES=<g> setsid nohup bash pipeline_two_stage.sh qwen3     > … &
#   GPU_MEM=0.45 UNTIL=label CUDA_VISIBLE_DEVICES=<g> setsid nohup bash pipeline_two_stage.sh ministral > … &
#
# 인자: ministral 또는 qwen3. 환경 변수: GPU_MEM (vLLM 이 쓸 GPU 메모리 비율, 기본 0.90), UNTIL=label (라벨까지 하고 멈춤).
# GPU 1장 (모델마다 따로). 단계마다 결과가 있으면 건너뛰므로, 끊기면 같은 명령으로 이어 간다.
# 결과: evaluate-agent/out/<RUN>/sweqapro-rl/ (롤아웃), golden_step/out/<RUN>/ (강제 답변·점수·라벨),
#       rationale/out/<RUN>/datasets/teacher/train.jsonl (학습 데이터)
set -euo pipefail

WHICH=$1                                            # ministral 또는 qwen3
HERE=$(cd "$(dirname "$0")" && pwd)                 # code/build-dataset
PY=${PY:-python}
export PYTHONPATH=$HERE/../third_party/SWE-QA-Pro/eval
TEACHER=Qwen/Qwen3.8-27B

# ---------- 모델 설정 (pipeline_sft_self.sh 와 같다) ----------
# 시작 어댑터 = 표 1 의 <모델>-SIS (validation 으로 고른 checkpoint)
if [ "$WHICH" = ministral ]; then
    KEY=ministral-8b                                # evaluate-agent/models.yaml 의 키
    MODEL=mistralai/Ministral-3-8B-Instruct-2512-BF16
    ADAPTER=$HERE/../train/SHFT/out/$KEY-expert-sis-w5/checkpoint-189     # e3
    TOKENIZER=$HERE/../train/out/ministral-8b-tokenizer
    MODEL_ARGS=(--tool-call-parser mistral --limit-mm-per-prompt '{"image":0}' --tokenizer-mode hf --tokenizer "$TOKENIZER")
    PORT=8091
elif [ "$WHICH" = qwen3 ]; then
    KEY=qwen3-8b-nothink
    MODEL=Qwen/Qwen3-8B
    ADAPTER=$HERE/../train/SHFT/out/$KEY-expert-sis-w5/checkpoint-252     # e4
    TOKENIZER=$MODEL
    MODEL_ARGS=(--tool-call-parser hermes --default-chat-template-kwargs '{"enable_thinking": false}')   # 생각 끔
    PORT=8092
else
    echo "usage: bash pipeline_two_stage.sh ministral|qwen3" >&2
    exit 1
fi
RUN=$KEY-expert-sis-online
GPU_MEM=${GPU_MEM:-0.90}
TRAJECTORIES=$HERE/../evaluate-agent/out/$RUN/sweqapro-rl/results
LORA_RANK=$(python3 -c "import json; print(json.load(open('$ADAPTER/adapter_config.json'))['r'])")
mkdir -p "$HERE/golden_step/out/$RUN" "$HERE/rationale/out/$RUN"
# 롤아웃 서버(run_agent)는 models.yaml 의 gpu_memory_utilization 을 쓰므로, 비율만 바꾼 사본을 만들어 준다
CONFIG=$HERE/golden_step/out/$RUN/models.yaml
sed "s/gpu_memory_utilization: 0.90/gpu_memory_utilization: $GPU_MEM/" "$HERE/../evaluate-agent/models.yaml" > "$CONFIG"
SERVER=""
trap '[ -n "$SERVER" ] && kill $SERVER 2>/dev/null || true' EXIT    # 중간에 실패해도 띄운 vLLM 서버는 내린다 (GPU 반납)
echo "$(date '+%F %T') start $RUN (adapter $ADAPTER)"

serve() {   # serve <로그> <vllm serve 인자…>: 서버를 띄우고 뜰 때까지 기다린다 (뜨다 죽으면 멈춘다)
    vllm serve "${@:2}" --port $PORT --gpu-memory-utilization $GPU_MEM > "$1" 2>&1 &
    SERVER=$!
    until curl -s localhost:$PORT/v1/models | grep -q id; do
        kill -0 $SERVER
        sleep 10
    done
}

# ---------- 1) 롤아웃: RL 문항(train + validation)을 시작 모델로 (run_agent 가 서버를 띄우고 내린다) ----------
cd "$HERE/../evaluate-agent"
$PY -m run_agent -r $RUN -b sweqapro-rl -m $KEY --adapter "$ADAPTER" --max-workers 32 --config "$CONFIG"

# ---------- 2) 강제 답변: 모든 prefix 에서 답 3개 (어댑터를 얹은 서버, 요청 이름은 adapter) ----------
cd "$HERE"
serve golden_step/out/$RUN/serve.log $MODEL --max-model-len 32768 --enable-auto-tool-choice \
    --enable-lora --max-lora-rank $LORA_RANK --lora-modules adapter="$ADAPTER" "${MODEL_ARGS[@]}"
$PY -m golden_step.answer -r $RUN --trajectories "$TRAJECTORIES" --base-url http://localhost:$PORT/v1 --model adapter \
    --max-workers 8 --inner-workers 8
kill $SERVER
SERVER=""
sleep 10

# ---------- 3) 채점 (Luna, OPENAI_API_KEY) → 4) 라벨 (golden step) ----------
$PY -m golden_step.judge -r $RUN --trajectories "$TRAJECTORIES"
$PY -m golden_step.label -r $RUN
if [ "${UNTIL:-}" = label ]; then
    echo "$(date '+%F %T') stopped after labels: $RUN"
    exit 0
fi

# ---------- 5) teacher 서술 → 6) 학습 데이터 조립 (split 의 train 문항만) ----------
if [ ! -f rationale/out/$RUN/datasets/teacher/train.jsonl ]; then
    serve rationale/out/$RUN/serve_teacher.log $TEACHER --max-model-len 40960 --max-num-seqs 256
    $PY -m rationale.generate -r $RUN --trajectories "$TRAJECTORIES" --base-url http://localhost:$PORT/v1 --model $TEACHER
    kill $SERVER
    SERVER=""
    sleep 10
    $PY -m rationale.assemble -r $RUN --trajectories "$TRAJECTORIES" --model "$TOKENIZER"
fi

echo "$(date '+%F %T') done $RUN -> rationale/out/$RUN/datasets/teacher/train.jsonl"
