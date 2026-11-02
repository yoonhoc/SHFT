#!/usr/bin/env bash
# 정지 학습 데이터 한 모델분을 한 번에: [RL 문항 롤아웃 →] 모델 서버 → 모든 프리픽스 강제 답변 → 채점(Luna) → 라벨(i*) → teacher 서버 → 정지 서술 → 조립.
# 기본은 자기 롤아웃 판(run-id <키>-own, 조립은 split 의 train 문항만), SFT=1 이면 SFT 궤적(expert) 판(run-id <키>, 롤아웃 없음).
# GPU 1장, 끝난 단계의 파일은 건너뛴다. python 은 저자 코드(sweqapro)가 있는 환경. 로그는 golden_step/out/<run-id>/ 와 rationale/out/<run-id>/ 안.
#
#   cd build-dataset && set -a && source ../.env && set +a
#   CUDA_VISIBLE_DEVICES=3 setsid nohup bash pipeline.sh gemma4-e4b google/gemma-4-E4B-it \
#     --tool-call-parser gemma4 --limit-mm-per-prompt '{"image":0,"audio":0}' > golden_step/out/gemma4-e4b-own.log 2>&1 < /dev/null &
#   SFT=1 TOKENIZER=../train/out/ministral-8b-tokenizer CUDA_VISIBLE_DEVICES=3 setsid nohup bash pipeline.sh ministral-8b mistralai/Ministral-3-8B-Instruct-2512-BF16 \
#     --tool-call-parser mistral --limit-mm-per-prompt '{"image":0}' --tokenizer-mode hf --tokenizer ../train/out/ministral-8b-tokenizer > golden_step/out/ministral-8b.log 2>&1 < /dev/null &
#
# 인자: <models.yaml 키> <vLLM 모델 id> [강제 답변 서버의 vllm serve 추가 인자: tool parser 등. Qwen3 는 --tool-call-parser hermes]
# 환경 변수: SFT (SFT 궤적 판), PORT (기본 8091), TEACHER (기본 Qwen/Qwen3.8-27B), TOKENIZER (조립의 길이 계산용, 기본 모델 id)
set -euo pipefail
KEY=$1
MODEL=$2
shift 2
PORT=${PORT:-8091}
TEACHER=${TEACHER:-Qwen/Qwen3.8-27B}
TOKENIZER=${TOKENIZER:-$MODEL}
export PYTHONPATH=$PWD/../third_party/SWE-QA-Pro/eval
trap 'kill $! 2>/dev/null || true' EXIT

serve() {   # serve <로그> <vllm serve 인자…>: 서버를 띄우고 뜰 때까지 기다린다 (뜨다 죽으면 멈춘다)
    vllm serve "${@:2}" --port $PORT > "$1" 2>&1 &
    until curl -s localhost:$PORT/v1/models | grep -q id; do kill -0 $! || exit 1; sleep 10; done
}

if [ -n "${SFT:-}" ]; then   # SFT 궤적 판: 궤적은 dataset.py 가 HF 에서 읽는다
    RUN=$KEY
    TRAJ=()
else                          # 자기 롤아웃 판: 1) 롤아웃 (run_agent 가 서버를 띄우고 내린다)
    RUN=$KEY-own
    TRAJ=(--trajectories $PWD/../evaluate-agent/out/$RUN/sweqapro-rl/results)
    (cd ../evaluate-agent && python -m run_agent -r $RUN -b sweqapro-rl -m $KEY --max-workers 32)
fi

# 2) 강제 답변 (저자 러너처럼 reasoning parser 없이), 3) 채점 (OPENAI_API_KEY), 4) 라벨
mkdir -p golden_step/out/$RUN
serve golden_step/out/$RUN/serve.log $MODEL --max-model-len 32768 --enable-auto-tool-choice "$@"
# 한 궤적의 프리픽스 요청(inner)을 같이 보내야 vLLM prefix cache 에 맞는다: 64×1 로 돌리면 적중률 1%, 8×8 이면 90% 이상. KV cache 가 32k 요청 16개분(53만 token)이라 궤적 수는 8 까지
python -m golden_step.answer -r $RUN "${TRAJ[@]}" --base-url http://localhost:$PORT/v1 --model $MODEL --max-workers 8 --inner-workers 8
kill $!; sleep 10
python -m golden_step.judge -r $RUN "${TRAJ[@]}"
python -m golden_step.label -r $RUN

# 5) teacher 서술, 6) 조립
mkdir -p rationale/out/$RUN
serve rationale/out/$RUN/serve_teacher.log $TEACHER --max-model-len 40960 --max-num-seqs 256
python -m rationale.generate -r $RUN "${TRAJ[@]}" --base-url http://localhost:$PORT/v1 --model $TEACHER
kill $!; sleep 10
python -m rationale.assemble -r $RUN "${TRAJ[@]}" --model $TOKENIZER
echo "[pipeline] done: $RUN"
