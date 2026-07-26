#!/usr/bin/env bash
# finetune_sensevoice.sh — SenseVoiceSmall 파인튜닝 (리눅스 GPU 서버)
#
# 사전조건: bash setup_train_env.sh 로 venv 구성 + prepare_funasr_data.py 로
#           funasr_data/{train,val}.jsonl 생성이 끝나 있을 것.
#
# 사용:
#   bash finetune_sensevoice.sh                    # 기본값으로 학습
#   EPOCH=30 LR=0.00005 bash finetune_sensevoice.sh
#   DATA=funasr_data OUT=outputs bash finetune_sensevoice.sh
#
# 결과: $OUT/model.pt.ep*  (체크포인트), $OUT/config.yaml

set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-funasr_data}"
OUT="${OUT:-outputs}"
EPOCH="${EPOCH:-20}"
LR="${LR:-0.00005}"
BATCH="${BATCH:-4000}"
GPUS="${GPUS:-0}"

info() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# venv 판이면 여기서 활성화하고, conda 판이면 이미 활성화된 환경을 그대로 쓴다.
if [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
elif python -c 'import funasr' 2>/dev/null; then
  info "현재 파이썬 환경 사용: $(which python)"
else
  die "학습 환경이 없다. 'bash setup_train_env.sh' (venv) 또는
      'bash setup_train_env_conda.sh' 후 'conda activate voicebot-train' 을 먼저 할 것."
fi

[[ -f "$DATA/train.jsonl" ]] || die "$DATA/train.jsonl 이 없다. prepare_funasr_data.py 를 먼저 실행."
N_TRAIN=$(wc -l < "$DATA/train.jsonl")
N_VAL=$(wc -l < "$DATA/val.jsonl")
info "학습 $N_TRAIN줄 / 검증 $N_VAL줄, 최대 $EPOCH epoch, lr=$LR"

# 데이터가 적으면 기본 학습률(0.0002)로는 바로 과적합한다.
if (( N_TRAIN < 200 )); then
  cat <<'EOF'

  [!] 학습 데이터가 200줄 미만이다.
      이 실행은 "학습 -> export -> 파이 배포" 경로가 뚫리는지 확인하는
      용도로만 의미가 있다. 이 결과물의 성능 수치는 믿지 말 것
      (과적합해서 학습 데이터만 잘 맞히고 나머지는 오히려 나빠진다).
      성능을 보려면 300발화 이상 녹음한 뒤 다시 돌릴 것.

EOF
fi

TRAIN_PY=$(python -c "import funasr.bin.train_ds as m; print(m.__file__)")
info "train_ds.py: $TRAIN_PY"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

info "학습 시작. 로그: $OUT/train.log"
torchrun --nnodes 1 --nproc_per_node "$NPROC" "$TRAIN_PY" \
  ++model="iic/SenseVoiceSmall" \
  ++trust_remote_code=true \
  ++train_data_set_list="$DATA/train.jsonl" \
  ++valid_data_set_list="$DATA/val.jsonl" \
  ++dataset_conf.data_split_num=1 \
  ++dataset_conf.batch_sampler="BatchSampler" \
  ++dataset_conf.batch_size="$BATCH" \
  ++dataset_conf.sort_size=1024 \
  ++dataset_conf.batch_type="token" \
  ++dataset_conf.num_workers=4 \
  ++train_conf.max_epoch="$EPOCH" \
  ++train_conf.log_interval=1 \
  ++train_conf.resume=false \
  ++train_conf.validate_interval=200 \
  ++train_conf.save_checkpoint_interval=200 \
  ++train_conf.keep_nbest_models=10 \
  ++train_conf.avg_nbest_model=5 \
  ++optim_conf.lr="$LR" \
  ++output_dir="$OUT" 2>&1 | tee "$OUT/train.log"

echo
info "학습 완료. 산출물:"
ls -lh "$OUT" | grep -E 'model.pt|config' | sed 's/^/    /' || true
echo
info "다음: python export_onnx.py --model-dir $OUT"
