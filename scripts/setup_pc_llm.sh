#!/usr/bin/env bash
# PC(WSL) 에서 파이와 똑같은 모델을 GPU 로 돌리는 환경을 만든다.
#
# 왜 ollama 가 아니고 llama-cpp-python 인가:
#   파이가 쓰는 엔진(llama.cpp) · 모델 파일(gguf) · 샘플러 설정을 그대로 쓰면
#   PC 로 붙었을 때와 로컬로 떨어졌을 때 답변 성격이 달라지지 않는다.
#   게다가 llama-cpp-python[server] 가 OpenAI 호환 /v1 을 그대로 제공하므로
#   파이의 REMOTE_LLM_URL 이 손댈 것 없이 붙는다.
#
# CUDA 툴킷(nvcc)이 WSL 에 이미 깔려 있어야 한다. sudo 는 쓰지 않는다.
#
# 한 번만 돌리면 된다:  bash scripts/setup_pc_llm.sh
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$BASE_DIR/.venv-llm"
MODEL_DIR="$BASE_DIR/models"
MODEL_FILE="EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"   # 파이와 같은 파일
HF_REPO="LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF"
# RTX 5060(Blackwell) = sm_120. 다른 GPU 면 여기만 바꾼다(3060=86, 4060=89).
CUDA_ARCH="${CUDA_ARCH:-120}"
LOG="$BASE_DIR/.venv-llm-build.log"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31m!! %s\033[0m\n' "$*" >&2; exit 1; }

# ---------- 0. 전제조건 ----------
say "전제조건 확인"
command -v nvcc >/dev/null || die "nvcc 가 없다. CUDA 툴킷을 깔거나 PATH 에 /usr/local/cuda/bin 을 넣어라."
command -v g++  >/dev/null || die "g++ 가 없다: sudo apt install build-essential"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "nvidia-smi 실패. WSL GPU 패스스루를 확인하라."
echo "nvcc: $(nvcc -V | tail -1)"
python3 -c 'import sysconfig,os,sys; p=sysconfig.get_paths()["include"]+"/Python.h"; sys.exit(0 if os.path.exists(p) else 1)' \
  || die "Python.h 가 없다: sudo apt install python3-dev"

# ---------- 1. venv ----------
if [ -d "$VENV" ]; then
  say "venv 이미 있음: $VENV (건너뜀)"
else
  say "venv 생성: $VENV"
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -qU pip wheel

# ---------- 2. llama-cpp-python (CUDA) ----------
if python -c 'import llama_cpp' 2>/dev/null; then
  say "llama-cpp-python 이미 설치됨 (다시 빌드하려면 .venv-llm 을 지워라)"
else
  say "llama-cpp-python CUDA 빌드 (sm_$CUDA_ARCH). 5~15분 걸린다. 로그: $LOG"
  # -DGGML_CUDA=on 이 GPU 오프로딩을 켠다. 아키텍처를 명시하지 않으면
  # CUDA 13 이 Blackwell 을 못 잡고 PTX JIT 로 새거나 빌드가 터진다.
  CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
  FORCE_CMAKE=1 \
    python -m pip install --no-cache-dir --verbose 'llama-cpp-python[server]' >"$LOG" 2>&1 \
    || die "빌드 실패. 로그 마지막을 보라: tail -40 $LOG"
  python -c 'import llama_cpp' || die "빌드는 됐는데 import 가 안 된다: $LOG"
fi
# 파이의 voice_common 을 그대로 import 해서 쓰기 위한 것(같은 시스템 프롬프트/분기 로직).
python -m pip install -q python-dotenv

# ---------- 3. 모델 ----------
mkdir -p "$MODEL_DIR"
if [ -s "$MODEL_DIR/$MODEL_FILE" ]; then
  say "모델 이미 있음: $MODEL_DIR/$MODEL_FILE ($(du -h "$MODEL_DIR/$MODEL_FILE" | cut -f1))"
else
  say "모델 다운로드: $HF_REPO / $MODEL_FILE (약 1.6GB)"
  # -C - 로 중단된 다운로드를 이어받는다.
  curl -fL -C - --retry 3 --progress-bar \
    -o "$MODEL_DIR/$MODEL_FILE" \
    "https://huggingface.co/$HF_REPO/resolve/main/$MODEL_FILE?download=true" \
    || die "다운로드 실패"
fi

# ---------- 4. bashrc 에 chat 등록 ----------
BASHRC="$HOME/.bashrc"
MARK_BEGIN="# >>> voicebot chat (자동 생성, 지우려면 이 블록째로) >>>"
MARK_END="# <<< voicebot chat <<<"
if grep -qF "$MARK_BEGIN" "$BASHRC" 2>/dev/null; then
  say "~/.bashrc 에 chat 이 이미 있음 (건너뜀)"
else
  say "~/.bashrc 에 chat 등록"
  {
    echo ""
    echo "$MARK_BEGIN"
    echo "chat() { bash \"$BASE_DIR/scripts/pc_llm_chat.sh\" \"\$@\"; }"
    echo "$MARK_END"
  } >> "$BASHRC"
fi

say "끝났다"
cat <<EOF

  새 터미널을 열거나  source ~/.bashrc  하고

      chat

  만 치면 GPU 서버가 뜨고 바로 대화창이 열린다.
  서버는 백그라운드에 남아 파이가 쓸 수 있게 0.0.0.0 에 붙는다.
  서버만 띄우려면:  chat --serve-only

EOF
