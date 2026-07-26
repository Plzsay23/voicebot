#!/usr/bin/env bash
# setup_train_env_conda.sh — 리눅스 GPU 서버에 SenseVoice 파인튜닝 환경을 conda 로 만든다.
# venv 판(setup_train_env.sh)과 하는 일은 같다. 서버 파이썬이 3.9~3.11 이 아니거나
# conda 를 쓰는 게 편할 때 이쪽을 쓴다.
#
# 사용:
#   bash setup_train_env_conda.sh                  # CUDA 버전 자동 감지
#   CUDA_TAG=cu126 bash setup_train_env_conda.sh   # 직접 지정
#   ENV_NAME=vb-train bash setup_train_env_conda.sh
#
# 만들어지는 것: conda 환경 (기본 이름 voicebot-train, 파이썬 3.10)
#
# 참고: conda 로 cuda-toolkit 을 따로 깔 필요는 없다. pip 의 torch 휠이 CUDA
#       런타임을 통째로 들고 있어서, 서버에는 NVIDIA 드라이버만 있으면 된다.

set -euo pipefail
cd "$(dirname "$0")"

ENV_NAME="${ENV_NAME:-voicebot-train}"
PY_VER="${PY_VER:-3.10}"

info() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. conda
command -v conda >/dev/null 2>&1 || die "conda 가 PATH 에 없다. 'module load anaconda' 같은 걸 먼저 하거나 miniconda 를 설치할 것."
# 스크립트(비대화형 셸)에서는 conda activate 가 그냥 안 먹는다. hook 을 먼저 심는다.
eval "$(conda shell.bash hook)"
info "conda: $(conda --version)  ($(dirname "$(dirname "$(command -v conda)")"))"

# ---------------------------------------------------------------- 2. GPU 확인
command -v nvidia-smi >/dev/null || die "nvidia-smi 가 없다. GPU 노드에서 실행할 것."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'

# 드라이버가 지원하는 CUDA 버전으로 torch 휠을 고른다.
if [[ -z "${CUDA_TAG:-}" ]]; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if   (( DRV >= 570 )); then CUDA_TAG=cu128
  elif (( DRV >= 560 )); then CUDA_TAG=cu126
  elif (( DRV >= 525 )); then CUDA_TAG=cu121
  else                        CUDA_TAG=cu118
  fi
fi
info "torch 빌드: $CUDA_TAG (드라이버 기준 자동 선택, CUDA_TAG= 로 변경 가능)"

# ---------------------------------------------------------------- 3. 환경
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  warn "conda 환경 '$ENV_NAME' 이 이미 있다. 재사용한다. (새로 만들려면 conda env remove -n $ENV_NAME)"
else
  info "conda 환경 '$ENV_NAME' 생성 중 (python $PY_VER)..."
  conda create -y -n "$ENV_NAME" "python=$PY_VER" pip
fi
conda activate "$ENV_NAME"
info "활성화됨: $(python -V)  ($(which python))"

python -m pip install -q --upgrade pip setuptools wheel

# ---------------------------------------------------------------- 4. 패키지
# conda 채널의 pytorch 대신 pip 휠을 쓴다. CUDA 빌드를 태그로 정확히 고를 수 있고
# funasr 쪽 의존성과 섞였을 때 conda/pip 충돌이 덜하다.
info "PyTorch 설치 중 (2~3GB, 몇 분 걸림)..."
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

info "FunASR 및 부속 패키지 설치 중..."
pip install -U funasr modelscope huggingface_hub
pip install -U onnx onnxruntime funasr-onnx
pip install -U soundfile librosa numpy
# fetch_open_data.py 용. datasets 4.x 는 오디오 디코딩을 torchcodec 으로 바꿔서
# item["audio"]["array"] 접근이 깨진다. 3.x 로 묶는다.
pip install -U "datasets>=2.19,<4"

# ---------------------------------------------------------------- 5. 검증
info "설치 확인:"
python - <<'PY'
import torch, sys
print(f"    torch          {torch.__version__}  (CUDA {torch.version.cuda})")
print(f"    GPU 사용 가능    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU            {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"    compute cap.   sm_{cap[0]}{cap[1]}")
    try:                                    # 실제 연산이 되는지까지 확인
        x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")
        torch.cuda.synchronize()
        print("    GPU 연산 테스트  통과")
    except Exception as e:
        print(f"    GPU 연산 테스트  실패: {e}")
        print("    -> torch 빌드와 GPU 아키텍처가 안 맞는다. CUDA_TAG 를 바꿔 재설치할 것.")
        sys.exit(1)
else:
    print("    -> GPU를 못 쓴다. GPU 노드인지, 드라이버/토치 조합이 맞는지 확인할 것.")
    sys.exit(1)
import funasr
print(f"    funasr         {funasr.__version__}")
PY

echo
info "완료. 다음부터는 아래로 활성화:"
echo "    conda activate $ENV_NAME"
