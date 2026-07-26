#!/usr/bin/env bash
# setup_train_env.sh — 리눅스 GPU 서버에 SenseVoice 파인튜닝 환경을 만든다.
# sudo 없이 사용자 홈에만 설치한다(공용 서버 가정).
#
# 사용:
#   bash setup_train_env.sh            # CUDA 버전 자동 감지
#   CUDA_TAG=cu121 bash setup_train_env.sh   # 직접 지정
#
# 만들어지는 것: ./venv  (이 스크립트가 있는 폴더 기준)

set -euo pipefail
cd "$(dirname "$0")"

info() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- 1. 파이썬
PY=""
for c in python3.11 python3.10 python3.9 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print("%d%02d" % sys.version_info[:2])')
    if (( v >= 309 && v <= 311 )); then PY="$c"; break; fi
  fi
done
[[ -n "$PY" ]] || die "python 3.9~3.11 이 필요하다. 'module avail python' 으로 찾아보거나 conda 를 쓸 것."
info "파이썬: $PY ($($PY -V))"

# ---------------------------------------------------------------- 2. GPU 확인
command -v nvidia-smi >/dev/null || die "nvidia-smi 가 없다. GPU 노드에서 실행할 것."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/    /'

# 드라이버가 지원하는 CUDA 버전으로 torch 휠을 고른다.
if [[ -z "${CUDA_TAG:-}" ]]; then
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if   (( DRV >= 570 )); then CUDA_TAG=cu128
  elif (( DRV >= 525 )); then CUDA_TAG=cu121
  else                        CUDA_TAG=cu118
  fi
fi
info "torch 빌드: $CUDA_TAG (드라이버 기준 자동 선택, CUDA_TAG= 로 변경 가능)"

# ---------------------------------------------------------------- 3. venv
if [[ -d venv ]]; then
  warn "venv 가 이미 있다. 재사용한다. (새로 만들려면 rm -rf venv)"
else
  info "venv 생성 중..."
  "$PY" -m venv venv
fi
source venv/bin/activate
python -m pip install -q --upgrade pip setuptools wheel

# ---------------------------------------------------------------- 4. 패키지
info "PyTorch 설치 중 (2~3GB, 몇 분 걸림)..."
pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

info "FunASR 및 부속 패키지 설치 중..."
pip install -U funasr modelscope huggingface_hub
# onnxscript 는 torch 2.6+ 의 torch.onnx.export 가 내부에서 import 한다.
# 없으면 export_onnx.py 가 ModuleNotFoundError 로 죽는다.
pip install -U onnx onnxruntime onnxscript funasr-onnx
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
echo "    source $(pwd)/venv/bin/activate"
