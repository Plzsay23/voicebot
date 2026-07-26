#!/usr/bin/env bash
# Piper TTS 설치 — 라즈베리파이에서 실행한다.
#
#   bash ~/voicebot/scripts/setup_piper.sh
#
# 하는 일:
#   1. voicebot venv 에 piper-tts 설치 (aarch64 휠 있음, 1.6.0+)
#   2. 한국어 음성 ko_KR-kss-medium 을 models/piper 로 내려받음 (약 63MB)
#   3. 합성 스모크 테스트 + 재생
#
# 되돌리기: .env 에 TTS_BACKEND=edge 를 넣으면 코드 수정 없이 edge-tts 로 돌아간다.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$BASE_DIR/.venv/bin/python"
DATA_DIR="$BASE_DIR/models/piper"
VOICE="${PIPER_VOICE:-ko_KR-kss-medium}"

if [ ! -x "$PY" ]; then
  echo "venv 파이썬이 없다: $PY" >&2
  exit 1
fi

# onnxruntime 은 SenseVoice(funasr_onnx)와 silero VAD 가 같이 쓴다.
# piper-tts 가 이걸 끌어올려 버리면 STT 가 조용히 깨질 수 있으므로 전후를 비교한다.
ORT_BEFORE="$("$PY" -c 'import onnxruntime;print(onnxruntime.__version__)' 2>/dev/null || echo none)"

echo "== piper-tts 설치 =="
"$PY" -m pip install --upgrade piper-tts

ORT_AFTER="$("$PY" -c 'import onnxruntime;print(onnxruntime.__version__)' 2>/dev/null || echo none)"
if [ "$ORT_BEFORE" != "$ORT_AFTER" ]; then
  echo
  echo "!! onnxruntime 이 $ORT_BEFORE -> $ORT_AFTER 로 바뀌었다."
  echo "   STT(SenseVoice)/VAD 가 이걸 공유한다. 스택 띄워서 전사가 되는지 꼭 확인할 것."
  echo "   깨졌으면:  $PY -m pip install 'onnxruntime==$ORT_BEFORE'"
  echo
fi

echo "== 음성 내려받기: $VOICE =="
mkdir -p "$DATA_DIR"
"$PY" -m piper.download_voices "$VOICE" --data-dir "$DATA_DIR"

echo "== 스모크 테스트 =="
OUT=/tmp/piper_test.wav
time "$PY" -m piper -m "$VOICE" --data-dir "$DATA_DIR" -f "$OUT" \
  -- "안녕하세요, 저는 제리입니다. 무엇을 도와드릴까요?"

ls -lh "$OUT"
paplay "$OUT" || echo "재생 실패 — 파일은 만들어졌으니 스피커 설정을 확인해라."

cat <<EOF

끝났다. voice_common.py 의 기본값이 이미 TTS_BACKEND=piper 라 .env 수정은 필요 없다.
edge-tts 로 되돌리려면:  echo 'TTS_BACKEND=edge' >> $BASE_DIR/.env
EOF
