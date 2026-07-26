#!/usr/bin/env bash
# Piper vs edge-tts 를 같은 문장으로 번갈아 들려주고 합성 시간을 잰다.
# 파이에서 실행:  bash ~/voicebot/scripts/compare_tts.sh
# 문장을 직접 주려면: bash ~/voicebot/scripts/compare_tts.sh "들려줄 문장"
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$BASE_DIR/.venv/bin/python"

TEXTS=("${@}")
if [ ${#TEXTS[@]} -eq 0 ]; then
  TEXTS=(
    "네, 알겠습니다."
    "오늘 서울 날씨는 흐리고 최고 기온은 이십팔 도입니다."
    "죄송해요, 무슨 말인지 잘 못 알아들었어요. 다시 한번 말씀해 주시겠어요?"
  )
fi

synth() {  # $1=backend $2=text $3=out
  TTS_BACKEND="$1" "$PY" -c '
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path("'"$BASE_DIR"'") / "ros_nodes"))
import voice_common as vc
text, out = sys.argv[1], Path(sys.argv[2])
t0 = time.time()
vc.synthesize(text, out)
print(f"  합성 {time.time()-t0:.2f}초  ->  {out}")
' "$2" "$3"
}

for text in "${TEXTS[@]}"; do
  echo
  echo "=================================================="
  echo "문장: $text"
  for be in piper edge; do
    echo "-- $be"
    synth "$be" "$text" "/tmp/tts_$be.wav"
    paplay "/tmp/tts_$be.wav" || true
    sleep 0.4
  done
done

echo
echo "마음에 드는 쪽을 .env 에 박아라:  echo 'TTS_BACKEND=piper' >> $BASE_DIR/.env"
