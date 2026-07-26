#!/usr/bin/env bash
# record_noise.sh — 잡음 증강 학습에 쓸 배경 소음을 녹음한다. 파이에서 실행.
#
# DeepFilterNet 이 파이4에서 실시간 처리가 불가능하다고 판명됐으므로
# (RTF 1.0~1.7, 플러그인이 패닉을 내며 마이크가 죽음), 잡음 대응은
# 학습 단계에서 한다. 여기서 녹음한 소음을 발화에 섞어 데이터를 불린다.
#
# 사용:
#   bash ~/voicebot/scripts/record_noise.sh 선풍기 30
#   bash ~/voicebot/scripts/record_noise.sh 에어컨 30
#   bash ~/voicebot/scripts/record_noise.sh 키보드 30
#
# 저장: ~/voicebot/training/data/noise/<이름>.wav  (16kHz mono)

set -euo pipefail

NAME="${1:-}"
DUR="${2:-30}"
OUT_DIR="$HOME/voicebot/training/data/noise"

if [[ -z "$NAME" ]]; then
  cat <<'EOF'
사용법: bash record_noise.sh <이름> [초]

권장 목록 (각 30초씩):
  선풍기   에어컨   키보드   TV   사람말소리   창밖   냉장고   무음

'무음'도 꼭 받을 것 — 아무 소리 안 나는 상태의 마이크 자체 노이즈다.
실제로 조용할 때 인식이 나빠지는 걸 막아준다.

주의: 잡음만 녹음해야 한다. 녹음 중에 말하지 말 것.
EOF
  exit 1
fi

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$NAME.wav"

echo "'$NAME' 잡음을 ${DUR}초 녹음합니다."
echo "녹음 중에는 말하지 마세요. 해당 소음만 나게 하세요."
for i in 3 2 1; do printf '  %d...\r' "$i"; sleep 1; done
printf '  ● 녹음 중 (%s초)\n' "$DUR"

TARGET=$(( DUR * 16000 * 2 ))
parecord --rate=16000 --channels=1 --format=s16le --file-format=wav "$OUT" &
PID=$!
waited=0
while [[ ! -s "$OUT" ]] && (( waited < 100 )); do sleep 0.1; ((waited++)); done
while (( $(stat -c%s "$OUT" 2>/dev/null || echo 0) < TARGET )) \
      && (( waited < 100 + DUR * 20 )); do sleep 0.1; ((waited++)); done
kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

python3 - "$OUT" <<'PY'
import sys, wave
import numpy as np
p = sys.argv[1]
with wave.open(p) as w:
    sr, n = w.getframerate(), w.getnframes()
    a = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
rms = float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0
db = 20 * np.log10(rms) if rms > 0 else -99
print(f"  저장: {p}  {n / sr:.1f}s  RMS {db:.1f} dBFS")
if db < -60:
    print("  ⚠ 거의 무음이다. '무음' 샘플이면 정상, 아니면 소음원을 확인할 것.")
PY

echo
echo "현재 모아둔 잡음:"
ls -1 "$OUT_DIR" 2>/dev/null | sed 's/^/    /' || echo "    (없음)"
