#!/usr/bin/env bash
# compare_deepfilter.sh — 잡음 제거 전/후를 같은 조건에서 녹음해 비교한다.
# 원본 마이크로 5초, DeepFilterMic 으로 5초 녹음한 뒤 차례로 들려준다.
#
# 사용: bash ~/voicebot/scripts/compare_deepfilter.sh [녹음초]

set -euo pipefail
DUR="${1:-5}"
OUT="/tmp/df_compare"
mkdir -p "$OUT"

info() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

pactl list sources short | grep -q DeepFilterMic \
  || { echo "DeepFilterMic 이 없다. 먼저 setup_deepfilter.sh 를 실행할 것."; exit 1; }

ORIG=$(pactl list sources short \
       | awk '$2 !~ /\.monitor$/ && $2 ~ /^alsa_input/ {print $2; exit}')
info "원본 마이크: $ORIG"
echo
echo "  두 번 녹음합니다. 매번 같은 말을 같은 크기로 하세요."
echo "  잡음(선풍기·키보드·에어컨 등)도 같이 내면 차이가 잘 보입니다."
echo

countdown() { for i in 3 2 1; do printf '  %d...\r' "$i"; sleep 1; done; printf '  ● 녹음!  \n'; }

# 소스가 SUSPENDED 상태면 실제로 소리가 흐르기까지 1초 이상 걸린다. 벽시계로
# 재서 끊으면 그만큼 뒷부분이 잘리므로, 파일이 목표 길이만큼 쌓일 때까지 기다린다.
TARGET_BYTES=$(( DUR * 16000 * 2 ))
record() {  # record <device> <파일>
  local dev="$1" out="$2" waited=0
  parecord --device="$dev" --rate=16000 --channels=1 --format=s16le \
           --file-format=wav "$out" &
  local pid=$!
  # 첫 데이터가 들어올 때까지 대기(소스 깨어나는 시간) — 카운트다운은 이미 끝났으므로
  # 여기서 기다린 만큼은 녹음 길이에 포함되지 않는다.
  while [[ ! -s "$out" ]] && (( waited < 100 )); do sleep 0.1; ((waited++)); done
  while (( $(stat -c%s "$out" 2>/dev/null || echo 0) < TARGET_BYTES )) \
        && (( waited < 100 + DUR * 20 )); do sleep 0.1; ((waited++)); done
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
}

info "[1/2] 원본 (잡음 제거 없음) — ${DUR}초"
countdown
record "$ORIG" "$OUT/before.wav"

echo
info "[2/2] DeepFilter 적용 — ${DUR}초"
countdown
record DeepFilterMic "$OUT/after.wav"

for f in "$OUT/before.wav" "$OUT/after.wav"; do
  python3 -c "import wave,sys; w=wave.open(sys.argv[1]); print('    %s  %.2fs' % (sys.argv[1], w.getnframes()/w.getframerate()))" "$f" 2>/dev/null || true
done

echo
info "재생: 원본"; paplay "$OUT/before.wav"
sleep 1
info "재생: DeepFilter 적용"; paplay "$OUT/after.wav"

echo
info "파일: $OUT/before.wav , $OUT/after.wav"
info "PipeWire CPU 점유율 (지금 필터가 도는 상태):"
ps -o pcpu=,comm= -p "$(pgrep -d, -f '[p]ipewire')" 2>/dev/null | sed 's/^/    /' || true
