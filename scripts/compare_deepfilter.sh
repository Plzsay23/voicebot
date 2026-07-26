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

info "[1/2] 원본 (잡음 제거 없음) — ${DUR}초"
countdown
parecord --device="$ORIG" --rate=16000 --channels=1 --format=s16le \
         --file-format=wav "$OUT/before.wav" &
sleep "$DUR"; kill %1 2>/dev/null || true; wait 2>/dev/null || true

echo
info "[2/2] DeepFilter 적용 — ${DUR}초"
countdown
parecord --device=DeepFilterMic --rate=16000 --channels=1 --format=s16le \
         --file-format=wav "$OUT/after.wav" &
sleep "$DUR"; kill %1 2>/dev/null || true; wait 2>/dev/null || true

echo
info "재생: 원본"; paplay "$OUT/before.wav"
sleep 1
info "재생: DeepFilter 적용"; paplay "$OUT/after.wav"

echo
info "파일: $OUT/before.wav , $OUT/after.wav"
info "PipeWire CPU 점유율 (지금 필터가 도는 상태):"
ps -o pcpu=,comm= -p "$(pgrep -d, -f '[p]ipewire')" 2>/dev/null | sed 's/^/    /' || true
