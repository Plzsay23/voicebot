#!/usr/bin/env bash
# `chat` 이 부르는 스크립트. GPU 서버를 (없으면) 띄우고 대화창을 연다.
#
#   chat                 서버 확인/기동 후 터미널 대화
#   chat --serve-only    서버만 띄우고 끝
#   chat --stop          서버 종료
#   chat --log           서버 로그 따라가기
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$BASE_DIR/.venv-llm"
MODEL="$BASE_DIR/models/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"
PORT="${PC_LLM_PORT:-8080}"
# -1 = 전 레이어 GPU. GPU 메모리가 부족해 죽으면 20 같은 숫자로 낮춘다.
NGL="${PC_LLM_GPU_LAYERS:--1}"
NCTX="${PC_LLM_CTX:-4096}"
ALIAS="exaone-3.5-2.4b"
LOG="$BASE_DIR/.pc_llm_server.log"
PIDFILE="$BASE_DIR/.pc_llm_server.pid"

say()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

alive() { curl -fsS -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

case "${1:-}" in
  --stop)
    if [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null; then
      rm -f "$PIDFILE"; say "서버 종료했다."
    else
      pkill -f "llama_cpp.server.*--port $PORT" && say "서버 종료했다." || warn "돌고 있는 서버가 없다."
    fi
    bash "$BASE_DIR/scripts/vault_search.sh" --stop
    exit 0 ;;
  --log)
    tail -f "$LOG"; exit 0 ;;
esac

[ -d "$VENV" ] || die "환경이 없다. 먼저:  bash $BASE_DIR/scripts/setup_pc_llm.sh"
[ -s "$MODEL" ] || die "모델이 없다. 먼저:  bash $BASE_DIR/scripts/setup_pc_llm.sh"

# ---------- 서버 ----------
if alive; then
  say "서버 이미 떠 있다 (:$PORT)"
else
  say "GPU 서버 기동 중... (모델 로딩 로그: chat --log)"
  nohup "$VENV/bin/python" -m llama_cpp.server \
    --model "$MODEL" \
    --model_alias "$ALIAS" \
    --n_gpu_layers "$NGL" \
    --n_ctx "$NCTX" \
    --host 0.0.0.0 --port "$PORT" \
    >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"

  for _ in $(seq 1 120); do
    alive && break
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$LOG"; die "서버가 죽었다. 위 로그를 보라 (GPU 메모리 부족이면 PC_LLM_GPU_LAYERS=20 chat)"; }
    sleep 1
  done
  alive || { tail -20 "$LOG"; die "서버가 120초 안에 안 떴다."; }
  say "준비 완료 (:$PORT)"
fi

# GPU 를 실제로 쓰는지 한 줄로 확인시켜 준다(offloaded 0 이면 CPU 로 돌고 있는 것).
grep -m1 -o "offloaded [0-9]*/[0-9]* layers to GPU" "$LOG" 2>/dev/null | sed 's/^/  GPU: /'

# ---------- 볼트 검색 서버 ----------
# LLM 과 생명주기를 맞춰 둔다("PC 켜져 있을 때만" 이 두 기능의 공통 전제다).
# 볼트가 없거나 기동에 실패해도 LLM 은 그대로 쓸 수 있어야 하므로 실패는 경고만.
if [ "${PC_LLM_NO_VAULT:-}" != "1" ]; then
  bash "$BASE_DIR/scripts/vault_search.sh" >/dev/null 2>&1 \
    && say "볼트 검색 서버도 떠 있다 (:${VAULT_PORT:-8081})" \
    || warn "볼트 검색 서버는 못 띄웠다(LLM 은 정상). 이유: bash scripts/vault_search.sh"
fi

# ---------- 파이가 붙을 주소 ----------
IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+')"
echo
say "파이의 ~/voicebot/.env 에 넣을 값:"
echo "  REMOTE_LLM_URL=http://$IP:$PORT/v1"
case "$IP" in
  172.1[6-9].*|172.2[0-9].*|172.3[01].*)
    warn "  ↑ 이건 WSL 내부 NAT 주소라 파이에서 닿지 않는다."
    warn "    docs/PC_LLM.md 의 '파이에서 PC 로 닿게 하기' 를 먼저 처리하라." ;;
esac
echo

[ "${1:-}" = "--serve-only" ] && exit 0

# ---------- 대화 ----------
exec "$VENV/bin/python" "$BASE_DIR/scripts/pc_chat.py" --port "$PORT"
