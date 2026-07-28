#!/usr/bin/env bash
# 볼트 검색 서버 기동/종료. PC 가 켜져 있을 때만 파이가 이걸 쓴다.
#
#   bash scripts/vault_search.sh            서버 확인/기동
#   bash scripts/vault_search.sh --stop     종료
#   bash scripts/vault_search.sh --log      로그 따라가기
#   bash scripts/vault_search.sh --test "질문"   검색만 해보기
#
# 의존성이 없어서 venv 가 필요 없다(시스템 python3 로 돈다).
set -uo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${VAULT_PORT:-8081}"
LOG="$BASE_DIR/.vault_server.log"
PIDFILE="$BASE_DIR/.vault_server.pid"
export VAULT_DIR="${VAULT_DIR:-/mnt/c/ysj/ResearchWiki}"

PY="$(command -v python3 || command -v python)"

say()  { printf '\033[1;36m%s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m%s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m%s\033[0m\n' "$*" >&2; exit 1; }

alive() { curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

case "${1:-}" in
  --stop)
    if [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null; then
      rm -f "$PIDFILE"; say "볼트 서버 종료했다."
    else
      pkill -f "[v]ault_server.py" && say "볼트 서버 종료했다." || warn "돌고 있는 볼트 서버가 없다."
    fi
    exit 0 ;;
  --log)
    tail -f "$LOG"; exit 0 ;;
  --test)
    shift
    Q="${1:-제리}"
    alive || die "서버가 안 떠 있다. 먼저:  bash $BASE_DIR/scripts/vault_search.sh"
    curl -fsS --get "http://127.0.0.1:$PORT/search" \
         --data-urlencode "q=$Q" --data-urlencode "k=3" --data-urlencode "maxchars=600" \
      | "$PY" -c '
import json, sys
d = json.load(sys.stdin)
print("질의: " + d["query"])
rs = d["results"]
if not rs:
    print("  (검색 결과 없음)")
for r in rs:
    print("  [%s] %s  (%s)" % (r["score"], r["title"], r["path"]))
    print("        " + r["text"][:200])
'
    exit 0 ;;
esac

[ -n "$PY" ] || die "python3 가 없다."
[ -d "$VAULT_DIR" ] || die "볼트가 없다: $VAULT_DIR  (VAULT_DIR 로 지정하라)"

if alive; then
  say "볼트 서버 이미 떠 있다 (:$PORT)"
else
  say "볼트 서버 기동 중... (첫 인덱싱은 수십 초 걸릴 수 있다: --log)"
  nohup "$PY" "$BASE_DIR/scripts/vault_server.py" >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 180); do
    alive && break
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { tail -20 "$LOG"; die "서버가 죽었다. 위 로그를 보라."; }
    sleep 1
  done
  alive || { tail -20 "$LOG"; die "서버가 180초 안에 안 떴다."; }
fi

curl -fsS -m 2 "http://127.0.0.1:$PORT/health" | sed 's/^/  /'
echo

IP="$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K\S+')"
say "파이의 ~/voicebot/.env 에 넣을 값:"
echo "  VAULT_SEARCH_URL=http://$IP:$PORT"
case "$IP" in
  172.1[6-9].*|172.2[0-9].*|172.3[01].*)
    warn "  ↑ WSL 내부 NAT 주소라 파이에서 닿지 않는다. docs/VAULT_RAG.md 를 보라." ;;
esac
