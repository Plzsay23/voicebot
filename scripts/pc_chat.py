#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PC(WSL) 터미널 대화창. `chat` 이 마지막에 실행하는 것.

일부러 ros_nodes/voice_common.py 를 그대로 import 해서 쓴다. 시스템 프롬프트,
문장 분할, 원격 호출 코드가 파이와 한 벌이므로 여기서 잘 나오면 파이에서도
같은 답이 나오고, 여기서 깨지면 파이에서도 깨진다(= 이 창이 곧 회귀 테스트다).
"""

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="8080")
args = ap.parse_args()

# voice_common 이 import 시점에 읽는 값들. load_dotenv 는 이미 있는 환경변수를
# 덮지 않으므로 여기서 넣은 것이 이긴다.
os.environ["REMOTE_LLM_URL"] = f"http://127.0.0.1:{args.port}/v1"
os.environ.setdefault("REMOTE_LLM_PROBE_TTL", "60")
# ddgs 를 이 venv 에 넣지 않았다. 검색은 파이에서만 쓴다.
os.environ["WEB_SEARCH_ENABLED"] = "false"
# 볼트 서버는 이 PC 에서 도는 것이니 여기서도 붙는다. 파이와 같은 검색·같은
# 프롬프트로 확인하려는 것이다(볼트 서버가 안 떠 있으면 조용히 꺼진다).
os.environ.setdefault("VAULT_SEARCH_URL", f"http://127.0.0.1:{os.getenv('VAULT_PORT', '8081')}")

sys.path.insert(0, str(BASE_DIR / "ros_nodes"))
import voice_common as vc  # noqa: E402

if not vc.remote_llm_target():
    print("서버에 붙지 못했다. chat --log 로 서버 로그를 보라.", file=sys.stderr)
    sys.exit(1)

print("=" * 60)
print(f" {vc.BOT_NAME} / {vc.remote_llm_target()}  @ {os.environ['REMOTE_LLM_URL']}")
print(" 파이와 같은 시스템 프롬프트·문장분할을 쓴다. 종료: q")
if vc.vault_available():
    print(f" 볼트 검색: 켜짐 ({os.environ['VAULT_SEARCH_URL']})")
else:
    print(" 볼트 검색: 꺼짐 (bash scripts/vault_search.sh 로 띄운다)")
print("=" * 60)

history = []
while True:
    try:
        user = input("\n나 > ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not user:
        continue
    if user.lower() in ("q", "quit", "exit", "종료"):
        break

    t0 = time.time()
    first = None
    print(f"{vc.BOT_NAME} > ", end="", flush=True)
    try:
        context = vc.gather_context(user)
        if context:
            print(f"\n   [{context[0][0]} {len(context[0][1])}자]\n{vc.BOT_NAME} > ",
                  end="", flush=True)
        # 로컬 폴백은 이 창에서 의미가 없으므로 llm=None 을 넘긴다.
        # 원격이 죽으면 vc 가 로컬을 시도하다 죽는데, 그건 아래에서 잡는다.
        for sentence in vc.ask_llm_stream(None, history, user, context):
            if first is None:
                first = time.time() - t0
            print(sentence, end=" ", flush=True)
    except Exception as e:
        print(f"\n[에러] {e}", file=sys.stderr)
        continue
    total = time.time() - t0
    print(f"\n   [첫문장 {first or 0:.2f}s / 전체 {total:.2f}s / {vc.LAST_LLM_SOURCE}]")
