#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문장 분할 회귀 테스트.

`_split_ready` / `_emit_sentences` 는 **모든 발화가 지나가는** 코드다. 여기가
깨지면 RAG 같은 특정 기능이 아니라 말하기 전체가 이상해지는데, 증상이
"가끔 이상하게 끊어 읽는다" 라서 눈치채기 어렵다. 그래서 따로 묶어뒀다.

    파이:  ~/voicebot/.venv/bin/python test_split.py
    PC:    .venv-llm/bin/python test_split.py

LLM 도 모델도 로딩하지 않는다. venv 파이썬이 필요한 건 voice_common 이
import 하는 python-dotenv 때문뿐이다.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("REMOTE_LLM_URL", "")
os.environ["MIN_TTS_CHARS"] = "0"          # piper 기본값 기준으로 본다

sys.path.insert(0, str(Path(__file__).resolve().parent / "ros_nodes"))
import voice_common as vc  # noqa: E402

fails = []


def emit(text, chunk=3, finish="stop"):
    """text 를 chunk 글자씩 흘려 넣어 실제 스트리밍처럼 문장을 뽑는다."""
    def pieces():
        for i in range(0, len(text), chunk):
            yield (text[i:i + chunk], None)
        yield ("", finish)
    return list(vc._emit_sentences(pieces(), []))


def check(name, text, expected, **kw):
    got = emit(text, **kw)
    ok = got == expected
    print(("  OK   " if ok else "  FAIL ") + name)
    if not ok:
        print("        기대:", expected)
        print("        실제:", got)
        fails.append(name)


check("보통 문장 두 개",
      "네 알겠습니다. 지금 확인해 볼게요.",
      ["네 알겠습니다.", "지금 확인해 볼게요."])

# 아래 셋은 "마침표 뒤에 공백이 와야 문장 끝" 규칙이 지키는 것들.
check("소수점",
      "정확도는 3.14 퍼센트 올랐습니다. 끝입니다.",
      ["정확도는 3.14 퍼센트 올랐습니다.", "끝입니다."])

check("파일명",
      "eval_stt.py 가 판정했습니다. 그게 문제였어요.",
      ["eval_stt.py 가 판정했습니다.", "그게 문제였어요."])

check("IP 주소",
      "주소는 192.168.0.41 입니다.",
      ["주소는 192.168.0.41 입니다."])

# 목록 번호에서 자르면 "일"만 읽는 한 글자 문장이 나간다.
check("목록 번호",
      "이유는 둘입니다. 1. 속도 2. 정확도",
      ["이유는 둘입니다.", "1. 속도 2. 정확도"])

check("물음표·느낌표",
      "정말요? 놀랍네요! 확인해 볼게요.",
      ["정말요?", "놀랍네요!", "확인해 볼게요."])

check("마지막 문장에 마침표 없음",
      "네 그렇습니다. 확인 중이에요",
      ["네 그렇습니다.", "확인 중이에요"])

check("토큰 한도로 잘림 → 미완성 조각은 버린다",
      "첫 문장입니다. 두 번째 문장이 중간에",
      ["첫 문장입니다."], finish="length")

check("따옴표가 마침표 뒤에",
      '그가 "안녕." 이라고 했다. 끝.',
      ['그가 "안녕." 이라고 했다.', "끝."])

# 조각 크기가 결과를 바꾸면 안 된다(스트리밍이라 경계가 어디든 올 수 있다).
check("한 글자씩 흘려도 같은 결과",
      "네 알겠습니다. 지금 확인해 볼게요.",
      ["네 알겠습니다.", "지금 확인해 볼게요."], chunk=1)

check("통째로 한 번에 와도 같은 결과",
      "네 알겠습니다. 지금 확인해 볼게요.",
      ["네 알겠습니다.", "지금 확인해 볼게요."], chunk=999)

print()
print("실패 %d개" % len(fails))
sys.exit(1 if fails else 0)
