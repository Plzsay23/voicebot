# -*- coding: utf-8 -*-
"""
ROS 2 음성비서 노드들이 공유하는 로직 모음.
STT(SenseVoice) / LLM(EXAONE) / 웹검색(DuckDuckGo) / 웨이크워드 / TTS(edge-tts).
chatbot.py 의 로직을 노드에서 재사용하기 좋게 정리한 것.
"""

import os
import sys
import time
import difflib
import subprocess
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent  # ~/voicebot
load_dotenv(dotenv_path=BASE_DIR / ".env")


# ---------------- 설정 ----------------
SENSEVOICE_DIR = Path(
    os.getenv("SENSEVOICE_DIR", str(BASE_DIR / "models" / "sensevoice_ko"))
)
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
SENSEVOICE_QUANTIZE = os.getenv("SENSEVOICE_QUANTIZE", "true").lower() in (
    "1", "true", "yes", "y",
)

LOCAL_LLM_MODEL_PATH = Path(
    os.getenv(
        "LOCAL_LLM_MODEL_PATH",
        str(BASE_DIR / "models" / "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"),
    )
)
N_CTX = int(os.getenv("N_CTX", "2048"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_THREADS_BATCH = int(os.getenv("N_THREADS_BATCH", "4"))
N_BATCH = int(os.getenv("N_BATCH", "128"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "160"))
# 문장이 이 길이보다 짧으면 다음 문장과 합쳐서 TTS로 보낸다.
# edge-tts 는 문장마다 네트워크 왕복이 있어서, "네." 같은 조각을 따로 보내면
# 합성 오버헤드가 발화 길이보다 커진다.
MIN_TTS_CHARS = int(os.getenv("MIN_TTS_CHARS", "12"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))

BOT_NAME = os.getenv("BOT_NAME", "제리").strip()

EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "ko-KR-SunHiNeural")

WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in (
    "1", "true", "yes", "y",
)
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "kr-kr")

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

SYSTEM_PROMPT = (
    "너는 한국어 음성비서다. 이름은 제리다. "
    "음성 대화이므로 아주 짧게 답한다. 기본 1~2문장, 최대 3문장. "
    "첫 문장에 결론부터 말한다. 인사말·서론·되묻기·요약으로 시작하지 마라. "
    "목록이나 마크다운을 쓰지 말고 말하듯 이어서 답한다. "
    "모르면 모른다고 솔직히 말한다."
)


# ---------------- 텍스트/웨이크워드 ----------------
def normalize_text(text: str) -> str:
    text = text or ""
    for ch in [" ", ".", ",", "?", "!", "~", "\n", "\t"]:
        text = text.replace(ch, "")
    return text.strip()


def has_wake_name(text: str) -> bool:
    if not BOT_NAME:
        return True
    bot = normalize_text(BOT_NAME)
    norm = normalize_text(text)
    if not bot:
        return True
    if bot in norm:
        return True
    # STT 오인식(제리->제야/데리/저리) 대비 퍼지 매칭
    n = len(bot)
    for i in range(max(1, len(norm))):
        for w in (n, n + 1):
            seg = norm[i:i + w]
            if not seg:
                continue
            if difflib.SequenceMatcher(None, seg, bot).ratio() >= 0.5:
                return True
    return False


def remove_wake_name(text: str) -> str:
    if not BOT_NAME:
        return text.strip()
    if BOT_NAME in text:
        result = text.replace(BOT_NAME, "", 1).strip()
        return result.lstrip("야아어,.:;!? ").strip()
    parts = text.strip().split(maxsplit=1)
    if parts:
        first = normalize_text(parts[0])
        bot = normalize_text(BOT_NAME)
        if bot and difflib.SequenceMatcher(None, first, bot).ratio() >= 0.4:
            rest = parts[1] if len(parts) > 1 else ""
            return rest.strip().lstrip("야아어,.:;!? ").strip()
    return text.strip()


# ---------------- STT ----------------
def load_stt():
    from funasr_onnx import SenseVoiceSmall

    model = SenseVoiceSmall(
        str(SENSEVOICE_DIR),
        batch_size=1,
        quantize=SENSEVOICE_QUANTIZE,
        intra_op_num_threads=N_THREADS,
    )
    return model


def _clean_stt_text(text: str) -> str:
    if not text:
        return ""
    try:
        from funasr_onnx.utils.postprocess_utils import (
            rich_transcription_postprocess as post,
        )
        text = post(text)
    except Exception:
        pass
    while "<|" in text and "|>" in text:
        s = text.index("<|")
        e = text.index("|>", s) + 2
        text = text[:s] + text[e:]
    return text.strip()


def transcribe(stt_model, wav_path: str) -> str:
    res = stt_model([wav_path], language=STT_LANGUAGE, use_itn=True)
    raw = res[0] if res else ""
    if isinstance(raw, dict):
        raw = raw.get("text", "")
    return _clean_stt_text(raw)


# ---------------- 웹검색 ----------------
SEARCH_TRIGGERS = (
    "검색", "찾아", "알아봐", "최신", "요즘", "오늘", "지금", "현재", "뉴스",
    "날씨", "기온", "환율", "주가", "가격", "시세", "며칠", "무슨 요일",
    "몇 시", "언제", "누구", "어디", "얼마",
)


def needs_search(text: str) -> bool:
    if not WEB_SEARCH_ENABLED:
        return False
    return any(kw in text for kw in SEARCH_TRIGGERS)


def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
    except Exception:
        return ""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=WEB_SEARCH_REGION,
                               max_results=WEB_SEARCH_RESULTS):
                title = (r.get("title") or "").strip()
                body = (r.get("body") or "").strip()[:180]
                if body:
                    results.append(f"- {title}: {body}")
        return "\n".join(results)
    except Exception:
        return ""


# ---------------- LLM ----------------
def load_llm():
    from llama_cpp import Llama

    return Llama(
        model_path=str(LOCAL_LLM_MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS_BATCH,
        n_batch=N_BATCH,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


def build_messages(history, user_text, context=None):
    system = SYSTEM_PROMPT
    if context:
        system += (
            "\n\n아래는 사용자 질문과 관련된 웹 검색 결과다. "
            "이 정보를 근거로 최신 사실에 맞게 답하라. "
            "검색 결과에 없는 내용은 지어내지 마라.\n\n"
            f"[검색 결과]\n{context}"
        )
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


_SENT_END = ".!?…\n"


def _split_ready(buf: str):
    """buf 에서 '완성된 문장'을 최대한 떼어내고 (문장리스트, 남은버퍼) 를 준다.

    파이4에서 LLM 이 2.2 tok/s 로 기어가므로, 답변이 다 나올 때까지 기다리지 않고
    문장이 완성되는 즉시 TTS 로 넘기려고 쓴다. 문장 경계에서만 자르기 때문에
    말이 중간에 끊기는 일은 생기지 않는다.
    """
    out = []
    start = 0
    for i, ch in enumerate(buf):
        if ch not in _SENT_END:
            continue
        # "3.14", "1.5초" 처럼 소수점을 문장 끝으로 착각하지 않도록
        if ch == "." and i + 1 < len(buf) and buf[i + 1].isdigit():
            continue
        # 종결부호 뒤에 붙는 따옴표·괄호까지 함께 가져간다
        end = i + 1
        while end < len(buf) and buf[end] in '"\'”’)]』」':
            end += 1
        cand = buf[start:end].strip()
        if len(cand) < MIN_TTS_CHARS:
            # 너무 짧으면 자르지 않고 다음 문장과 합친다
            continue
        out.append(cand)
        start = end
    return out, buf[start:]


def ask_llm_stream(llm, history, user_text, context=None):
    """문장이 완성될 때마다 하나씩 내주는 제너레이터. 끝나면 전체 답변을 반환한다.

    `max_tokens` 에 걸려 잘린 경우 마지막 미완성 조각은 버린다. 어중간하게
    끊긴 말을 읽어주는 것보다 한 문장 덜 말하는 편이 낫다.
    """
    messages = build_messages(history, user_text, context)
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stream=True,
    )

    buf = ""
    spoken = []
    finish = None
    for chunk in stream:
        choice = chunk["choices"][0]
        finish = choice.get("finish_reason") or finish
        piece = choice.get("delta", {}).get("content") or ""
        if not piece:
            continue
        buf += piece
        ready, buf = _split_ready(buf)
        for s in ready:
            spoken.append(s)
            yield s

    tail = buf.strip()
    if tail:
        if finish == "length" and tail[-1] not in _SENT_END:
            # 토큰 한도에 걸려 문장이 끝나지 않았다 -> 읽지 않는다
            pass
        else:
            spoken.append(tail)
            yield tail

    answer = " ".join(spoken).strip()
    if not answer:
        answer = "죄송합니다. 답변을 생성하지 못했습니다."
        yield answer

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]
    return answer


def ask_llm(llm, history, user_text, context=None):
    messages = build_messages(history, user_text, context)
    out = llm.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
    )
    answer = (out["choices"][0]["message"]["content"] or "").strip()
    if not answer:
        answer = "죄송합니다. 답변을 생성하지 못했습니다."
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[:-MAX_HISTORY_MESSAGES]
    return answer


# ---------------- TTS ----------------
REPLY_MP3 = BASE_DIR / "reply.mp3"
REPLY_WAV = BASE_DIR / "reply.wav"


def synthesize(text: str, out_wav: Path):
    """text 를 합성해 out_wav 로 쓴다(재생하지 않음).

    재생과 분리해 둔 이유: 문장 단위 스트리밍에서 앞 문장을 재생하는 동안
    뒤 문장을 미리 합성해야 문장 사이가 벌어지지 않는다. 고정 파일명을 쓰면
    그 둘이 같은 파일을 두고 부딪힌다.
    """
    mp3 = out_wav.with_suffix(".mp3")
    for p in (mp3, out_wav):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # 런처가 venv를 activate하지 않고 venv 파이썬을 직접 실행하므로 .venv/bin이
    # PATH에 없다. 실행파일 대신 모듈로 호출해 PATH 의존을 없앤다.
    subprocess.run(
        [sys.executable, "-m", "edge_tts", "--voice", EDGE_TTS_VOICE, "--text", text,
         "--write-media", str(mp3)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(mp3), "-ar", "48000", "-ac", "2", str(out_wav)],
        check=True,
    )
    try:
        mp3.unlink()
    except Exception:
        pass
    return out_wav


def play_wav(path: Path):
    subprocess.run(["paplay", str(path)], check=True)


def synthesize_and_play(text: str):
    synthesize(text, REPLY_WAV)
    play_wav(REPLY_WAV)
