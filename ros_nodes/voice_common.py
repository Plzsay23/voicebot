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
    "친절하고 자연스러운 한국어로 대화한다. "
    "음성으로 듣기 좋게 핵심만 2~3문장 이내로 짧게 답한다. "
    "장황하게 늘어놓지 말고 바로 답한다. "
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


def synthesize_and_play(text: str):
    for p in (REPLY_MP3, REPLY_WAV):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    subprocess.run(
        ["edge-tts", "--voice", EDGE_TTS_VOICE, "--text", text,
         "--write-media", str(REPLY_MP3)],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(REPLY_MP3), "-ar", "48000", "-ac", "2", str(REPLY_WAV)],
        check=True,
    )
    subprocess.run(["paplay", str(REPLY_WAV)], check=True)
