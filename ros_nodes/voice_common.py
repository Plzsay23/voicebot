# -*- coding: utf-8 -*-
"""
ROS 2 음성비서 노드들이 공유하는 로직 모음.
STT(SenseVoice) / LLM(EXAONE) / 웹검색(DuckDuckGo) / 웨이크워드 / TTS(piper 또는 edge-tts).
chatbot.py 의 로직을 노드에서 재사용하기 좋게 정리한 것.
"""

import os
import re
import sys
import json
import time
import difflib
import subprocess
import urllib.error
import urllib.parse
import urllib.request
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
# 참고자료가 붙었을 때의 토큰 한도(로컬/원격). 따로 두는 이유:
# 자료를 붙이면 EXAONE 이 "요약해 드리겠습니다" 모드로 들어가 번호 목록을 뱉는다.
# 프롬프트로 "두 문장 안에" 를 못 박아도 지키다 말다 한다(2026-07-28 실측).
# 파이에서는 그 초과분이 그대로 대기시간이라 프롬프트가 아니라 한도로 막는다.
# 프롬프트는 최선노력이고 한도는 보장이다.
MAX_TOKENS_CONTEXT = int(os.getenv("MAX_TOKENS_CONTEXT", "120"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.1"))

# ---- 원격 LLM (PC) ----
# PC 에서 OpenAI 호환 서버가 돌고 있으면 그쪽에 생성을 맡기고, 안 돌고 있으면
# 파이의 로컬 gguf 로 떨어진다. 파이는 24시간 켜두고 PC 는 켤 때만 쓰는 구성.
#
# REMOTE_LLM_URL 은 /v1 까지 포함한 주소:
#   ollama          http://<PC>:11434/v1     (OLLAMA_HOST=0.0.0.0 필요)
#   llama.cpp server http://<PC>:8080/v1
#   LM Studio       http://<PC>:1234/v1
# 비워두면 원격 기능 자체를 끈다(= 기존 동작과 완전히 동일).
REMOTE_LLM_URL = os.getenv("REMOTE_LLM_URL", "").strip().rstrip("/")
# 비워두면 /v1/models 의 첫 모델을 쓴다.
REMOTE_LLM_MODEL = os.getenv("REMOTE_LLM_MODEL", "").strip()
REMOTE_LLM_API_KEY = os.getenv("REMOTE_LLM_API_KEY", "").strip()
# 생존 확인 타임아웃. PC 가 꺼져 있으면 이 시간만큼 손해 보고 로컬로 간다.
# LAN 이면 정상 응답은 수 ms 이므로 짧게 잡아도 된다.
REMOTE_LLM_PROBE_TIMEOUT = float(os.getenv("REMOTE_LLM_PROBE_TIMEOUT", "1.0"))
# 생존 확인 결과 캐시(초). 매 발화마다 찔러보지 않기 위한 것.
REMOTE_LLM_PROBE_TTL = float(os.getenv("REMOTE_LLM_PROBE_TTL", "10"))
# 생성 요청 타임아웃(초). 청크 사이 간격 기준.
REMOTE_LLM_TIMEOUT = float(os.getenv("REMOTE_LLM_TIMEOUT", "30"))
# PC 는 빠르니 토큰 한도를 파이보다 넉넉히 준다.
REMOTE_MAX_TOKENS = int(os.getenv("REMOTE_MAX_TOKENS", "256"))
# 자료가 붙었을 때. PC 는 빨라도 **읽어주는 데 걸리는 시간**은 같으므로 묶는다.
REMOTE_MAX_TOKENS_CONTEXT = int(os.getenv("REMOTE_MAX_TOKENS_CONTEXT", "200"))

BOT_NAME = os.getenv("BOT_NAME", "제리").strip()

# 웨이크워드 퍼지 매칭 임계값.
#
# 0.5 였다. 원본 SenseVoice 가 "제리"를 "제야/재이/젤리"로 흘려 듣던 시절,
# 그걸 건져내려고 느슨하게 잡아둔 값이다. 파인튜닝 후에는 STT 가 "제리"를
# 글자 그대로 받아 적으므로(홀드아웃 23/23) 그 보상책이 손해만 남긴다.
#
# 2026-07-27 홀드아웃 75발화 실측 — 재현율 / 오검출:
#     0.5   파인튜닝 100% / 23.1%(52개 중 12개)   ← 아무 말에나 깨어난다
#     0.6   파인튜닝 100% /  0.0%                  ← 채택
#   "처리가 완료되었습니다", "거리가 얼마나 되나요", "제조 일자를 확인해 보세요"
#   같은 문장이 0.5 에서 전부 통과했다.
#
# 원본 모델로 되돌린다면 0.5 로 낮춰야 한다(0.6 에서 재현율 69.6%로 떨어진다).
WAKE_MATCH_RATIO = float(os.getenv("WAKE_MATCH_RATIO", "0.6"))

# ---- TTS ----
# "piper" = 온디바이스(오프라인), "edge" = edge-tts(네트워크).
# piper 를 기본으로 두되, 모델이 없거나 합성이 실패하면 edge 로 자동 폴백한다.
# 되돌리려면 .env 에 TTS_BACKEND=edge 한 줄.
TTS_BACKEND = os.getenv("TTS_BACKEND", "piper").strip().lower()

EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "ko-KR-SunHiNeural")

# 한국어 Piper 음성은 2026-07 기준 이것 하나뿐이다(KSS 단일화자, 22.05kHz).
# scripts/setup_piper.sh 가 PIPER_DATA_DIR 아래로 받아둔다.
PIPER_VOICE = os.getenv("PIPER_VOICE", "ko_KR-kss-medium")
PIPER_DATA_DIR = Path(os.getenv("PIPER_DATA_DIR", str(BASE_DIR / "models" / "piper")))
# 문장 사이 무음(초). 스트리밍에서 문장을 따로 합성하므로 0 이면 붙어 들린다.
PIPER_SENTENCE_SILENCE = float(os.getenv("PIPER_SENTENCE_SILENCE", "0.2"))

# 재생 음량 배수. 두 백엔드 공통(합성 후 ffmpeg 단계에서 적용).
# 시스템 볼륨을 건드리지 않으므로 다른 소리(경고음 등)에는 영향이 없다.
# 1.0 이 원본. 너무 낮추면 에코 억제엔 유리하지만 잘 안 들린다.
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.7"))

# 문장이 이 길이보다 짧으면 다음 문장과 합쳐서 TTS로 보낸다.
# edge-tts 는 문장마다 네트워크 왕복이 있어서 "네." 같은 조각을 따로 보내면
# 합성 오버헤드가 발화 길이보다 커진다. piper 는 로컬이라 그 비용이 없으므로
# 합치지 않고 바로 내보내는 편이 첫 소리가 빨리 난다.
MIN_TTS_CHARS = int(os.getenv("MIN_TTS_CHARS", "0" if TTS_BACKEND == "piper" else "12"))

WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "true").lower() in (
    "1", "true", "yes", "y",
)
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))
WEB_SEARCH_REGION = os.getenv("WEB_SEARCH_REGION", "kr-kr")

# ---- 볼트 RAG (PC 의 옵시디언 리서치위키) ----
# 볼트를 파이에 복제하지 않는다. PC 가 켜져 있을 때만 scripts/vault_server.py 가
# 뜨고 파이는 발화마다 거기에 물어본다. 꺼져 있으면 조용히 RAG 없이 답한다 —
# REMOTE_LLM_URL 과 정확히 같은 정책이라 "PC 켜면 똑똑해진다" 로 일관된다.
# 비워두면 기능 자체가 꺼진다.
VAULT_SEARCH_URL = os.getenv("VAULT_SEARCH_URL", "").strip().rstrip("/")
VAULT_SEARCH_RESULTS = int(os.getenv("VAULT_SEARCH_RESULTS", "3"))
VAULT_PROBE_TIMEOUT = float(os.getenv("VAULT_PROBE_TIMEOUT", "1.0"))
VAULT_PROBE_TTL = float(os.getenv("VAULT_PROBE_TTL", "10"))
VAULT_SEARCH_TIMEOUT = float(os.getenv("VAULT_SEARCH_TIMEOUT", "3.0"))

# 컨텍스트 예산(문자 수). **이게 이 기능의 진짜 비용이다.**
# 파이4 에서는 프롬프트 토큰이 그대로 대기시간으로 돌아온다(생성 2.2 tok/s).
# 웹검색이 이미 3건×180자 ≈ 500자를 넣고 실사용 중이므로 그 봉투를 그대로 쓴다.
# PC 가 답할 때는 GPU 라 비용이 없어서 넉넉히 준다.
VAULT_MAX_CHARS_LOCAL = int(os.getenv("VAULT_MAX_CHARS_LOCAL", "500"))
VAULT_MAX_CHARS_REMOTE = int(os.getenv("VAULT_MAX_CHARS_REMOTE", "1500"))

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))

SYSTEM_PROMPT = (
    "너는 한국어 음성비서다. 이름은 제리다. "
    "음성 대화이므로 아주 짧게 답한다. 기본 1~2문장, 최대 3문장. "
    "첫 문장에 결론부터 말한다. 인사말·서론·되묻기·요약으로 시작하지 마라. "
    "목록이나 마크다운을 쓰지 말고 말하듯 이어서 답한다. "
    "이모지·이모티콘·특수기호는 절대 쓰지 마라. 소리 내어 읽을 수 있는 말만 쓴다. "
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
    # STT 오인식 대비 퍼지 매칭. 임계값은 STT 성능에 맞춰 조정해야 한다.
    n = len(bot)
    for i in range(max(1, len(norm))):
        for w in (n, n + 1):
            seg = norm[i:i + w]
            # 문자열 끝에서는 슬라이스가 짧게 잘린다. 한 글자짜리 조각을 그대로
            # 비교하면 ratio("리","제리")=0.667 이라 임계값 0.6 을 넘어버려서
            # "빨리", "언제" 처럼 리/제 로 끝나는 문장이 전부 웨이크워드로 잡혔다.
            if len(seg) < n:
                continue
            if difflib.SequenceMatcher(None, seg, bot).ratio() >= WAKE_MATCH_RATIO:
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


# ---------------- 볼트 검색 (PC) ----------------
# 볼트를 뒤져볼 신호. 넉넉하게 잡았다 — 좁게 잡았더니 "TTS 를 왜 바꿨는지 기록
# 찾아줘" 가 트리거에 안 걸려서, 답이 볼트에 그대로 있는데도 LLM 이 "내부
# 데이터베이스에서 확인 가능합니다" 라고 지어냈다. 지어내는 것보다는 헛검색이 낫다.
#
# 넓혀도 안전한 이유는 VAULT_MIN_SCORE 가 뒤를 받치기 때문이다. 트리거는 "찾아볼까"
# 만 정하고, 실제로 컨텍스트를 붙일지는 검색 점수가 정한다.
VAULT_TRIGGERS = (
    "논문", "연구", "위키", "노트", "볼트", "메모", "자료", "문헌", "레퍼런스",
    "베이스라인", "실험", "알고리즘", "프로젝트", "어디까지", "진행",
    "정리", "적어", "기록", "찾아", "알아봐", "뭐였", "어땠", "왜 ",
)

# 검색 점수 문턱. 이 아래는 "관련 자료 없음" 으로 본다.
#
# 2026-07-28 실측(178문서 볼트, 질문 15개): 관련 있는 질문의 1위 점수는 15.4~66.8,
# 무관한 질문("피자 시켜줘", "지금 몇 시야")은 3.9~10.7 로 깨끗하게 갈렸다.
# 13 은 그 사이다. 볼트가 커지면 다시 재야 한다.
VAULT_MIN_SCORE = float(os.getenv("VAULT_MIN_SCORE", "13"))

# (확인시각, 살아있나) — 원격 LLM 과 같은 방식의 캐시. 발화마다 찔러보지 않는다.
_vault_probe = (0.0, False)


def needs_vault(text: str) -> bool:
    if not VAULT_SEARCH_URL:
        return False
    return any(kw in text for kw in VAULT_TRIGGERS)


def vault_available() -> bool:
    """PC 의 볼트 서버가 지금 떠 있나. 결과는 TTL 동안 캐시한다."""
    global _vault_probe
    if not VAULT_SEARCH_URL:
        return False
    checked, ok = _vault_probe
    now = time.time()
    if now - checked < VAULT_PROBE_TTL:
        return ok
    try:
        with urllib.request.urlopen(
            f"{VAULT_SEARCH_URL}/health", timeout=VAULT_PROBE_TIMEOUT
        ) as r:
            ok = bool(json.load(r).get("ok"))
    except Exception as e:
        # PC 가 꺼져 있는 정상적인 경우도 여기로 온다. 한 줄만 남긴다.
        print(f"[vault] 서버 없음({type(e).__name__}) → 볼트 없이 답한다", file=sys.stderr)
        ok = False
    _vault_probe = (now, ok)
    return ok


def vault_search(query: str, max_chars: int = None) -> str:
    """PC 의 볼트에서 관련 조각을 받아 컨텍스트 문자열로 만든다. 없으면 "".

    예산(max_chars)을 서버에 넘겨 **서버가 자르게** 한다. 받아놓고 여기서
    자르면 마지막 결과가 통째로 날아가서, 예산을 넘기는 쪽이 손해를 본다.
    """
    global _vault_probe
    if not vault_available():
        return ""
    if max_chars is None:
        # PC 가 생성까지 맡으면 프롬프트 비용이 없으니 넉넉히, 파이가 답하면 짧게.
        max_chars = (VAULT_MAX_CHARS_REMOTE if remote_llm_target()
                     else VAULT_MAX_CHARS_LOCAL)
    params = urllib.parse.urlencode({
        "q": query, "k": VAULT_SEARCH_RESULTS, "maxchars": max_chars,
        "minscore": VAULT_MIN_SCORE,
    })
    try:
        with urllib.request.urlopen(
            f"{VAULT_SEARCH_URL}/search?{params}", timeout=VAULT_SEARCH_TIMEOUT
        ) as r:
            data = json.load(r)
    except Exception as e:
        _vault_probe = (0.0, False)  # 다음 발화에서 다시 확인하게 만든다
        print(f"[vault] 검색 실패({type(e).__name__})", file=sys.stderr)
        return ""
    lines = []
    for item in data.get("results") or []:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        title = (item.get("title") or item.get("path") or "").strip()
        lines.append(f"- {title}: {text}" if title else f"- {text}")
    return "\n".join(lines)


def gather_context(user_text: str):
    """질문에 붙일 참고자료를 모은다. [(라벨, 본문), ...] 또는 None.

    볼트를 먼저 본다. 개인 노트가 웹 스니펫보다 이 사용자에게 정확하고,
    둘 다 붙이면 컨텍스트 예산을 두 배로 쓰기 때문이다. 볼트가 비었을 때만
    웹으로 내려간다.

    웹검색 트리거에 걸린 질문도 일단 볼트를 본다. 볼트 검색은 PC 에서 1ms 짜리라
    공짜에 가깝고, 점수가 낮으면 어차피 빈손으로 돌아온다. "제리, 내 STT 결과
    어땠지" 처럼 두 트리거의 경계에 있는 질문을 놓치지 않으려는 것.
    """
    if needs_vault(user_text) or needs_search(user_text):
        body = vault_search(user_text)
        if body:
            return [("내 연구 노트", body)]
    if needs_search(user_text):
        body = web_search(user_text)
        if body:
            return [("웹 검색 결과", body)]
    return None


# ---------------- LLM ----------------
# 마지막 생성에 실제로 쓰인 백엔드("remote"/"local"). 로그용.
LAST_LLM_SOURCE = "local"

# (확인시각, 모델명 또는 None) — None 은 "죽어 있음".
_remote_probe = (0.0, None)


def _remote_headers():
    h = {"Content-Type": "application/json"}
    if REMOTE_LLM_API_KEY:
        h["Authorization"] = f"Bearer {REMOTE_LLM_API_KEY}"
    return h


def _probe_remote():
    """PC 서버에 물어 쓸 모델명을 정한다. 죽어 있으면 None."""
    req = urllib.request.Request(
        f"{REMOTE_LLM_URL}/models", headers=_remote_headers()
    )
    with urllib.request.urlopen(req, timeout=REMOTE_LLM_PROBE_TIMEOUT) as r:
        data = json.load(r)
    ids = [m.get("id") for m in (data.get("data") or []) if m.get("id")]
    if REMOTE_LLM_MODEL:
        if ids and REMOTE_LLM_MODEL not in ids:
            print(
                f"[llm] 원격에 {REMOTE_LLM_MODEL} 이 없다(있는 것: {', '.join(ids)}). "
                "그대로 요청은 해본다.",
                file=sys.stderr,
            )
        return REMOTE_LLM_MODEL
    return ids[0] if ids else None


def remote_llm_target():
    """지금 원격을 쓸 수 있으면 모델명, 못 쓰면 None. 결과는 TTL 동안 캐시한다."""
    global _remote_probe
    if not REMOTE_LLM_URL:
        return None
    checked, model = _remote_probe
    now = time.time()
    if now - checked < REMOTE_LLM_PROBE_TTL:
        return model
    try:
        model = _probe_remote()
        if model is None:
            print("[llm] 원격 서버에 모델이 없다 → 로컬 사용", file=sys.stderr)
    except Exception as e:
        # PC 가 꺼져 있는 정상적인 경우도 여기로 온다. 시끄럽지 않게 한 줄만.
        print(f"[llm] 원격 없음({type(e).__name__}) → 로컬 사용", file=sys.stderr)
        model = None
    _remote_probe = (now, model)
    return model


def _mark_remote_down():
    global _remote_probe
    _remote_probe = (time.time(), None)


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


def _as_blocks(context):
    """context 를 [(라벨, 본문)] 로 정규화한다.

    문자열이 그대로 오면 예전처럼 웹 검색 결과로 본다(chatbot.py 등 옛 호출부 호환).
    """
    if not context:
        return []
    if isinstance(context, str):
        return [("웹 검색 결과", context)]
    blocks = []
    for item in context:
        if isinstance(item, dict):
            label, body = item.get("label", "참고자료"), item.get("body", "")
        else:
            label, body = item
        if body:
            blocks.append((label, body))
    return blocks


def build_messages(history, user_text, context=None):
    system = SYSTEM_PROMPT
    blocks = _as_blocks(context)
    if blocks:
        # **순서가 중요하다.** 참고자료를 먼저 깔고 지시를 맨 뒤에 둔다.
        # 지시를 앞에 두고 자료를 뒤에 붙였더니 EXAONE 이 "요약해 드리겠습니다"
        # 모드로 들어가 번호 목록을 뱉었다(1500자를 붙이면 매번). 모델이 마지막에
        # 읽는 것이 자료가 아니라 "짧게 말하라" 가 되도록 뒤집은 것이다.
        system += "\n\n아래는 사용자 질문과 관련해 찾아온 참고자료다."
        for label, body in blocks:
            system += f"\n\n[{label}]\n{body}"
        system += (
            "\n\n이 자료를 근거로 답하라. 자료에 없는 내용은 지어내지 마라. "
            # 검색은 질의어가 조금이라도 겹치면 무언가를 돌려준다. 엉뚱한 자료를
            # 억지로 끼워 맞춰 답하는 것을 막으려면 무시해도 된다고 알려줘야 한다.
            "자료가 질문과 상관없으면 그냥 무시하고 아는 대로 답하라.\n"
            "자료가 아무리 길어도 답은 음성으로 읽어줄 것이므로 두 문장 안에 끝낸다. "
            "번호·항목·목록으로 나열하지 마라. 자료를 요약하지 말고 질문에만 답하라."
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
        # 마침표는 뒤에 공백이 와야 문장 끝이다. "3.14", "eval_stt.py",
        # "192.168.0.41" 처럼 소수점·파일명·IP 를 문장 끝으로 착각하지 않는다.
        # 볼트 노트에는 파일명이 널려 있어서 이게 없으면 "eval stt." / "py가…" 로
        # 끊어 읽는다.
        #
        # 마침표가 버퍼의 마지막 글자면 **아직 판단하지 않고 다음 조각을 기다린다.**
        # 문장 하나가 토큰 하나만큼 늦게 나가지만(파이에서 ~0.5초), 끊어 읽는 것보다
        # 낫다. 스트림이 거기서 끝나면 _emit_sentences 의 tail 처리가 받아준다.
        if ch == ".":
            if i + 1 >= len(buf):
                continue
            if not buf[i + 1].isspace():
                continue
            # "1. 데이터 조정" 같은 목록 번호. 여기서 자르면 "일"만 읽는 한 글자
            # 문장이 튀어나온다. 프롬프트로 목록을 금지해도 모델이 가끔 어긴다.
            # 마침표 앞을 숫자만큼 되짚어, 그 앞이 공백(또는 처음)이면 번호로 본다.
            # 문장 중간의 "2." 도 잡으려면 start 기준이 아니라 이렇게 봐야 한다.
            # 부수 효과로 "2026. 7. 28." 같은 날짜 표기도 안 쪼개진다.
            j = i
            while j > 0 and buf[j - 1].isdigit():
                j -= 1
            if j < i and (j == 0 or buf[j - 1].isspace()):
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


def _pieces_local(llm, messages, max_tokens=None):
    """로컬 llama.cpp 에서 (토큰조각, finish_reason) 을 흘려준다."""
    stream = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens or MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stream=True,
    )
    for chunk in stream:
        choice = chunk["choices"][0]
        yield (choice.get("delta", {}).get("content") or "",
               choice.get("finish_reason"))


def _pieces_remote(messages, model, max_tokens=None):
    """PC 의 OpenAI 호환 서버에서 같은 모양으로 흘려준다(SSE)."""
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens or REMOTE_MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{REMOTE_LLM_URL}/chat/completions", data=body,
        headers=_remote_headers(), method="POST",
    )
    with urllib.request.urlopen(req, timeout=REMOTE_LLM_TIMEOUT) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            yield (choice.get("delta", {}).get("content") or "",
                   choice.get("finish_reason"))


def _emit_sentences(pieces, spoken):
    """토큰 조각 스트림을 완성된 문장 단위로 내준다.

    `max_tokens` 에 걸려 잘린 경우 마지막 미완성 조각은 버린다. 어중간하게
    끊긴 말을 읽어주는 것보다 한 문장 덜 말하는 편이 낫다.
    """
    buf = ""
    finish = None
    for piece, fr in pieces:
        finish = fr or finish
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


def ask_llm_stream(llm, history, user_text, context=None):
    """문장이 완성될 때마다 하나씩 내주는 제너레이터. 끝나면 전체 답변을 반환한다.

    PC 서버가 살아 있으면 그쪽에서 생성하고, 없거나 도중에 끊기면 로컬로 간다.
    단, 이미 한 문장이라도 말한 뒤에 끊긴 경우에는 로컬로 이어붙이지 않는다.
    같은 답을 두 번 말하거나 앞뒤가 안 맞는 말이 이어지는 편이 더 나쁘다.
    """
    global LAST_LLM_SOURCE
    messages = build_messages(history, user_text, context)
    spoken = []
    # 참고자료가 붙었으면 더 짧게 묶는다(위 MAX_TOKENS_CONTEXT 주석 참고).
    has_ctx = bool(_as_blocks(context))
    local_cap = MAX_TOKENS_CONTEXT if has_ctx else MAX_TOKENS
    remote_cap = REMOTE_MAX_TOKENS_CONTEXT if has_ctx else REMOTE_MAX_TOKENS

    model = remote_llm_target()
    if model:
        LAST_LLM_SOURCE = "remote"
        try:
            for s in _emit_sentences(_pieces_remote(messages, model, remote_cap), spoken):
                yield s
        except Exception as e:
            _mark_remote_down()
            if spoken:
                print(f"[llm] 원격 생성이 중간에 끊겼다: {e}", file=sys.stderr)
            else:
                print(f"[llm] 원격 실패 → 로컬로 재생성: {e}", file=sys.stderr)
                LAST_LLM_SOURCE = "local"
                for s in _emit_sentences(_pieces_local(llm, messages, local_cap), spoken):
                    yield s
    else:
        LAST_LLM_SOURCE = "local"
        for s in _emit_sentences(_pieces_local(llm, messages, local_cap), spoken):
            yield s

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
    """한 번에 전체 답변을 돌려주는 버전. 원격/로컬 선택과 history 갱신은
    ask_llm_stream 이 하는 것을 그대로 쓴다(분기 로직을 두 벌 두지 않는다)."""
    parts = list(ask_llm_stream(llm, history, user_text, context))
    return " ".join(parts).strip()


# ---------------- TTS ----------------
REPLY_MP3 = BASE_DIR / "reply.mp3"
REPLY_WAV = BASE_DIR / "reply.wav"


# 합성 전에 지워야 하는 문자들.
#
# 프롬프트로 이모지를 금지해도 LLM 은 가끔 어긴다. 그때 piper 는 이모지를
# 음소로 바꾸지 못해 문장을 통째로 이상하게 읽거나 실패한다(edge-tts 는
# "웃는 얼굴" 처럼 이름을 읽어버린다). 어느 쪽이든 원하는 소리가 아니므로
# 합성 직전에 한 번 더 걸러낸다.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"       # 이모티콘·픽토그램·국기·카드 등 SMP 전역
    "\U0001FB00-\U0001FBFF"       # 레거시 컴퓨팅 기호
    "\U000E0020-\U000E007F"       # 태그 문자(국기 시퀀스 꼬리)
    "←-⇿"               # 화살표
    "⌀-⏿"               # 기타 기술기호(⏰ ⌛ …)
    "①-⓿"               # 원문자
    "■-➿"               # 도형·기타기호·딩뱃(★ ☀ ✅ ❤ …)
    "⬀-⯿"               # 추가 화살표·도형
    "︀-️"               # 변이 선택자(FE0F 이모지 표현)
    "​-‍⁠﻿"   # 폭 없는 공백·ZWJ·BOM
    "⃣©®™"    # 키캡·저작권·등록상표·상표
    "]+",
    flags=re.UNICODE,
)
# 마크다운 잔재. 그대로 두면 별표를 읽거나 운율이 깨진다.
_MARKUP_RE = re.compile(r"[*_`#>|]+")
# 읽을 거리가 남아 있는지 판단할 때 쓴다(한글/영문/숫자).
_SPEAKABLE_RE = re.compile(r"[0-9A-Za-z가-힣ㄱ-ㆎ]")


def strip_for_tts(text: str) -> str:
    """이모지·마크다운 기호를 걷어낸 '읽을 수 있는' 문장을 돌려준다.

    읽을 내용이 하나도 남지 않으면(예: 이모지만 있는 문장) 빈 문자열.
    """
    if not text:
        return ""
    text = _EMOJI_RE.sub(" ", text)
    text = _MARKUP_RE.sub(" ", text)
    # 기호를 지우면서 생긴 공백을 정리한다. 문장부호 앞 공백도 붙여준다.
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:)\]}])", r"\1", text)
    if not _SPEAKABLE_RE.search(text):
        return ""
    return text


def piper_available() -> bool:
    """piper 모듈과 음성 파일이 둘 다 있는지."""
    if not (PIPER_DATA_DIR / f"{PIPER_VOICE}.onnx").exists():
        return False
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    return True


def _resample_to_output(src: Path, out_wav: Path):
    """스피커 싱크에 맞춰 48k 스테레오로 맞추고 음량을 적용한다."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if abs(TTS_VOLUME - 1.0) > 1e-3:
        cmd += ["-filter:a", f"volume={TTS_VOLUME}"]
    cmd += ["-ar", "48000", "-ac", "2", str(out_wav)]
    subprocess.run(cmd, check=True)


def _synthesize_piper(text: str, out_wav: Path):
    raw = out_wav.with_name(out_wav.stem + "_piper.wav")
    try:
        # 텍스트를 `--` 뒤 위치인자로 넘긴다. 텍스트가 '-' 로 시작해도 안전하다.
        subprocess.run(
            [sys.executable, "-m", "piper",
             "-m", PIPER_VOICE,
             "--data-dir", str(PIPER_DATA_DIR),
             "--sentence-silence", str(PIPER_SENTENCE_SILENCE),
             "-f", str(raw),
             "--", text],
            check=True,
        )
        _resample_to_output(raw, out_wav)
    finally:
        try:
            raw.unlink()
        except Exception:
            pass
    return out_wav


def _synthesize_edge(text: str, out_wav: Path):
    mp3 = out_wav.with_suffix(".mp3")
    try:
        # 런처가 venv를 activate하지 않고 venv 파이썬을 직접 실행하므로 .venv/bin이
        # PATH에 없다. 실행파일 대신 모듈로 호출해 PATH 의존을 없앤다.
        subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", EDGE_TTS_VOICE,
             "--text", text, "--write-media", str(mp3)],
            check=True,
        )
        _resample_to_output(mp3, out_wav)
    finally:
        try:
            mp3.unlink()
        except Exception:
            pass
    return out_wav


def synthesize(text: str, out_wav: Path):
    """text 를 합성해 out_wav 로 쓴다(재생하지 않음). 읽을 게 없으면 None.

    재생과 분리해 둔 이유: 문장 단위 스트리밍에서 앞 문장을 재생하는 동안
    뒤 문장을 미리 합성해야 문장 사이가 벌어지지 않는다. 고정 파일명을 쓰면
    그 둘이 같은 파일을 두고 부딪힌다.
    """
    text = strip_for_tts(text)
    if not text:
        return None

    for p in (out_wav.with_suffix(".mp3"), out_wav):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    if TTS_BACKEND == "piper":
        if piper_available():
            try:
                return _synthesize_piper(text, out_wav)
            except Exception as e:
                # 여기서 죽으면 비서가 벙어리가 된다. 말은 나오게 하고 로그만 남긴다.
                print(f"[tts] piper 실패, edge-tts 로 폴백: {e}", file=sys.stderr)
        else:
            print(
                f"[tts] piper 음성 없음({PIPER_DATA_DIR / (PIPER_VOICE + '.onnx')}) "
                "→ edge-tts 사용. scripts/setup_piper.sh 를 돌려라.",
                file=sys.stderr,
            )
    return _synthesize_edge(text, out_wav)


def play_wav(path: Path):
    subprocess.run(["paplay", str(path)], check=True)


def synthesize_and_play(text: str):
    if synthesize(text, REPLY_WAV) is None:
        return
    play_wav(REPLY_WAV)
