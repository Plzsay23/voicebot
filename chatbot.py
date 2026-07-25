# -*- coding: utf-8 -*-

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from llama_cpp import Llama
from funasr_onnx import SenseVoiceSmall

try:
    from funasr_onnx.utils.postprocess_utils import (
        rich_transcription_postprocess as sv_postprocess,
    )
except Exception:  # pragma: no cover - postprocess 위치가 버전마다 다를 수 있음
    sv_postprocess = None


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


# 로컬 STT: SenseVoiceSmall (ONNX). 폴더에 model.onnx / am.mvn /
# config.yaml / chn_jpn_yue_eng_ko_spectok.bpe.model 이 있어야 한다.
SENSEVOICE_DIR = Path(
    os.getenv("SENSEVOICE_DIR", str(BASE_DIR / "models" / "sensevoice_ko"))
)
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ko")
SENSEVOICE_QUANTIZE = os.getenv("SENSEVOICE_QUANTIZE", "false").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

LOCAL_LLM_MODEL_PATH = Path(
    os.getenv(
        "LOCAL_LLM_MODEL_PATH",
        str(BASE_DIR / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    )
)

EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "ko-KR-SunHiNeural")

BOT_NAME = os.getenv("BOT_NAME", "제리").strip()
REQUIRE_WAKE_NAME = os.getenv("REQUIRE_WAKE_NAME", "true").lower() in (
    "1",
    "true",
    "yes",
    "y",
)

# 비어 있으면 시스템 기본 입력 장치(ReSpeaker)를 사용한다.
INPUT_DEVICE = os.getenv("INPUT_DEVICE", "").strip()
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
CHANNELS = int(os.getenv("CHANNELS", "1"))
RECORD_SECONDS = float(os.getenv("RECORD_SECONDS", "5"))

RECORD_PATH = BASE_DIR / "input.wav"
REPLY_MP3_PATH = BASE_DIR / "reply.mp3"
REPLY_WAV_PATH = BASE_DIR / "reply.wav"


# Pi 4B 4GB용 Qwen 속도/품질 타협 설정
N_CTX = int(os.getenv("N_CTX", "512"))
N_THREADS = int(os.getenv("N_THREADS", "4"))
N_THREADS_BATCH = int(os.getenv("N_THREADS_BATCH", "4"))
N_BATCH = int(os.getenv("N_BATCH", "128"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "4"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "60"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
REPEAT_PENALTY = float(os.getenv("REPEAT_PENALTY", "1.08"))


SYSTEM_PROMPT = (
    "너는 라즈베리파이 4B에서 실행되는 작은 한국어 음성비서다. "
    "답변은 짧고 명확하게 한국어로 한다. "
    "음성으로 듣기 좋게 1~2문장으로 답한다. "
    "모르면 모른다고 말한다."
)


def safe_print(*args, sep=" ", end="\n"):
    text = sep.join(str(a) for a in args) + end

    try:
        sys.stdout.buffer.write(text.encode("utf-8", errors="replace"))
        sys.stdout.buffer.flush()
    except Exception:
        try:
            print(text.encode("ascii", errors="replace").decode("ascii"), end="")
        except Exception:
            pass


def check_binary(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"필수 명령어가 없습니다: {name}")


def check_env():
    if not SENSEVOICE_DIR.exists():
        raise FileNotFoundError(f"SenseVoice 모델 폴더가 없습니다: {SENSEVOICE_DIR}")

    model_file = "model_quant.onnx" if SENSEVOICE_QUANTIZE else "model.onnx"
    if not (SENSEVOICE_DIR / model_file).exists():
        raise FileNotFoundError(
            f"SenseVoice ONNX 파일이 없습니다: {SENSEVOICE_DIR / model_file}"
        )

    if not LOCAL_LLM_MODEL_PATH.exists():
        raise FileNotFoundError(f"로컬 LLM 모델 파일이 없습니다: {LOCAL_LLM_MODEL_PATH}")

    if REQUIRE_WAKE_NAME and not BOT_NAME:
        raise RuntimeError("REQUIRE_WAKE_NAME=true인데 BOT_NAME이 비어 있습니다.")

    check_binary("parecord")
    check_binary("paplay")
    check_binary("edge-tts")
    check_binary("ffmpeg")


def run_command(cmd, allow_timeout_124=False, input_text=None, capture=False):
    result = subprocess.run(
        cmd,
        input=input_text,
        text=True if input_text is not None else False,
        encoding="utf-8" if input_text is not None else None,
        errors="replace" if input_text is not None else None,
        capture_output=capture,
    )

    if allow_timeout_124:
        if result.returncode not in (0, 124):
            raise RuntimeError(f"Command failed: {cmd}, returncode={result.returncode}")
    else:
        if result.returncode != 0:
            if capture:
                stderr = result.stderr or ""
                stdout = result.stdout or ""
                raise RuntimeError(
                    f"Command failed: {cmd}, returncode={result.returncode}\n"
                    f"stdout={stdout}\nstderr={stderr}"
                )
            raise RuntimeError(f"Command failed: {cmd}, returncode={result.returncode}")

    return result


def record_audio(seconds: float):
    if RECORD_PATH.exists():
        RECORD_PATH.unlink()

    cmd = [
        "timeout",
        "--signal=INT",
        f"{seconds}s",
        "parecord",
        f"--rate={SAMPLE_RATE}",
        f"--channels={CHANNELS}",
        "--format=s16le",
        "--file-format=wav",
        str(RECORD_PATH),
    ]

    # INPUT_DEVICE가 지정된 경우에만 장치를 강제하고,
    # 비어 있으면 시스템 기본 입력 장치를 사용한다.
    if INPUT_DEVICE:
        cmd.insert(4, f"--device={INPUT_DEVICE}")

    safe_print(f"\n[REC] {seconds:g}초 녹음합니다. 말씀하십시오.")
    run_command(cmd, allow_timeout_124=True)

    if not RECORD_PATH.exists() or RECORD_PATH.stat().st_size == 0:
        raise RuntimeError("녹음 파일이 생성되지 않았습니다.")

    safe_print(f"[REC] 저장 완료: {RECORD_PATH.name} ({RECORD_PATH.stat().st_size} bytes)")


def load_stt():
    safe_print("[STT] 로컬 SenseVoice 모델 로딩 중...")
    safe_print(f"[STT] dir={SENSEVOICE_DIR}")

    t0 = time.time()

    model = SenseVoiceSmall(
        str(SENSEVOICE_DIR),
        batch_size=1,
        quantize=SENSEVOICE_QUANTIZE,
        intra_op_num_threads=N_THREADS,
    )

    safe_print(f"[STT] 로딩 완료: {time.time() - t0:.2f}s")

    # 첫 추론은 onnxruntime 그래프 최적화 때문에 매우 느리다(~1분).
    # 무음 버퍼로 미리 워밍업해서 실제 대화 첫 턴이 느려지지 않게 한다.
    try:
        safe_print("[STT] 워밍업 중... (최초 1회, 시간이 걸릴 수 있음)")
        warmup_wav = BASE_DIR / "_stt_warmup.wav"
        import wave

        with wave.open(str(warmup_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * SAMPLE_RATE)  # 1초 무음

        t0 = time.time()
        model([str(warmup_wav)], language=STT_LANGUAGE, use_itn=True)
        safe_print(f"[STT] 워밍업 완료: {time.time() - t0:.2f}s")

        try:
            warmup_wav.unlink()
        except Exception:
            pass
    except Exception as e:
        safe_print(f"[STT] 워밍업 건너뜀: {e}")

    return model


def _clean_stt_text(text: str) -> str:
    if not text:
        return ""

    if sv_postprocess is not None:
        try:
            text = sv_postprocess(text)
        except Exception:
            pass

    # 혹시 남아 있는 <|...|> 특수 토큰 제거
    while "<|" in text and "|>" in text:
        start = text.index("<|")
        end = text.index("|>", start) + 2
        text = text[:start] + text[end:]

    return text.strip()


def transcribe_audio(stt_model) -> str:
    safe_print("[STT] SenseVoice 전사 중...")

    t0 = time.time()
    result = stt_model([str(RECORD_PATH)], language=STT_LANGUAGE, use_itn=True)
    elapsed = time.time() - t0

    raw = result[0] if result else ""
    if isinstance(raw, dict):
        raw = raw.get("text", "")

    text = _clean_stt_text(raw)

    safe_print(f"[USER] {text}")
    safe_print(f"[STT_TIME] {elapsed:.2f}s")
    return text


def normalize_text(text: str) -> str:
    text = text or ""

    for ch in [" ", ".", ",", "?", "!", "~", "\n", "\t"]:
        text = text.replace(ch, "")

    return text.strip()


def has_wake_name(text: str) -> bool:
    if not BOT_NAME:
        return True

    return normalize_text(BOT_NAME) in normalize_text(text)


def remove_wake_name(text: str) -> str:
    if not BOT_NAME:
        return text.strip()

    result = text.replace(BOT_NAME, "", 1).strip()
    result = result.lstrip("야아어,.:;!? ")
    return result.strip()


def build_prompt(history):
    prompt = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
    )

    for msg in history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            prompt += f"<|im_start|>user\n{content}\n<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}\n<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"

    return prompt


def trim_history(history):
    if len(history) > MAX_HISTORY_MESSAGES:
        return history[-MAX_HISTORY_MESSAGES:]

    return history


def load_local_llm():
    safe_print("[LLM] 로컬 Qwen 모델 로딩 중...")
    safe_print(f"[LLM] model={LOCAL_LLM_MODEL_PATH.name}")
    safe_print(f"[LLM] n_ctx={N_CTX}, n_threads={N_THREADS}, n_batch={N_BATCH}")

    t0 = time.time()

    llm = Llama(
        model_path=str(LOCAL_LLM_MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS_BATCH,
        n_batch=N_BATCH,
        use_mmap=True,
        use_mlock=True,
        verbose=False,
    )

    safe_print(f"[LLM] 로딩 완료: {time.time() - t0:.2f}s")
    return llm


def ask_local_llm(llm: Llama, history: list[dict], user_text: str):
    history.append({"role": "user", "content": user_text})
    history[:] = trim_history(history)

    prompt = build_prompt(history)

    safe_print("[LLM] 로컬 Qwen 답변 생성 중...")

    t0 = time.time()

    output = llm(
        prompt,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stop=[
            "<|im_end|>",
            "<|im_start|>user",
            "<|im_start|>system",
        ],
    )

    elapsed = time.time() - t0

    answer = output["choices"][0]["text"].strip()

    if not answer:
        answer = "죄송합니다. 답변을 생성하지 못했습니다."

    history.append({"role": "assistant", "content": answer})
    history[:] = trim_history(history)

    safe_print(f"[BOT] {answer}")
    safe_print(f"[LLM_TIME] {elapsed:.2f}s")

    return answer


def synthesize_speech_edge_tts(text: str):
    if REPLY_MP3_PATH.exists():
        REPLY_MP3_PATH.unlink()

    if REPLY_WAV_PATH.exists():
        REPLY_WAV_PATH.unlink()

    safe_print("[TTS] edge-tts mp3 생성 중...")

    cmd_tts = [
        "edge-tts",
        "--voice",
        EDGE_TTS_VOICE,
        "--text",
        text,
        "--write-media",
        str(REPLY_MP3_PATH),
    ]

    run_command(cmd_tts, capture=True)

    if not REPLY_MP3_PATH.exists() or REPLY_MP3_PATH.stat().st_size == 0:
        raise RuntimeError("edge-tts mp3 파일이 생성되지 않았습니다.")

    safe_print("[TTS] ffmpeg mp3 → wav 변환 중...")

    cmd_ffmpeg = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(REPLY_MP3_PATH),
        "-ar",
        "48000",
        "-ac",
        "2",
        str(REPLY_WAV_PATH),
    ]

    run_command(cmd_ffmpeg, capture=True)

    if not REPLY_WAV_PATH.exists() or REPLY_WAV_PATH.stat().st_size == 0:
        raise RuntimeError("ffmpeg wav 파일이 생성되지 않았습니다.")

    safe_print(
        f"[TTS] 저장 완료: {REPLY_WAV_PATH.name} "
        f"({REPLY_WAV_PATH.stat().st_size} bytes)"
    )


def play_audio():
    safe_print("[PLAY] 재생 중...")
    run_command(["paplay", str(REPLY_WAV_PATH)])


def print_audio_status():
    safe_print("\n[AUDIO] 현재 기본 입출력")
    subprocess.run(["pactl", "info"])

    safe_print("\n[AUDIO] 입력 장치")
    subprocess.run(["pactl", "list", "sources", "short"])

    safe_print("\n[AUDIO] 출력 장치")
    subprocess.run(["pactl", "list", "sinks", "short"])


def main():
    check_env()

    stt_model = load_stt()
    llm = load_local_llm()
    history = []

    input_label = INPUT_DEVICE if INPUT_DEVICE else "시스템 기본 마이크 (ReSpeaker)"

    safe_print("======================================")
    safe_print(" Raspberry Pi Local Voice Chatbot")
    safe_print(" STT: Local SenseVoice (ONNX)")
    safe_print(" LLM: Local Qwen")
    safe_print(" TTS: edge-tts")
    safe_print(" Convert: ffmpeg mp3 -> wav")
    safe_print(f" 입력: {input_label}")
    safe_print(" 출력: 시스템 기본 스피커 (C-Media USB)")
    safe_print(f" 녹음 시간: {RECORD_SECONDS:g}초")
    safe_print(f" 호출 이름 사용: {REQUIRE_WAKE_NAME}")
    if REQUIRE_WAKE_NAME:
        safe_print(f" 호출 이름: {BOT_NAME}")
    safe_print(" 실행: Enter")
    safe_print(" 종료: q")
    safe_print("======================================")

    print_audio_status()

    while True:
        cmd = input("\nEnter: 녹음 시작 / q: 종료 > ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            safe_print("종료합니다.")
            break

        try:
            record_audio(RECORD_SECONDS)

            transcript = transcribe_audio(stt_model)

            if not transcript:
                safe_print("[WARN] 인식된 문장이 없습니다.")
                continue

            if transcript in ("종료", "그만", "끝", "종료해"):
                safe_print("종료합니다.")
                break

            if REQUIRE_WAKE_NAME and not has_wake_name(transcript):
                safe_print(f'[IGNORE] "{BOT_NAME}" 호출 이름이 없어 답변하지 않습니다.')
                continue

            user_text = remove_wake_name(transcript) if REQUIRE_WAKE_NAME else transcript

            if not user_text:
                user_text = "네, 불렀습니까?"

            answer = ask_local_llm(llm, history, user_text)

            synthesize_speech_edge_tts(answer)

            play_audio()

        except KeyboardInterrupt:
            safe_print("\n중단했습니다.")
            break

        except Exception as e:
            safe_print(f"[ERROR] {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()