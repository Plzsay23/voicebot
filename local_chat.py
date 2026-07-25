# -*- coding: utf-8 -*-

import sys
import time
from pathlib import Path
from llama_cpp import Llama


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent

# 현재 쓰는 모델 파일명에 맞게 수정
MODEL_PATH = BASE_DIR / "models" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"


# Pi 4B 4GB용 속도 우선 설정
N_CTX = 256
N_THREADS = 4
N_THREADS_BATCH = 4
N_BATCH = 128

MAX_HISTORY_MESSAGES = 3
MAX_TOKENS = 40
TEMPERATURE = 0.4
TOP_P = 0.9
REPEAT_PENALTY = 1.08


SYSTEM_PROMPT = (
    "너는 라즈베리파이 4B에서 실행되는 작은 한국어 챗봇이다. "
    "답변은 짧고 명확하게 한국어로 한다. "
    "음성비서처럼 1~2문장으로 답한다. "
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


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일이 없습니다:\n{MODEL_PATH}\n\n"
            "MODEL_PATH의 파일명을 실제 .gguf 파일명으로 수정하십시오."
        )

    safe_print("======================================")
    safe_print(" Local LLM Chat Test - Fast Mode")
    safe_print(f" Model: {MODEL_PATH.name}")
    safe_print(f" n_ctx: {N_CTX}")
    safe_print(f" n_threads: {N_THREADS}")
    safe_print(f" n_batch: {N_BATCH}")
    safe_print(f" max_tokens: {MAX_TOKENS}")
    safe_print(" 종료: q / quit / exit")
    safe_print("======================================")
    safe_print("[LOAD] 모델 로딩 중...")

    t0 = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_threads_batch=N_THREADS_BATCH,
        n_batch=N_BATCH,
        use_mmap=True,
        use_mlock=True,
        verbose=False,
    )

    dt = time.time() - t0
    safe_print(f"[LOAD] 완료: {dt:.2f}s")

    return llm


def generate_answer(llm, history):
    prompt = build_prompt(history)

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

    dt = time.time() - t0

    answer = output["choices"][0]["text"].strip()

    if not answer:
        answer = "죄송합니다. 답변을 생성하지 못했습니다."

    return answer, dt


def main():
    llm = load_model()

    history = []

    while True:
        try:
            user_text = input("\nYou > ").strip()

            if user_text.lower() in ("q", "quit", "exit"):
                safe_print("종료합니다.")
                break

            if not user_text:
                continue

            history.append({
                "role": "user",
                "content": user_text,
            })

            history = trim_history(history)

            safe_print("Bot > ", end="")

            answer, elapsed = generate_answer(llm, history)

            safe_print(answer)
            safe_print(f"[TIME] {elapsed:.2f}s")

            history.append({
                "role": "assistant",
                "content": answer,
            })

            history = trim_history(history)

        except KeyboardInterrupt:
            safe_print("\n종료합니다.")
            break

        except Exception as e:
            safe_print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()