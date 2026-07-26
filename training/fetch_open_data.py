#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_open_data.py — 공개 한국어 음성 데이터셋을 내려받아 record_dataset.py 와
같은 포맷(wav 16k mono + manifest.jsonl)으로 변환한다. 학습 PC에서 실행.

내 목소리 데이터만으로 파인튜닝하면 모델이 원래 알던 한국어를 잊어버린다
(catastrophic forgetting). 공개 데이터를 함께 섞어서 학습하기 위한 도구다.

준비:
    pip install "datasets>=2.19" soundfile librosa

사용:
    python training/fetch_open_data.py --dataset zeroth --hours 3
    python training/fetch_open_data.py --dataset fleurs --hours 2
    python training/fetch_open_data.py --list

출력:
    training/data/<이름>/wav/*.wav
    training/data/<이름>/manifest.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BASE_DIR / "data"
TARGET_SR = 16000

# 회원가입/승인 없이 바로 받을 수 있는 것만 넣었다.
SOURCES = {
    "zeroth": {
        "hf_id": "Bingsu/zeroth-korean",
        "config": None,
        "split": "train",
        "text_key": "text",
        "license": "CC BY 4.0",
        "note": "한국어 낭독체 51시간. 가장 무난한 기본 코퍼스.",
    },
    "fleurs": {
        "hf_id": "google/fleurs",
        "config": "ko_kr",
        "split": "train",
        "text_key": "transcription",
        "license": "CC BY 4.0",
        "note": "다국어 낭독체 중 한국어 ~10시간. 문장이 길고 격식체.",
    },
    "commonvoice": {
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "config": "ko",
        "split": "train",
        "text_key": "sentence",
        "license": "CC0",
        "note": "크라우드소싱 음성. 화자·마이크가 다양해 잡음 강건성에 도움. "
                "HF 로그인(huggingface-cli login)과 약관 동의 필요.",
    },
}


def print_sources():
    print("사용 가능한 공개 데이터셋:\n")
    for key, s in SOURCES.items():
        print(f"  {key:12s} {s['hf_id']}")
        print(f"  {'':12s} 라이선스 {s['license']} — {s['note']}\n")
    print("추가로 승인 절차가 필요한 대형 코퍼스:")
    print("  KsponSpeech (AI Hub, 한국어 자유대화 1000시간) — aihub.or.kr 회원가입 후 신청.")
    print("  가장 품질이 좋지만 다운로드 승인에 시간이 걸리고 용량이 크다(수백 GB).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SOURCES), help="받을 데이터셋")
    ap.add_argument("--hours", type=float, default=3.0, help="받을 분량(시간)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not args.dataset:
        print_sources()
        return 0

    try:
        import soundfile as sf
        from datasets import load_dataset, Audio
    except ImportError as e:
        print(f"의존성이 없습니다: {e}")
        print('설치: pip install "datasets>=2.19" soundfile librosa')
        return 1

    src = SOURCES[args.dataset]
    out_dir = args.out / args.dataset
    wav_dir = out_dir / "wav"
    manifest = out_dir / "manifest.jsonl"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print(f"{src['hf_id']} 스트리밍 다운로드 중... (목표 {args.hours}시간)")
    ds = load_dataset(src["hf_id"], src["config"], split=src["split"],
                      streaming=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))

    budget = args.hours * 3600
    total = 0.0
    n = 0
    with manifest.open("w", encoding="utf-8") as f:
        for item in ds:
            audio = item["audio"]
            text = (item.get(src["text_key"]) or "").strip()
            if not text:
                continue
            dur = len(audio["array"]) / TARGET_SR
            if dur < 0.5 or dur > 20.0:
                continue

            path = wav_dir / f"{n:06d}.wav"
            sf.write(path, audio["array"], TARGET_SR, subtype="PCM_16")
            f.write(json.dumps({
                "index": n,
                "audio_filepath": str(path.relative_to(out_dir)),
                "text": text,
                "duration": round(dur, 3),
                "speaker": args.dataset,
                "sample_rate": TARGET_SR,
            }, ensure_ascii=False) + "\n")

            n += 1
            total += dur
            if n % 100 == 0:
                print(f"  {n}개 / {total / 3600:.2f}시간")
            if total >= budget:
                break

    print(f"\n완료: {n}개, {total / 3600:.2f}시간 -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
