#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_funasr_data.py — record_dataset.py 가 만든 manifest.jsonl 을
FunASR(SenseVoice) 학습이 요구하는 jsonl 포맷으로 변환하고 train/val 로 나눈다.

FunASR 이 요구하는 한 줄:
  {"key": "...", "source": "/abs/path.wav", "source_len": 140,
   "target": "제리야 오늘 날씨 어때", "target_len": 12,
   "text_language": "<|ko|>", "emo_target": "<|NEUTRAL|>",
   "event_target": "<|Speech|>", "with_or_wo_itn": "<|woitn|>"}

사용:
    python training/prepare_funasr_data.py --data training/data/yjhan
    python training/prepare_funasr_data.py --data training/data/yjhan training/data/zeroth
    python training/prepare_funasr_data.py --data training/data/yjhan \
        --noise-dir training/data/noise --noise-copies 2

옵션:
    --repeat N       내 목소리 데이터를 N배로 복제(공개 데이터와 섞을 때 비중 조절)
    --noise-dir      잡음 wav 폴더. 있으면 잡음 섞은 복사본을 추가로 만든다.
    --noise-copies   발화 하나당 만들 잡음 버전 수
    --val-ratio      검증셋 비율(기본 0.15)
"""

import argparse
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
FRAME_SHIFT_MS = 10          # fbank hop. source_len = 길이(ms) / 10
SNR_CHOICES = (5, 10, 15, 20)  # 잡음 섞을 때 쓸 신호대잡음비(dB)


def read_wav(path: Path):
    """16bit PCM wav 를 mono float32 로 읽는다. 스테레오면 평균내서 합친다."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        if w.getsampwidth() != 2:
            raise ValueError(f"16bit PCM 이 아니다(sampwidth={w.getsampwidth()})")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32) / 32768.0
    if ch > 1:
        audio = audio[:len(audio) // ch * ch].reshape(-1, ch).mean(axis=1)
    return audio, sr


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """선형보간 리샘플. 잡음을 발화 샘플레이트에 맞추는 용도라 이 정도면 충분하다."""
    if sr_from == sr_to or audio.size == 0:
        return audio
    n_out = int(round(len(audio) * sr_to / sr_from))
    return np.interp(np.linspace(0, len(audio) - 1, n_out),
                     np.arange(len(audio)), audio).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((clipped * 32767).astype(np.int16).tobytes())


def mix_noise(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """지정한 SNR 이 되도록 잡음 크기를 맞춰서 더한다."""
    if len(noise) < len(speech):
        reps = int(np.ceil(len(speech) / max(1, len(noise))))
        noise = np.tile(noise, reps)
    start = random.randint(0, max(0, len(noise) - len(speech)))
    noise = noise[start:start + len(speech)]

    p_speech = float(np.mean(speech ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 1e-12 or p_speech <= 1e-12:
        return speech
    scale = np.sqrt(p_speech / (p_noise * (10 ** (snr_db / 10))))
    mixed = speech + noise * scale
    peak = float(np.abs(mixed).max())
    if peak > 1.0:                      # 클리핑 방지
        mixed = mixed / peak * 0.99
    return mixed


def make_entry(key: str, wav: Path, text: str, duration: float) -> dict:
    return {
        "key": key,
        "source": str(wav.resolve()).replace("\\", "/"),
        "source_len": max(1, int(duration * 1000 / FRAME_SHIFT_MS)),
        "target": text,
        "target_len": len(text),
        "text_language": "<|ko|>",
        "emo_target": "<|NEUTRAL|>",
        "event_target": "<|Speech|>",
        "with_or_wo_itn": "<|woitn|>",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="manifest.jsonl 이 있는 폴더들")
    ap.add_argument("--out", type=Path, default=BASE_DIR / "funasr_data")
    ap.add_argument("--repeat", type=int, default=1,
                    help="첫 번째 --data(내 목소리)를 N배 복제")
    ap.add_argument("--noise-dir", type=Path, default=None)
    ap.add_argument("--noise-copies", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- 잡음 로드 -------------------------------------------------------
    noises = []
    if args.noise_dir and args.noise_dir.exists():
        bad = []
        for p in sorted(args.noise_dir.glob("*.wav")):
            # 잡음 파일 하나가 깨졌다고 전체 변환이 죽으면 안 된다.
            # (녹음이 중간에 끊기면 wav 헤더가 안 닫혀 이런 파일이 남는다)
            try:
                audio, sr = read_wav(p)
            except Exception as e:
                bad.append((p, e))
                continue
            if audio.size == 0:
                bad.append((p, "내용이 비어 있음"))
                continue
            noises.append((p.stem, audio, sr))
        print(f"잡음 파일 {len(noises)}개 로드: {args.noise_dir}")
        for p, e in bad:
            print(f"  [!] 건너뜀 {p.name}: {e}  -> 다시 녹음할 것")
    if args.noise_copies and not noises:
        print("경고: --noise-copies 를 줬지만 쓸 수 있는 잡음 파일이 없다. 증강을 건너뛴다.")
        args.noise_copies = 0

    aug_dir = args.out / "augmented"
    entries = []

    for di, data_dir in enumerate(args.data):
        manifest = data_dir / "manifest.jsonl"
        if not manifest.exists():
            print(f"건너뜀(manifest 없음): {manifest}")
            continue
        records = [json.loads(l) for l in
                   manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        repeat = args.repeat if di == 0 else 1
        n_before = len(entries)

        for r in records:
            wav = data_dir / r["audio_filepath"]
            if not wav.exists():
                continue
            spk = r.get("speaker", data_dir.name)
            base_key = f"{spk}_{Path(r['audio_filepath']).stem}"
            for k in range(repeat):
                suffix = "" if repeat == 1 else f"_r{k}"
                entries.append(make_entry(base_key + suffix, wav,
                                          r["text"], r["duration"]))

            # 잡음 증강본 생성 (첫 번째 데이터셋에만 적용)
            if di == 0 and args.noise_copies and noises:
                try:
                    speech, sr = read_wav(wav)
                except Exception as e:
                    print(f"  [!] 증강 건너뜀 {wav.name}: {e}")
                    continue
                for k in range(args.noise_copies):
                    nname, naudio, nsr = random.choice(noises)
                    naudio = resample(naudio, nsr, sr)   # 안 맞으면 맞춰준다
                    if naudio.size == 0:
                        continue
                    snr = random.choice(SNR_CHOICES)
                    mixed = mix_noise(speech, naudio, snr)
                    out_wav = aug_dir / f"{base_key}_{nname}_snr{snr}.wav"
                    write_wav(out_wav, mixed, sr)
                    entries.append(make_entry(
                        f"{base_key}_{nname}_snr{snr}", out_wav,
                        r["text"], len(mixed) / sr))

        print(f"{data_dir.name}: {len(records)}개 -> {len(entries) - n_before}줄"
              f"{f' (x{repeat} 복제)' if repeat > 1 else ''}")

    if not entries:
        print("변환할 데이터가 없습니다.")
        return 1

    random.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_ratio))
    val, train = entries[:n_val], entries[n_val:]

    for name, rows in (("train", train), ("val", val)):
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        total_s = sum(e["source_len"] for e in rows) * FRAME_SHIFT_MS / 1000
        print(f"{path}  {len(rows)}줄  {total_s / 60:.1f}분")

    if len(train) < 50:
        print("\n주의: 학습 데이터가 매우 적다. 경로 검증용으로는 충분하지만,")
        print("      실제 성능 개선을 보려면 300발화 이상 녹음할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
