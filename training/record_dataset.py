#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
record_dataset.py — STT 파인튜닝용 음성 데이터셋 녹음 도구 (라즈베리파이에서 실행)

prompts_ko.txt 의 문장을 하나씩 보여주고, 사용자가 읽은 음성을 실제 서비스와
똑같은 마이크/경로(parecord, 16k mono s16)로 녹음해 manifest.jsonl 로 저장한다.
녹음 조건이 추론 조건과 같아야 파인튜닝 효과가 나오므로 반드시 파이에서 실행할 것.

사용:
    python3 training/record_dataset.py                 # 이어서 녹음(자동 resume)
    python3 training/record_dataset.py --speaker yjhan
    python3 training/record_dataset.py --list          # 진행 상황만 보기

조작:
    Enter  녹음 시작 -> 문장 읽기 -> Enter  녹음 종료
    저장 직후: [Enter]=채택  r=다시  s=건너뛰기  q=종료(진행상황 저장됨)

출력:
    training/data/<speaker>/wav/0001.wav      16kHz mono 16bit
    training/data/<speaker>/manifest.jsonl    {"audio_filepath","text","duration"}
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPTS = BASE_DIR / "prompts_ko.txt"
DEFAULT_OUT = BASE_DIR / "data"

SR = 16000
SAMPLE_WIDTH = 2

# 무음 트리밍: 이 값보다 큰 구간만 발화로 보고 앞뒤로 PAD_MS 만큼 여유를 남긴다.
TRIM_THRESHOLD = 0.02      # 진폭(0~1) 기준
TRIM_WIN_MS = 20
PAD_MS = 200
MIN_DURATION = 0.3         # 이보다 짧으면 실패로 보고 다시 녹음 권유
MAX_DURATION = 20.0


def load_prompts(path: Path):
    prompts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def load_manifest(path: Path):
    """이미 녹음한 항목을 {index: record} 로 반환 (resume용)."""
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "index" in rec:
            done[int(rec["index"])] = rec
    return done


class Recorder:
    """parecord 를 띄워 raw PCM 을 모으는 간단한 녹음기."""

    def __init__(self, device=None, sr=SR):
        self.device = device
        self.sr = sr
        self.proc = None
        self.chunks = []
        self.thread = None
        self.stop_flag = threading.Event()

    def start(self):
        cmd = ["parecord", f"--rate={self.sr}", "--channels=1",
               "--format=s16le", "--raw"]
        if self.device:
            cmd.append(f"--device={self.device}")
        self.chunks = []
        self.stop_flag.clear()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while not self.stop_flag.is_set():
            data = self.proc.stdout.read(4096)
            if not data:
                break
            self.chunks.append(data)

    def stop(self) -> bytes:
        self.stop_flag.set()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)
        return b"".join(self.chunks)


def trim_silence(pcm: bytes, sr=SR) -> bytes:
    """앞뒤 무음을 잘라낸다. 발화를 못 찾으면 원본을 그대로 돌려준다."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    win = max(1, int(sr * TRIM_WIN_MS / 1000))
    n_win = len(audio) // win
    if n_win == 0:
        return pcm
    frames = audio[: n_win * win].reshape(n_win, win)
    loud = np.abs(frames).max(axis=1) > TRIM_THRESHOLD
    if not loud.any():
        return pcm
    pad = max(1, int(PAD_MS / TRIM_WIN_MS))
    first = max(0, int(np.argmax(loud)) - pad)
    last = min(n_win, n_win - int(np.argmax(loud[::-1])) + pad)
    return pcm[first * win * SAMPLE_WIDTH: last * win * SAMPLE_WIDTH]


def write_wav(path: Path, pcm: bytes, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(sr)
        w.writeframes(pcm)


def play(path: Path):
    try:
        subprocess.run(["paplay", str(path)], check=False)
    except FileNotFoundError:
        print("  (paplay 없음 — 재생 건너뜀)")


def peak_dbfs(pcm: bytes) -> float:
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    return 20 * np.log10(peak) if peak > 0 else -99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--speaker", default=os.getenv("USER", "speaker"))
    ap.add_argument("--device", default=None, help="parecord --device 이름")
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--list", action="store_true", help="진행 상황만 출력")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"문장이 없습니다: {args.prompts}")
        return 1

    spk_dir = args.out / args.speaker
    wav_dir = spk_dir / "wav"
    manifest = spk_dir / "manifest.jsonl"
    done = load_manifest(manifest)

    print(f"화자: {args.speaker}")
    print(f"문장: {len(prompts)}개 / 녹음 완료: {len(done)}개 "
          f"/ 남은 것: {len(prompts) - len(done)}개")
    print(f"저장 위치: {spk_dir}")
    if args.list:
        return 0

    todo = [(i, t) for i, t in enumerate(prompts) if i not in done]
    if not todo:
        print("\n모두 녹음되었습니다. 다시 녹음하려면 manifest.jsonl 의 해당 줄을 지우세요.")
        return 0

    print("\n[조작] Enter=녹음 시작/종료   저장 후: Enter=채택 r=다시 s=건너뛰기 q=종료")
    print("[요령] 마이크에서 30cm 정도, 평소 말하는 속도와 톤으로. 조용한 환경에서.\n")

    rec = Recorder(device=args.device, sr=args.sr)
    saved = 0
    try:
        for idx, text in todo:
            while True:
                print(f"\n[{idx + 1}/{len(prompts)}]  \033[1m{text}\033[0m")
                cmd = input("  Enter=녹음 시작 (s=건너뛰기, q=종료) > ").strip().lower()
                if cmd == "q":
                    raise KeyboardInterrupt
                if cmd == "s":
                    break

                rec.start()
                input("  ● 녹음 중... 문장을 읽고 Enter > ")
                pcm = rec.stop()

                dur = len(pcm) / SAMPLE_WIDTH / args.sr
                pcm = trim_silence(pcm, args.sr)
                dur = len(pcm) / SAMPLE_WIDTH / args.sr
                peak = peak_dbfs(pcm)

                if dur < MIN_DURATION:
                    print(f"  ✗ 너무 짧습니다({dur:.2f}s). 다시 녹음하세요.")
                    continue
                if dur > MAX_DURATION:
                    print(f"  ✗ 너무 깁니다({dur:.1f}s). 다시 녹음하세요.")
                    continue

                wav_path = wav_dir / f"{idx:04d}.wav"
                write_wav(wav_path, pcm, args.sr)
                warn = ""
                if peak < -30:
                    warn = "  ⚠ 소리가 작습니다(마이크에 가까이)"
                elif peak > -1.5:
                    warn = "  ⚠ 클리핑 위험(조금 떨어져서)"
                print(f"  저장: {wav_path.name}  {dur:.2f}s  peak {peak:.1f} dBFS{warn}")
                play(wav_path)

                ans = input("  [Enter]=채택  r=다시  s=건너뛰기  q=종료 > ").strip().lower()
                if ans == "r":
                    continue
                if ans == "s":
                    wav_path.unlink(missing_ok=True)
                    break
                if ans == "q":
                    raise KeyboardInterrupt

                with manifest.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "index": idx,
                        "audio_filepath": str(wav_path.relative_to(spk_dir)),
                        "text": text,
                        "duration": round(dur, 3),
                        "speaker": args.speaker,
                        "sample_rate": args.sr,
                    }, ensure_ascii=False) + "\n")
                saved += 1
                break
    except KeyboardInterrupt:
        print("\n중단합니다.")
    finally:
        rec.stop()

    done = load_manifest(manifest)
    total = sum(r.get("duration", 0) for r in done.values())
    print(f"\n이번 세션 {saved}개 저장. 누적 {len(done)}/{len(prompts)}개, "
          f"총 {total / 60:.1f}분.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
