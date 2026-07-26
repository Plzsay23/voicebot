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
    """녹음된 항목 목록을 그대로 반환."""
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def done_counts(records):
    """문장 index 별로 몇 번 녹음됐는지 센다 (여러 회차 지원)."""
    counts = {}
    for r in records:
        if "index" in r:
            i = int(r["index"])
            counts[i] = counts.get(i, 0) + 1
    return counts


class Recorder:
    """parecord 를 띄워 raw PCM 을 모으는 간단한 녹음기."""

    def __init__(self, device=None, sr=SR):
        self.device = device
        self.sr = sr
        self.proc = None
        self.chunks = []
        self.thread = None
        self.stop_flag = threading.Event()
        self.first_chunk = threading.Event()

    def start(self):
        """실제로 오디오가 흐르기 시작한 뒤에 리턴한다.

        마이크 소스가 SUSPENDED 상태면(특히 DeepFilterMic 같은 필터체인 노드)
        parecord 를 띄우고도 첫 데이터가 나오기까지 1초 이상 걸린다. 그걸
        기다리지 않고 사용자에게 "말하세요"라고 하면 발화 앞부분이 통째로 날아간다.
        """
        cmd = ["parecord", f"--rate={self.sr}", "--channels=1",
               "--format=s16le", "--raw"]
        if self.device:
            cmd.append(f"--device={self.device}")
        self.chunks = []
        self.stop_flag.clear()
        self.first_chunk.clear()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        if not self.first_chunk.wait(timeout=5.0):
            print("  ⚠ 마이크에서 데이터가 안 옵니다. 장치를 확인하세요.")

    def _read_loop(self):
        while not self.stop_flag.is_set():
            data = self.proc.stdout.read(4096)
            if not data:
                break
            self.chunks.append(data)
            self.first_chunk.set()

    def stop(self) -> bytes:
        # terminate 먼저 하고 읽기 스레드가 EOF까지 읽게 둔다. stop_flag 를 먼저
        # 세우면 파이프에 남은 마지막 조각(최대 128ms)을 버리게 된다.
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=2)
        self.stop_flag.set()
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
    ap.add_argument("--passes", type=int, default=1,
                    help="문장 목록을 몇 회차까지 녹음할지 (기본 1). "
                         "예: 3 이면 각 문장을 3번씩 녹음한다.")
    ap.add_argument("--fast", action="store_true",
                    help="재생·확인 없이 바로 다음 문장으로. 대량 녹음용.")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    if not prompts:
        print(f"문장이 없습니다: {args.prompts}")
        return 1

    spk_dir = args.out / args.speaker
    wav_dir = spk_dir / "wav"
    manifest = spk_dir / "manifest.jsonl"
    records = load_manifest(manifest)
    counts = done_counts(records)

    target = len(prompts) * args.passes
    done_n = len(records)
    print(f"화자: {args.speaker}")
    print(f"문장 {len(prompts)}개 x {args.passes}회 = 목표 {target}발화")
    print(f"녹음 완료 {done_n}개 / 남은 것 {max(0, target - done_n)}개")
    print(f"저장 위치: {spk_dir}")
    if args.list:
        for p in range(args.passes):
            n = sum(1 for i in range(len(prompts)) if counts.get(i, 0) > p)
            print(f"  {p + 1}회차: {n}/{len(prompts)}")
        return 0

    # 회차 단위로 돈다. 각 회차에서 아직 그만큼 녹음 안 된 문장만 대상.
    todo = []
    for p in range(args.passes):
        for i, t in enumerate(prompts):
            if counts.get(i, 0) <= p:
                todo.append((i, p, t))
    if not todo:
        print(f"\n목표 {target}발화를 모두 채웠습니다. "
              f"더 늘리려면 --passes 를 키우세요.")
        return 0

    if args.fast:
        print("\n[빠른 모드] 재생·확인 없이 바로 넘어갑니다. "
              "레벨 경고가 뜬 것만 나중에 확인하세요.")
        print("[조작] Enter=녹음 시작 → 읽고 → Enter=종료 후 자동 저장 (q=종료)")
    else:
        print("\n[조작] Enter=녹음 시작/종료   저장 후: Enter=채택 r=다시 s=건너뛰기 q=종료")
    print("[요령] 마이크에서 30cm 정도, 평소 말하는 속도와 톤으로.")
    if args.passes > 1:
        print("[중요] 회차마다 톤·속도·마이크와의 거리를 조금씩 바꿀 것. "
              "똑같이 읽으면 데이터를 늘린 효과가 없다.\n")

    rec = Recorder(device=args.device, sr=args.sr)
    saved = 0
    warned = []
    try:
        for pos, (idx, rnd, text) in enumerate(todo, 1):
            while True:
                print(f"\n[{pos}/{len(todo)}] ({rnd + 1}회차)  \033[1m{text}\033[0m")
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

                # 회차별로 파일명을 달리해 이전 회차를 덮어쓰지 않게 한다.
                name = f"{idx:04d}.wav" if rnd == 0 else f"{idx:04d}_r{rnd}.wav"
                wav_path = wav_dir / name
                write_wav(wav_path, pcm, args.sr)
                warn = ""
                if peak < -30:
                    warn = "  ⚠ 소리가 작음(마이크에 가까이)"
                elif peak > -1.5:
                    warn = "  ⚠ 클리핑 위험(조금 떨어져서)"
                print(f"  저장: {wav_path.name}  {dur:.2f}s  "
                      f"peak {peak:.1f} dBFS{warn}")
                if warn:
                    warned.append(wav_path.name)

                if not args.fast:
                    play(wav_path)
                    ans = input("  [Enter]=채택  r=다시  s=건너뛰기  q=종료 > ") \
                        .strip().lower()
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
                        "round": rnd,
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

    records = load_manifest(manifest)
    total = sum(r.get("duration", 0) for r in records)
    print(f"\n이번 세션 {saved}개 저장. 누적 {len(records)}/{target}발화, "
          f"총 {total / 60:.1f}분.")
    if warned:
        print(f"레벨 경고 {len(warned)}개: {', '.join(warned[:10])}"
              f"{' ...' if len(warned) > 10 else ''}")
        print("  들어보고 문제 있으면 manifest.jsonl 의 해당 줄만 지우고 다시 녹음하면 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
