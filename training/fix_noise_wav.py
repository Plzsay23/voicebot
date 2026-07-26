#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_noise_wav.py — 헤더가 깨진 wav 를 진단하고 복구한다.

녹음 프로세스가 SIGTERM/SIGKILL 로 끊기면 RIFF/data 청크의 크기 필드가
0 이나 쓰레기값인 채로 남는다. 오디오 샘플 자체는 파일에 다 들어있는데
wave.open() 은 "fmt chunk and/or data chunk missing" 으로 거부한다.
이 스크립트는 청크를 느슨하게 훑어서 fmt 정보를 찾고, data 를 파일 끝까지로
간주해 크기 필드만 다시 써준다. 원본은 .bak 으로 남긴다.

사용:
    python fix_noise_wav.py data/noise/*.wav      # 진단 + 복구
    python fix_noise_wav.py --dry-run data/noise/*.wav   # 진단만

헤더가 아예 없는 순수 raw PCM 이면 포맷을 알 수 없으므로 지정해줘야 한다:
    python fix_noise_wav.py --assume-raw --rate 16000 --channels 1 data/noise/x.wav
"""

import argparse
import shutil
import struct
import sys
import wave
from pathlib import Path


def scan_chunks(buf: bytes):
    """RIFF 청크를 느슨하게 훑는다. 크기 필드가 망가져 있어도 최대한 진행한다."""
    fmt = None          # (channels, sample_rate, bits)
    data_off = None
    data_len = None
    pos = 12
    while pos + 8 <= len(buf):
        cid = buf[pos:pos + 4]
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        body = pos + 8

        if cid == b"fmt " and body + 16 <= len(buf):
            _tag, ch, sr, _brate, _align, bits = struct.unpack_from("<HHIIHH", buf, body)
            fmt = (ch, sr, bits)
        elif cid == b"data":
            data_off = body
            # 크기 필드를 못 믿는다. 0 이거나 파일 밖을 가리키면 끝까지로 본다.
            data_len = size if 0 < size <= len(buf) - body else len(buf) - body
            break       # data 는 보통 마지막이다

        if size == 0 or body + size > len(buf):
            break       # 크기가 망가졌다. 더 못 걷는다.
        pos = body + size + (size & 1)      # 청크는 짝수 정렬

    return fmt, data_off, data_len


def repair(path: Path, args) -> bool:
    buf = path.read_bytes()
    if len(buf) < 44:
        print(f"{path.name}: 파일이 너무 작다({len(buf)} bytes). 복구 불가.")
        return False

    if args.assume_raw:
        fmt = (args.channels, args.rate, 16)
        data_off, data_len = 0, len(buf)
    else:
        if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
            print(f"{path.name}: RIFF/WAVE 헤더가 없다. 앞 16바이트: {buf[:16]!r}")
            print("  -> 순수 raw PCM 이면 --assume-raw --rate 16000 --channels 1 로 다시 실행.")
            return False
        fmt, data_off, data_len = scan_chunks(buf)

    if fmt is None:
        print(f"{path.name}: fmt 청크를 못 찾았다. 복구 불가.")
        return False
    if data_off is None:
        print(f"{path.name}: data 청크를 못 찾았다. 복구 불가.")
        return False

    ch, sr, bits = fmt
    if bits != 16:
        print(f"{path.name}: 16bit PCM 이 아니다({bits}bit). 이 스크립트는 처리 못 한다.")
        return False

    frame = ch * bits // 8
    data_len -= data_len % frame            # 프레임 경계로 자른다
    dur = data_len / (sr * frame)
    print(f"{path.name}: {ch}ch {sr}Hz {bits}bit, 데이터 {data_len} bytes = {dur:.1f}초")

    if args.dry_run:
        return True

    pcm = buf[data_off:data_off + data_len]
    shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(bits // 8)
        w.setframerate(sr)
        w.writeframes(pcm)

    with wave.open(str(path), "rb") as w:   # 실제로 열리는지 확인
        n = w.getnframes()
    print(f"  -> 복구 완료. {n / sr:.1f}초. 원본은 {path.name}.bak")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--dry-run", action="store_true", help="진단만 하고 고치지 않는다")
    ap.add_argument("--assume-raw", action="store_true",
                    help="헤더 없는 raw PCM 으로 간주")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    args = ap.parse_args()

    ok = 0
    for p in args.files:
        if not p.exists():
            print(f"{p}: 없음")
            continue
        try:                                # 멀쩡한 건 건드리지 않는다
            with wave.open(str(p), "rb") as w:
                print(f"{p.name}: 정상 ({w.getnframes() / w.getframerate():.1f}초). 건너뜀")
                ok += 1
                continue
        except Exception:
            pass
        if repair(p, args):
            ok += 1

    print(f"\n{ok}/{len(args.files)} 정상.")
    return 0 if ok == len(args.files) else 1


if __name__ == "__main__":
    sys.exit(main())
