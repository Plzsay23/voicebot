#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
average_checkpoints.py — 체크포인트 여러 개의 가중치를 평균내 하나로 합친다.

FunASR 의 avg_nbest_model 이 실패할 때 쓴다. 실패하는 이유:
검증 정확도(acc)가 0 으로만 찍히면 "상위 n개" 정렬이 무의미해져서
가장 오래된 ep1~ep5 를 평균 대상으로 고르는데, 그건 keep_nbest_models 가
이미 지워버린 뒤라 "No checkpoints found for averaging" 으로 끝난다.
(2026-07-26 실제로 겪음)

마지막 몇 epoch 을 평균내면 단일 체크포인트보다 대체로 조금 낫다.
학습 후반의 진동을 깎아주기 때문이다.

사용:
    python average_checkpoints.py outputs/model.pt.ep{16,17,18,19,20}
    python average_checkpoints.py --last 5 --dir outputs
    python average_checkpoints.py --last 5 --dir outputs -o outputs/model.pt.avg5

기본 출력: <dir>/model.pt.avg<개수>  -> export_onnx.py 가 이걸 자동으로 집는다.
"""

import argparse
import sys
from pathlib import Path

import torch


def load_ckpt(path: Path):
    """FunASR 체크포인트를 연다. {'state_dict': ...} 형태와 생 state_dict 둘 다 받는다."""
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj, obj["state_dict"]
    return None, obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", type=Path, nargs="*", help="평균낼 체크포인트들")
    ap.add_argument("--dir", type=Path, default=None,
                    help="--last 와 함께. model.pt.ep* 를 찾을 폴더")
    ap.add_argument("--last", type=int, default=0,
                    help="--dir 안에서 가장 최근 N개 epoch 을 쓴다")
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    ckpts = list(args.ckpts)
    if args.last:
        if not args.dir:
            print("--last 를 쓰려면 --dir 도 줘야 한다.")
            return 1
        eps = sorted(args.dir.glob("model.pt.ep*"),
                     key=lambda p: int(p.name.split("ep")[-1]))
        ckpts = eps[-args.last:]

    ckpts = [p for p in ckpts if p.exists()]
    if len(ckpts) < 2:
        print(f"평균낼 체크포인트가 부족하다({len(ckpts)}개). 최소 2개 필요.")
        return 1

    out = args.out or (ckpts[0].parent / f"model.pt.avg{len(ckpts)}")
    print(f"평균낼 체크포인트 {len(ckpts)}개:")
    for p in ckpts:
        print(f"    {p.name}")

    avg = None
    wrapper = None
    for i, p in enumerate(ckpts):
        w, sd = load_ckpt(p)
        if i == 0:
            wrapper = w
            # 옵티마이저 상태는 평균낼 이유가 없다(추론에 안 쓴다).
            # 여기서 떼어내면 결과 파일이 2.7GB -> 1GB 로 줄어든다.
            avg = {k: (v.clone().float() if torch.is_floating_point(v) else v.clone())
                   for k, v in sd.items()}
            continue
        if set(sd.keys()) != set(avg.keys()):
            print(f"[x] {p.name} 의 키 구성이 다르다. 같은 학습의 산출물이 맞는지 확인할 것.")
            return 1
        for k, v in sd.items():
            if torch.is_floating_point(avg[k]):
                avg[k] += v.float()

    n = len(ckpts)
    for k, v in avg.items():
        if torch.is_floating_point(v):
            avg[k] = (v / n).to(dtype=torch.float32)

    # 원본이 감싼 형태였으면 같은 형태로 저장한다(옵티마이저 상태는 뺀다).
    if wrapper is not None:
        payload = {k: v for k, v in wrapper.items()
                   if k not in ("state_dict", "optimizer", "scheduler", "scaler")}
        payload["state_dict"] = avg
    else:
        payload = avg

    torch.save(payload, str(out))
    print(f"\n저장: {out}  ({out.stat().st_size / 1e6:.0f}MB)")
    print("\n다음: python export_onnx.py --model-dir outputs --test-wav data/yjhan/wav/0000.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
