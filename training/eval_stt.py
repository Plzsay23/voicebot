#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_stt.py — manifest.jsonl 의 정답과 STT 출력을 비교해 정확도를 잰다.
파인튜닝 전/후를 같은 잣대로 비교하기 위한 기준점 측정용. 파이에서 실행.

사용:
    python3 training/eval_stt.py                              # 기본 화자
    python3 training/eval_stt.py --data training/data/yjhan
    python3 training/eval_stt.py --save baseline.json         # 결과 저장(나중에 비교)
    python3 training/eval_stt.py --compare baseline.json      # 저장본과 비교

지표:
    CER   글자 단위 오류율(낮을수록 좋음). 한국어는 WER보다 CER이 적절.
    완전일치 문장 단위로 통째로 맞은 비율.
    웨이크워드 has_wake_name() 이 통과시킨 비율. 실사용에서 가장 중요한 지표.
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "ros_nodes"))

import voice_common as vc  # noqa: E402


def cer(ref: str, hyp: str) -> float:
    """글자 단위 편집거리 / 정답 길이."""
    ref = vc.normalize_text(ref)
    hyp = vc.normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=BASE_DIR / "training" / "data" / "yjhan")
    ap.add_argument("--save", type=Path, help="결과를 JSON으로 저장")
    ap.add_argument("--compare", type=Path, help="저장된 결과와 비교")
    ap.add_argument("--quiet", action="store_true", help="맞은 것은 출력 안 함")
    args = ap.parse_args()

    manifest = args.data / "manifest.jsonl"
    if not manifest.exists():
        print(f"manifest 가 없습니다: {manifest}")
        return 1

    records = [json.loads(l) for l in
               manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"평가 대상: {len(records)}개 ({args.data})")
    print("SenseVoice 로딩 중...")
    model = vc.load_stt()

    results = []
    t_total = 0.0
    for r in records:
        wav = args.data / r["audio_filepath"]
        if not wav.exists():
            continue
        t0 = time.time()
        hyp = vc.transcribe(model, str(wav))
        dt = time.time() - t0
        t_total += dt
        ref = r["text"]
        e = cer(ref, hyp)
        exact = vc.normalize_text(ref) == vc.normalize_text(hyp)
        # 정답 판정에는 퍼지 매칭을 쓰면 안 된다. has_wake_name(ref) 로 재면
        # "처리가/거리가/제조" 같은 유사음 네거티브가 통째로 웨이크워드 풀로
        # 넘어가서, 인식률은 100% 로 부풀고 오검출은 23% -> 2% 로 축소된다.
        # (2026-07-27 확인: 실제 23/52 를 34/41 로 잘못 나누고 있었다.)
        # 정답은 글자 그대로 들어있는지로만 본다.
        wake_expected = vc.normalize_text(vc.BOT_NAME) in vc.normalize_text(ref)
        wake_got = vc.has_wake_name(hyp)
        results.append({"file": r["audio_filepath"], "ref": ref, "hyp": hyp,
                        "cer": e, "exact": exact,
                        "wake_expected": wake_expected, "wake_got": wake_got,
                        "sec": dt})

    n = len(results)
    if not n:
        print("평가할 파일이 없습니다.")
        return 1

    mean_cer = sum(r["cer"] for r in results) / n
    exact_n = sum(r["exact"] for r in results)
    wake_pool = [r for r in results if r["wake_expected"]]
    wake_ok = sum(r["wake_got"] for r in wake_pool)
    # 웨이크워드가 없어야 하는데 있다고 잘못 판정한 경우(오검출)
    nonwake_pool = [r for r in results if not r["wake_expected"]]
    false_wake = sum(r["wake_got"] for r in nonwake_pool)

    print("\n" + "=" * 68)
    for r in results:
        if args.quiet and r["exact"]:
            continue
        mark = "○" if r["exact"] else "✗"
        print(f"{mark} {r['file']}  CER {r['cer']:.2f}")
        print(f"    정답: {r['ref']}")
        print(f"    인식: {r['hyp']}")
    print("=" * 68)
    print(f"발화 수        {n}")
    print(f"CER            {mean_cer:.3f}   (0에 가까울수록 좋음)")
    print(f"완전일치       {exact_n}/{n}  ({exact_n / n * 100:.1f}%)")
    if wake_pool:
        print(f"웨이크워드 인식 {wake_ok}/{len(wake_pool)}  "
              f"({wake_ok / len(wake_pool) * 100:.1f}%)")
    if nonwake_pool:
        print(f"웨이크워드 오검출 {false_wake}/{len(nonwake_pool)}  "
              f"(낮을수록 좋음)")
    print(f"평균 추론시간   {t_total / n:.2f}s")

    summary = {"n": n, "cer": mean_cer, "exact": exact_n / n,
               "wake_recall": wake_ok / len(wake_pool) if wake_pool else None,
               "false_wake": false_wake / len(nonwake_pool) if nonwake_pool else None,
               "sec": t_total / n, "results": results}

    if args.compare and args.compare.exists():
        old = json.loads(args.compare.read_text(encoding="utf-8"))
        print("\n--- 비교 (" + str(args.compare) + " 대비) ---")

        def delta(name, key, lower_is_better=True):
            a, b = old.get(key), summary.get(key)
            if a is None or b is None:
                return
            d = b - a
            good = (d < 0) if lower_is_better else (d > 0)
            arrow = "개선" if good and abs(d) > 1e-9 else (
                "동일" if abs(d) < 1e-9 else "악화")
            print(f"  {name:14s} {a:.3f} -> {b:.3f}  ({d:+.3f}, {arrow})")

        delta("CER", "cer", True)
        delta("완전일치", "exact", False)
        delta("웨이크워드", "wake_recall", False)
        delta("추론시간", "sec", True)

    if args.save:
        args.save.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"\n저장: {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
