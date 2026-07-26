#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_prompts.py — 녹음 대본(프롬프트) 자동 생성. GPU 서버/PC 에서 실행.

손으로 "다양하게" 문장을 쓰면 반드시 발음이 편중된다. 대신 공개 한국어
코퍼스(수만 문장)에서 **음소 커버리지를 최대화하는 N문장을 탐욕적으로 고른다**.
TTS/ASR 녹음 대본을 만들 때 쓰는 표준 기법(greedy set cover)이다.

커버리지 단위는 세 가지. 한글을 자모로 쪼개서 센다:
    CV  초성+중성      (가, 까, 냐 …)      — 음절 시작
    VC  중성+종성      (악, 앙, 앎 …)      — 받침
    BD  앞음절종성+다음음절초성            — 어절 내부 경계. 한국어 음운변동
                                            (비음화·유음화·경음화)이 여기서 일어난다

도메인 문장(prompts_domain.txt)은 무조건 전부 넣고, 그것들이 이미 덮은
커버리지를 뺀 **나머지 구멍만** 공개 코퍼스 문장으로 메운다. 그래서 같은
분량이라도 손으로 고른 대본보다 음소 분포가 촘촘하다.

사용:
    # 오프라인(도메인 문장만으로 대본 만들기) — 네트워크 없이도 동작
    python training/build_prompts.py --count 0

    # 공개 코퍼스 섞어서 500문장 대본 (권장)
    pip install "datasets>=2.19,<4"
    python training/build_prompts.py --count 500 --source zeroth

    # 커버리지 리포트만 보기 (파일 안 씀)
    python training/build_prompts.py --count 500 --dry-run

출력:
    training/prompts_ko.txt     학습용 대본  (record_dataset.py 기본 입력)
    training/prompts_eval.txt   평가용 홀드아웃 — 학습 대본과 문장이 겹치지 않는다
    training/prompts_meta.json  어떤 문장이 어느 그룹/출처인지 + 커버리지 수치

주의: 대본을 다시 생성하면 문장 순서가 바뀐다. 이미 녹음을 시작했다면
record_dataset.py 는 텍스트 기준으로 이어녹음하므로 라벨은 안 깨지지만,
섞어 쓰지 말고 **새 대본으로는 새 speaker 디렉터리에** 녹음하는 편이 깔끔하다.
"""

import argparse
import heapq
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOMAIN_FILE = BASE_DIR / "prompts_domain.txt"
OUT_TRAIN = BASE_DIR / "prompts_ko.txt"
OUT_EVAL = BASE_DIR / "prompts_eval.txt"
OUT_META = BASE_DIR / "prompts_meta.json"

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ",
        "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",
        "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]

HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3

# 유닛 종류별 가중치. BD 는 종류가 훨씬 많아서(28x19) 그대로 두면 대본이
# 경계 채우기에만 쏠린다. 조금 낮춘다.
UNIT_WEIGHT = {"CV": 1.0, "VC": 1.0, "BD": 0.6}

# 녹음 가능한 문장 길이(음절 수). 너무 짧으면 정보가 없고, 너무 길면 읽다 틀린다.
# 읽다 틀린 발화는 라벨이 어긋난 학습 데이터가 되므로 상한을 넉넉히 잡지 않는다.
# (도메인 뱅크 문장은 이 필터를 거치지 않는다 — '응', '왜' 같은 짧은 발화가 필요하다)
MIN_SYLL = 4
MAX_SYLL = 35
# 캐시에는 길이 조건을 느슨하게 걸어 담는다. 그래야 --max-syll 을 바꿔도
# 캐시를 다시 받지 않고 다시 거를 수 있다(HF 스트리밍이 한 번에 5~15분 걸린다).
CACHE_MIN_SYLL = 2
CACHE_MAX_SYLL = 80

SOURCES = {
    "zeroth": {
        "hf_id": "Bingsu/zeroth-korean", "config": None, "split": "train",
        "text_key": "text", "license": "CC BY 4.0",
        "note": "한국어 낭독체 약 2만 문장. 기본 추천",
    },
    # 아래 둘은 스트리밍이어도 오디오 아카이브(tar)를 통째로 받는 구조라
    # 텍스트만 뽑는 데 수십 분이 걸린다. zeroth(parquet)로 충분하면 쓰지 말 것.
    "fleurs": {
        "hf_id": "google/fleurs", "config": "ko_kr", "split": "train",
        "text_key": "transcription", "license": "CC BY 4.0",
        "note": "위키 기반 격식체. 길고 고유명사가 많다. 받는 데 아주 느리다",
    },
    "commonvoice": {
        "hf_id": "mozilla-foundation/common_voice_17_0", "config": "ko",
        "split": "train", "text_key": "sentence", "license": "CC0",
        "note": "구어에 가깝고 짧다. HF 로그인 + 약관 동의 필요. 느리다",
    },
}


# ── 한글 분해 & 커버리지 유닛 ──────────────────────────────────────────

def decompose(ch):
    code = ord(ch) - HANGUL_START
    return CHO[code // 588], JUNG[(code % 588) // 28], JONG[code % 28]


def is_hangul(ch):
    return HANGUL_START <= ord(ch) <= HANGUL_END


def units(text):
    """문장에서 커버리지 유닛을 뽑는다. 어절 경계에서는 BD 를 만들지 않는다."""
    out = []
    for word in text.split():
        prev_jong = None          # None = 직전 음절 없음(어절 시작)
        for ch in word:
            if not is_hangul(ch):
                prev_jong = None
                continue
            cho, jung, jong = decompose(ch)
            out.append(("CV", cho, jung))
            if jong:
                out.append(("VC", jung, jong))
            if prev_jong is not None:
                out.append(("BD", prev_jong or "-", cho))
            prev_jong = jong
    return out


def n_syllables(text):
    return sum(1 for ch in text if is_hangul(ch))


# ── 문장 정규화·필터 ───────────────────────────────────────────────────

_PUNCT = re.compile(r"[^가-힣\s]")
_HAS_FOREIGN = re.compile(r"[0-9A-Za-z一-鿿]")
_WS = re.compile(r"\s+")


def normalize(raw, min_syll=MIN_SYLL, max_syll=MAX_SYLL):
    """읽을 수 있는 순수 한글 문장만 통과시킨다. 아니면 None."""
    if not raw:
        return None
    s = unicodedata.normalize("NFC", str(raw)).strip()
    # 숫자·영문·한자가 든 문장은 '읽는 법'이 정해지지 않는다. 라벨이 흔들리므로 버린다.
    if _HAS_FOREIGN.search(s):
        return None
    s = _WS.sub(" ", _PUNCT.sub(" ", s)).strip()
    if not s:
        return None
    n = n_syllables(s)
    if n < min_syll or n > max_syll:
        return None
    return s


# ── 입력 ───────────────────────────────────────────────────────────────

def load_domain(path):
    """'## 그룹' 마커로 나뉜 도메인 문장 뱅크를 읽는다. -> [(text, group)]"""
    items, seen, group = [], set(), "general"
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("##"):
            group = line[2:].strip() or "general"
            continue
        if not line or line.startswith("#"):
            continue
        if line in seen:
            print(f"  [경고] 도메인 뱅크에 중복 문장: {line}", file=sys.stderr)
            continue
        seen.add(line)
        items.append((line, group))
    return items


def iter_source(name, limit):
    """공개 코퍼스에서 텍스트만 스트리밍으로 읽는다(오디오는 디코딩하지 않는다)."""
    from datasets import load_dataset

    s = SOURCES[name]
    ds = load_dataset(s["hf_id"], s["config"], split=s["split"], streaming=True)
    key = s["text_key"]
    # 오디오 컬럼을 남겨두면 행을 꺼낼 때마다 wav 를 디코딩한다(librosa/soundfile
    # 이 없으면 아예 에러). 텍스트 컬럼만 남겨서 디코딩 자체를 막는다.
    # streaming 에서는 column_names 가 None 일 수 있어 select_columns 를 먼저 쓴다.
    try:
        ds = ds.select_columns([key])
    except Exception:
        try:
            drop = [c for c in (ds.column_names or []) if c != key]
            if drop:
                ds = ds.remove_columns(drop)
        except Exception:
            pass
    for i, row in enumerate(ds):
        if i >= limit:
            break
        yield row[key]


# ── 탐욕 선택 (lazy greedy / CELF) ─────────────────────────────────────

def gain(us, counts, min_count):
    """이 문장을 추가했을 때 새로 메워지는 커버리지의 양."""
    g = 0.0
    local = Counter()
    for u in us:
        cur = counts.get(u, 0) + local[u]
        if cur < min_count:
            # 한 번도 안 나온 유닛에 가산점 — 0->1 이 1->2 보다 훨씬 값지다
            g += UNIT_WEIGHT[u[0]] * (1.6 if cur == 0 else 1.0)
        local[u] += 1
    return g


def score(us, cost, counts, min_count):
    """**녹음 1초당** 새로 메우는 커버리지.

    문장당 커버리지로 고르면 긴 뉴스 문장이 이긴다(유닛이 많으니까). 하지만
    비용은 사용자가 읽는 시간이고, 긴 문장은 읽다 틀려 라벨까지 망가뜨린다.
    실제로 아끼려는 자원(녹음 시간)으로 나누는 게 맞다.
    """
    return gain(us, counts, min_count) / cost


def greedy_select(cands, n_pick, counts, min_count):
    """cands: [(text, units, cost)] -> 고른 text 리스트. counts 는 제자리 갱신."""
    heap = []
    for i, (_, us, cost) in enumerate(cands):
        heapq.heappush(heap, (-score(us, cost, counts, min_count), i, 0))

    picked, used = [], set()
    while heap and len(picked) < n_pick:
        neg, i, stamp = heapq.heappop(heap)
        if i in used:
            continue
        if stamp == len(picked):
            # 이 후보의 점수는 현재 상태에서 계산된 값 → 그대로 채택
            text, us, _ = cands[i]
            if -neg <= 0:
                break               # 더 메울 게 없다
            picked.append(text)
            used.add(i)
            for u in us:
                counts[u] = counts.get(u, 0) + 1
        else:
            heapq.heappush(heap, (-score(cands[i][1], cands[i][2],
                                         counts, min_count), i, len(picked)))
    return picked


# ── 리포트 ─────────────────────────────────────────────────────────────

def coverage_report(counts, min_count):
    kinds = {}
    for kind in ("CV", "VC", "BD"):
        seen = {u: c for u, c in counts.items() if u[0] == kind}
        kinds[kind] = {
            "종류": len(seen),
            f"{min_count}회 이상": sum(1 for c in seen.values() if c >= min_count),
            "1회뿐": sum(1 for c in seen.values() if c == 1),
        }
    return kinds


PER_UTT_OVERHEAD = 6.0     # Enter 치고 숨 고르는 시간
SEC_PER_SYLL = 0.17


def utt_seconds(text):
    """이 문장 하나를 녹음하는 데 드는 대략의 시간(초)."""
    return n_syllables(text) * SEC_PER_SYLL + PER_UTT_OVERHEAD


def estimate_minutes(texts, passes=1):
    return sum(utt_seconds(t) for t in texts) * passes / 60.0


# ── 출력 ───────────────────────────────────────────────────────────────

HEADER = """# 자동 생성된 녹음 대본 — build_prompts.py 가 만든다. 직접 고치지 마라.
# 문장을 더하거나 빼려면 prompts_domain.txt 를 고치고 다시 생성할 것.
#
# 생성 조건: {cmd}
# 학습용 {n_train}문장 / 평가용(홀드아웃) {n_eval}문장 — 서로 겹치지 않는다.
#
# 녹음:  {record_cmd}
"""


def write_prompts(path, texts, header):
    path.write_text(header + "\n" + "\n".join(texts) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=500,
                    help="전체 목표 문장 수(도메인 포함). 0 이면 도메인 문장만 쓴다")
    ap.add_argument("--source", action="append", default=None,
                    choices=list(SOURCES), help="공개 코퍼스(여러 번 지정 가능)")
    ap.add_argument("--text-file", action="append", type=Path, default=None,
                    help="한 줄 한 문장인 로컬 텍스트 파일도 후보로 쓴다(여러 번 가능)")
    ap.add_argument("--scan", type=int, default=30000,
                    help="코퍼스에서 훑어볼 원문 수")
    ap.add_argument("--min-syll", type=int, default=MIN_SYLL,
                    help="코퍼스 문장 최소 음절 수")
    ap.add_argument("--max-syll", type=int, default=MAX_SYLL,
                    help="코퍼스 문장 최대 음절 수. 길면 읽다 틀려 라벨이 깨진다")
    ap.add_argument("--min-count", type=int, default=3,
                    help="각 커버리지 유닛이 최소 몇 번 나오게 할지")
    ap.add_argument("--eval-ratio", type=float, default=0.15,
                    help="평가용으로 떼어낼 비율")
    ap.add_argument("--domain", type=Path, default=DOMAIN_FILE)
    ap.add_argument("--cache", type=Path, default=BASE_DIR / "corpus_cache.tsv",
                    help="코퍼스에서 거른 문장 캐시. 두 번째 실행부터는 즉시 끝난다")
    ap.add_argument("--refresh", action="store_true",
                    help="캐시를 무시하고 코퍼스를 다시 받는다")
    ap.add_argument("--speaker", default="yjhan2", help="안내 문구에 쓸 화자 이름")
    ap.add_argument("--passes", type=int, default=2, help="안내 문구에 쓸 회차")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않는다")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) 도메인 문장 — 전부 채택
    domain = load_domain(args.domain)
    print(f"도메인 뱅크: {len(domain)}문장 "
          f"({', '.join(f'{g} {c}' for g, c in Counter(g for _, g in domain).items())})")

    counts = {}
    for text, _ in domain:
        for u in units(text):
            counts[u] = counts.get(u, 0) + 1

    chosen = [(t, g, "domain") for t, g in domain]
    need = max(0, args.count - len(chosen))

    # 2) 공개 코퍼스로 남은 구멍 메우기
    if need and (args.source or args.text_file):
        # raw_pool 은 길이 조건을 느슨하게만 건 상태. 이걸 캐시하고,
        # 실제 --min-syll/--max-syll 은 그 다음에 건다. 그래야 길이를 바꿔도
        # HF 에서 다시 받지 않아도 된다(한 번 받는 데 5~15분).
        raw_pool, seen = [], set()
        cache = args.cache
        if cache and cache.exists() and not args.refresh:
            for line in cache.read_text(encoding="utf-8").splitlines():
                t, _, origin = line.partition("\t")
                if t and t not in seen:
                    seen.add(t)
                    raw_pool.append((t, origin or "cache"))
            print(f"  캐시({cache.name})에서 {len(raw_pool)}문장 — "
                  f"코퍼스를 다시 받으려면 --refresh")
        for path in (args.text_file or []):
            n0 = len(raw_pool)
            for line in path.read_text(encoding="utf-8").splitlines():
                s = normalize(line, CACHE_MIN_SYLL, CACHE_MAX_SYLL)
                if s and s not in seen:
                    seen.add(s)
                    raw_pool.append((s, path.name))
            print(f"  {path.name}: {len(raw_pool) - n0}문장")
        for name in ([] if raw_pool and not args.refresh else (args.source or [])):
            src = SOURCES[name]
            n0 = len(raw_pool)
            try:
                for raw in iter_source(name, args.scan):
                    s = normalize(raw, CACHE_MIN_SYLL, CACHE_MAX_SYLL)
                    if s and s not in seen:
                        seen.add(s)
                        raw_pool.append((s, name))
            except Exception as e:
                print(f"  [실패] {name}: {e}", file=sys.stderr)
                print("   pip install \"datasets>=2.19,<4\" 확인. "
                      "commonvoice 는 huggingface-cli login 도 필요하다.",
                      file=sys.stderr)
                continue
            print(f"  {name}({src['license']}): {len(raw_pool) - n0}문장")

        if raw_pool and cache:
            cache.write_text("\n".join(f"{t}\t{o}" for t, o in raw_pool),
                             encoding="utf-8")

        domain_texts = {t for t, _ in domain}
        pool = [(t, o) for t, o in raw_pool
                if t not in domain_texts
                and n_syllables(t) >= args.min_syll
                and n_syllables(t) <= args.max_syll]
        if raw_pool:
            print(f"  길이 조건({args.min_syll}~{args.max_syll}음절) 통과: "
                  f"{len(pool)}/{len(raw_pool)}문장")

        if pool:
            rng.shuffle(pool)     # 코퍼스 앞부분(같은 화자/주제)에 쏠리지 않게
            cands = [(t, units(t), utt_seconds(t)) for t, _ in pool]
            origin = dict(pool)
            # 목표 커버리지를 다 채우면 탐욕 선택이 멈춘다. 그때는 기준을 한 단계
            # 올려(각 유닛을 한 번씩 더 보게) 이어 뽑는다. 요청 수를 채울 때까지.
            picked, mc = [], args.min_count
            while len(picked) < need and cands and mc < args.min_count + 30:
                got = greedy_select(cands, need - len(picked), counts, mc)
                if not got:
                    mc += 1
                    continue
                picked += got
                taken = set(got)
                cands = [c for c in cands if c[0] not in taken]
            if mc > args.min_count:
                print(f"  커버리지 포화 → 최소 등장 횟수를 {args.min_count}"
                      f"에서 {mc}까지 올려가며 채움")
            chosen += [(t, "corpus", origin.get(t, "corpus")) for t in picked]
            print(f"  탐욕 선택: {len(picked)}문장 (요청 {need})")
    elif need:
        print("  (--source / --text-file 을 안 줘서 도메인 문장만 씁니다)")

    # 3) 학습/평가 분리 — 그룹별로 같은 비율씩 뗀다
    by_group = {}
    for item in chosen:
        by_group.setdefault(item[1], []).append(item)
    train, evalset = [], []
    for g, items in by_group.items():
        items = items[:]
        rng.shuffle(items)
        k = int(round(len(items) * args.eval_ratio))
        # 웨이크워드는 평가에서 빠지면 안 되고, 학습에서도 빠지면 안 된다
        k = min(max(k, 1 if len(items) > 3 else 0), len(items) - 1)
        evalset += items[:k]
        train += items[k:]
    rng.shuffle(train)
    rng.shuffle(evalset)

    # 4) 리포트
    print()
    for kind, v in coverage_report(counts, args.min_count).items():
        print(f"  {kind}: 종류 {v['종류']:4d} / "
              f"{args.min_count}회 이상 {v[f'{args.min_count}회 이상']:4d} / "
              f"1회뿐 {v['1회뿐']:4d}")
    n_wake = sum(1 for t, _, _ in chosen if "제리" in t)
    print(f"\n  전체 {len(chosen)}문장 — 웨이크워드 포함 {n_wake} / "
          f"미포함 {len(chosen) - n_wake}")
    print(f"  학습 {len(train)} / 평가 {len(evalset)}")
    print(f"  예상 녹음 시간: 학습 {estimate_minutes([t for t, _, _ in train], args.passes):.0f}분"
          f"({args.passes}회차) + 평가 {estimate_minutes([t for t, _, _ in evalset]):.0f}분(1회차)")

    if args.dry_run:
        print("\n[dry-run] 파일을 쓰지 않았습니다.")
        return 0

    cmd = " ".join(sys.argv[1:]) or "(기본값)"

    def header(record_cmd):
        return HEADER.format(cmd=cmd, n_train=len(train), n_eval=len(evalset),
                             record_cmd=record_cmd)

    write_prompts(OUT_TRAIN, [t for t, _, _ in train], header(
        f"python3 training/record_dataset.py --speaker {args.speaker} "
        f"--passes {args.passes} --fast"))
    write_prompts(OUT_EVAL, [t for t, _, _ in evalset], header(
        f"python3 training/record_dataset.py "
        f"--prompts training/prompts_eval.txt --speaker {args.speaker}_eval "
        f"--passes 1"))
    OUT_META.write_text(json.dumps({
        "cmd": cmd,
        "seed": args.seed,
        "min_count": args.min_count,
        "coverage": {k: v for k, v in coverage_report(counts, args.min_count).items()},
        "train": [{"text": t, "group": g, "origin": o} for t, g, o in train],
        "eval": [{"text": t, "group": g, "origin": o} for t, g, o in evalset],
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n  -> {OUT_TRAIN.name} / {OUT_EVAL.name} / {OUT_META.name} 저장")
    print("  코퍼스 문장을 쓴 경우 출처 표기(CC BY)는 hf_model_card.md 에 남길 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
