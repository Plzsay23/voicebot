#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
볼트 검색 서버 — 파이의 음성비서가 옵시디언 리서치위키를 RAG 로 쓰기 위한 것.

**볼트는 파이로 복제하지 않는다.** PC 가 켜져 있을 때만 이 서버가 뜨고, 파이는
발화마다 여기에 물어본다. 꺼져 있으면 조용히 RAG 없이 답한다 — 원격 LLM
(`REMOTE_LLM_URL`) 과 완전히 같은 정책이라 "PC 켜면 똑똑해진다" 로 일관된다.

의존성이 없다(표준 라이브러리만). 검색은 **문자 bigram BM25** 다:

- 형태소 분석기 없이 한국어가 된다.
- 질의가 STT 를 통과해서 오기 때문에 오타에 강해야 한다. "연합학습"이
  "연합 학쓥"으로 들어와도 bigram 이 3개 중 2개는 겹친다. 임베딩보다 이쪽이
  이 상황엔 낫고, onnxruntime 세션을 하나 더 띄우지 않아도 된다.

엔드포인트:
    GET /health              → {"ok":true,"docs":N,"chunks":M,...}
    GET /search?q=...&k=3&maxchars=600
                             → {"results":[{"title","path","text","score"},...]}

환경변수: VAULT_DIR VAULT_PORT VAULT_HOST VAULT_EXCLUDE VAULT_RESCAN_SECONDS
"""

import json
import os
import pickle
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import log
from pathlib import Path
from urllib.parse import urlparse, parse_qs

BASE_DIR = Path(__file__).resolve().parent.parent

VAULT = Path(os.getenv("VAULT_DIR", "/mnt/c/ysj/ResearchWiki"))
HOST = os.getenv("VAULT_HOST", "0.0.0.0")
PORT = int(os.getenv("VAULT_PORT", "8081"))

# 인덱싱에서 뺄 최상위 폴더(쉼표 구분). 볼트에는 개인 기록도 들어 있으므로
# 음성비서에게 읽히고 싶지 않은 폴더가 있으면 여기에 적는다. 예:
#   VAULT_EXCLUDE=06_Claude,05_Logs
EXCLUDE = {p.strip().strip("/") for p in os.getenv("VAULT_EXCLUDE", "").split(",") if p.strip()}
# 이건 항상 뺀다(내용이 아니거나 볼트 메타).
ALWAYS_EXCLUDE = {".git", ".obsidian", ".trash", ".smart-env", "node_modules", "99_Assets"}

# 한 조각의 최대 길이. 논문 원문이 180KB 통짜로 들어 있어서 헤딩으로 자른 뒤에도
# 더 잘라야 한다. 너무 크면 BM25 가 긴 문서에 유리해지고, 반환해도 컨텍스트 예산을
# 넘겨 잘린다.
MAX_CHUNK_CHARS = int(os.getenv("VAULT_MAX_CHUNK", "1200"))

# 볼트 변경 확인 주기(초). stat 만 훑는데도 `/mnt/c`(9p) 에서 4~5초 걸린다 —
# 싸지 않다. 백그라운드 스레드가 도는 것이라 요청을 막지는 않지만, 그래도
# 1분마다 돌릴 이유는 없어서 5분으로 둔다. 연구 노트의 신선도로는 충분하다.
RESCAN_SECONDS = float(os.getenv("VAULT_RESCAN_SECONDS", "300"))

CACHE = Path(os.getenv("VAULT_INDEX_CACHE", str(BASE_DIR / ".vault_index.pkl")))
CACHE_VERSION = 2  # 인덱스 포맷이 바뀌면 올린다(옛 캐시를 자동으로 버리게)

ALIASES_PATH = Path(os.getenv("VAULT_ALIASES", str(BASE_DIR / "scripts" / "vault_aliases.json")))

# BM25 파라미터. 기본값에서 손댈 이유는 없었다.
BM25_K1 = 1.2
BM25_B = 0.75
# 문서의 이 비율 이상에 나타나는 term 은 점수에 기여하지 않는다. 한국어 bigram 은
# "하는", "이다" 처럼 거의 모든 조각에 나오는 것이 있어서, 안 걸러내면 느리기만
# 하고 순위에는 도움이 안 된다.
MAX_DF_RATIO = 0.30


# ---------------- 텍스트 → term ----------------
# 영문/숫자는 단어 통째로, 한글은 bigram. 한글을 통째로 쓰면 조사 하나에
# 안 맞고, 영문을 bigram 으로 쪼개면 "fed" 가 "federated" 아무데나 붙는다.
_WORD_RE = re.compile(r"[a-z0-9]+|[가-힣]+")


def terms(text: str):
    for m in _WORD_RE.finditer(text.lower()):
        tok = m.group()
        if tok[0] < "가":  # ascii(영문/숫자)
            yield tok
        elif len(tok) == 1:
            yield tok
        else:
            for i in range(len(tok) - 1):
                yield tok[i:i + 2]


# ---------------- 문서 → 조각 ----------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# frontmatter 에서 살릴 줄. 나머지(originSessionId, modified 등)는 검색 잡음이다.
_FM_KEEP_RE = re.compile(r"^\s*(name|title|description|tags|aliases)\s*:", re.I)


def strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() in ("---", "..."):
            kept = [l for l in lines[1:i] if _FM_KEEP_RE.match(l)]
            return "\n".join(kept + lines[i + 1:])
    return text


# 반환할 때만 걷어내는 것들. **색인은 원문 그대로 한다** — 사용자가 "위키링크"나
# "볼드" 같은 표기까지 검색어로 쓸 일은 없지만, 원문을 색인해야 오프셋 고민이 없다.
# 여기서 지우는 목적은 순전히 **컨텍스트 예산 절약**이다. 파이4 에서는 `**` 하나가
# 그대로 대기시간이다.
_SNIP_FM_DROP_RE = re.compile(r"^\s*(name|tags|aliases|metadata|node_type|type|originSessionId|modified)\s*:.*$",
                              re.I | re.M)
_SNIP_FM_DESC_RE = re.compile(r"^\s*description\s*:\s*", re.I | re.M)
_SNIP_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")   # [[a|b]] -> a
_SNIP_MDLINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")              # [글](url) -> 글
# `_` 는 일부러 안 지운다. 이 볼트는 `eval_stt.py`, `voice_common.py` 처럼 밑줄이
# 든 식별자가 많아서, 지우면 LLM 이 잘못된 이름을 말한다. 읽어줄 때는 TTS 쪽
# strip_for_tts 가 어차피 걷어낸다.
_SNIP_MARK_RE = re.compile(r"[*`#>]+")


def clean_snippet(text: str) -> str:
    text = _SNIP_FM_DROP_RE.sub("", text)
    text = _SNIP_FM_DESC_RE.sub("", text)
    text = _SNIP_WIKILINK_RE.sub(r"\1", text)
    text = _SNIP_MDLINK_RE.sub(r"\1", text)
    text = _SNIP_MARK_RE.sub(" ", text)
    text = text.replace('"', "").replace("|", " ")
    return " ".join(text.split())


def split_long(body: str, limit: int):
    """빈 줄 경계에서 limit 이하로 묶는다. 문단 하나가 limit 을 넘으면 그것만 자른다."""
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(p) > limit:
            if cur:
                out.append(cur)
                cur = ""
            for i in range(0, len(p), limit):
                out.append(p[i:i + limit])
            continue
        if cur and len(cur) + len(p) + 2 > limit:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        out.append(cur)
    return out


def iter_chunks(text: str):
    """(헤딩 경로, 본문) 목록. 헤딩 단위로 자르고 큰 것은 문단 단위로 더 자른다."""
    text = strip_frontmatter(text)
    stack = []      # [(level, title)]
    buf = []
    results = []

    def flush():
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        crumb = " > ".join(t for _, t in stack)
        for piece in split_long(body, MAX_CHUNK_CHARS):
            results.append((crumb, piece))

    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()  # 이 본문은 직전 헤딩의 것이다. 스택을 고치기 전에 비운다.
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2).strip()))
        else:
            buf.append(line)
    flush()

    if not results:
        # 헤딩만 있고 본문이 없는 인덱스 노트. 제목이라도 검색되게 남긴다.
        head = " > ".join(t for _, t in stack)
        if head:
            results.append((head, head))
    return results


# ---------------- 인덱스 ----------------
class Index:
    def __init__(self):
        self.chunks = []        # [{"path","title","text"}]
        self.postings = {}      # term -> [(chunk_id, tf), ...]
        self.doclen = []
        self.avglen = 1.0
        self.ndocs = 0
        self.fingerprint = ""
        self.built_at = 0.0

    # --- 검색 ---
    def search(self, query: str, k: int = 3):
        if not self.chunks:
            return []
        qt = Counter(terms(query))
        if not qt:
            return []
        n = len(self.chunks)
        max_df = max(1, int(n * MAX_DF_RATIO))
        scores = {}
        for t, qf in qt.items():
            pl = self.postings.get(t)
            if not pl:
                continue
            df = len(pl)
            if df > max_df:
                continue
            idf = log(1 + (n - df + 0.5) / (df + 0.5))
            wq = idf * qf
            for cid, tf in pl:
                dl = self.doclen[cid]
                denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avglen)
                scores[cid] = scores.get(cid, 0.0) + wq * (tf * (BM25_K1 + 1)) / denom
        if not scores:
            return []
        top = sorted(scores.items(), key=lambda kv: -kv[1])

        # 같은 파일에서 여러 조각이 상위를 다 차지하면 다양성이 없다. 파일당 1개.
        out, seen = [], set()
        for cid, sc in top:
            c = self.chunks[cid]
            if c["path"] in seen:
                continue
            seen.add(c["path"])
            out.append({
                "path": c["path"],
                "title": c["title"],
                "text": c["text"],
                "score": round(sc, 3),
            })
            if len(out) >= k:
                break
        return out


def scan_files():
    files = []
    for p in sorted(VAULT.rglob("*.md")):
        rel = p.relative_to(VAULT)
        parts = set(rel.parts[:-1])
        if parts & ALWAYS_EXCLUDE:
            continue
        if EXCLUDE and (parts & EXCLUDE):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        files.append((rel.as_posix(), p, st.st_size, int(st.st_mtime)))
    return files


def fingerprint_of(files):
    import hashlib
    h = hashlib.sha1()
    h.update(f"v{CACHE_VERSION}|{MAX_CHUNK_CHARS}|{sorted(EXCLUDE)}".encode())
    for rel, _p, size, mtime in files:
        h.update(f"{rel}|{size}|{mtime}\n".encode())
    return h.hexdigest()


def build_index(files, fp) -> Index:
    idx = Index()
    postings = {}
    for rel, path, _size, _mtime in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        stem = path.stem
        for crumb, body in iter_chunks(text):
            title = f"{stem} > {crumb}" if crumb else stem
            cid = len(idx.chunks)
            idx.chunks.append({"path": rel, "title": title, "text": body})
            # 제목도 본문과 함께 색인한다. 파일명·헤딩이 질의어와 겹치는 일이 많다.
            tf = Counter(terms(f"{title}\n{body}"))
            idx.doclen.append(sum(tf.values()) or 1)
            for t, c in tf.items():
                postings.setdefault(t, []).append((cid, c))
    idx.postings = postings
    idx.ndocs = len(files)
    idx.avglen = (sum(idx.doclen) / len(idx.doclen)) if idx.doclen else 1.0
    idx.fingerprint = fp
    idx.built_at = time.time()
    return idx


# 캐시는 **평범한 dict 로** 절인다. Index 인스턴스를 그대로 pickle 하면 클래스
# 경로가 실행 방식에 따라 __main__.Index / vault_server.Index 로 갈려서, 서버로
# 띄웠다 모듈로 import 했다 하면 서로의 캐시를 못 읽는다(AttributeError).
_BLOB_FIELDS = ("chunks", "postings", "doclen", "avglen", "ndocs", "fingerprint")


def save_cache(idx: Index):
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        blob = {"v": CACHE_VERSION}
        blob.update({f: getattr(idx, f) for f in _BLOB_FIELDS})
        with CACHE.open("wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"[vault] 캐시 저장 실패({type(e).__name__}) — 동작에는 지장 없다", flush=True)


def load_or_build() -> Index:
    files = scan_files()
    fp = fingerprint_of(files)
    if CACHE.exists():
        try:
            with CACHE.open("rb") as f:
                blob = pickle.load(f)
            if (isinstance(blob, dict) and blob.get("v") == CACHE_VERSION
                    and blob.get("fingerprint") == fp):
                idx = Index()
                for f_ in _BLOB_FIELDS:
                    setattr(idx, f_, blob[f_])
                idx.built_at = time.time()
                print(f"[vault] 캐시 사용: {len(idx.chunks)}조각 / {idx.ndocs}문서", flush=True)
                return idx
        except Exception as e:
            print(f"[vault] 캐시를 못 읽었다({type(e).__name__}) → 다시 만든다", flush=True)
    t0 = time.time()
    print(f"[vault] 인덱싱 중... ({len(files)}문서)", flush=True)
    idx = build_index(files, fp)
    print(f"[vault] 완료: {len(idx.chunks)}조각, {len(idx.postings)}term, {time.time()-t0:.1f}s", flush=True)
    save_cache(idx)
    return idx


# ---------------- 별칭 ----------------
def load_aliases():
    """STT 가 흘려 듣는 고유명사를 되살리기 위한 치환표. 없으면 빈 dict."""
    try:
        with ALIASES_PATH.open(encoding="utf-8") as f:
            data = json.load(f)
        # "_" 로 시작하는 키는 주석용이다.
        return {str(k): str(v) for k, v in data.items()
                if k and v and not str(k).startswith("_")}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[vault] 별칭 파일 무시({type(e).__name__}): {ALIASES_PATH}", flush=True)
        return {}


def expand_query(q: str, aliases: dict) -> str:
    """별칭을 '원문 + 정식표기' 로 덧붙인다. 치환이 아니라 추가 — 원문이 맞았을
    수도 있으므로 지우면 손해다."""
    extra = [v for k, v in aliases.items() if k and k in q]
    return q + " " + " ".join(extra) if extra else q


# ---------------- 상태(핫 리로드) ----------------
class State:
    """인덱스를 들고 있고, 볼트 변경은 **백그라운드 스레드**가 따라잡는다.

    처음엔 요청 처리 중에 재스캔을 했는데, `scan_files()` 가 `/mnt/c`(9p) 에서
    **4~5초** 걸린다. 60초에 한 번 그 시간 동안 락을 쥐고 있으니, 하필 그때 걸린
    요청은 파이의 1초 프로브/3초 검색 타임아웃을 그냥 넘겨버렸다. 실제로 답변
    도중에 볼트가 통째로 빠지는 것으로 나타났다(그러면 LLM 이 지어낸다).

    그래서 요청 경로에서는 참조 하나만 읽는다. 교체는 통째로 하므로(GIL 아래
    참조 대입은 원자적) 검색 중에 인덱스가 반쯤 바뀌는 일은 없고 락도 필요 없다.
    """

    def __init__(self):
        self.index = load_or_build()
        self.aliases = load_aliases()
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def get(self) -> Index:
        return self.index

    def _watch(self):
        while not self._stop.wait(RESCAN_SECONDS):
            try:
                files = scan_files()
                fp = fingerprint_of(files)
                if fp == self.index.fingerprint:
                    continue
                print("[vault] 볼트가 바뀌었다 → 재인덱싱", flush=True)
                idx = build_index(files, fp)
                self.index = idx
                self.aliases = load_aliases()
                save_cache(idx)
                print(f"[vault] 재인덱싱 완료: {len(idx.chunks)}조각", flush=True)
            except Exception as e:
                # 재인덱싱이 실패해도 검색은 계속돼야 한다. 옛 인덱스를 그대로 쓴다.
                print(f"[vault] 재인덱싱 실패({type(e).__name__}) — 이전 인덱스 유지",
                      flush=True)


STATE = None  # main 에서 채운다


# ---------------- HTTP ----------------
class Handler(BaseHTTPRequestHandler):
    server_version = "VaultSearch/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)

        if u.path in ("/health", "/"):
            idx = STATE.get()
            return self._json(200, {
                "ok": True,
                "vault": str(VAULT),
                "docs": idx.ndocs,
                "chunks": len(idx.chunks),
                "excluded": sorted(EXCLUDE),
            })

        if u.path == "/search":
            q = (qs.get("q") or [""])[0].strip()
            if not q:
                return self._json(400, {"error": "q 가 비었다"})
            try:
                k = max(1, min(10, int((qs.get("k") or ["3"])[0])))
            except ValueError:
                k = 3
            try:
                maxchars = max(80, min(4000, int((qs.get("maxchars") or ["600"])[0])))
            except ValueError:
                maxchars = 600
            try:
                minscore = float((qs.get("minscore") or ["0"])[0])
            except ValueError:
                minscore = 0.0

            idx = STATE.get()
            hits = idx.search(expand_query(q, STATE.aliases), k)

            # 관련성 문턱. BM25 는 term 이 하나만 겹쳐도 뭔가를 돌려주므로,
            # 이게 없으면 "피자 시켜줘" 에 논문 노트가 딸려간다.
            #
            # 2026-07-28 실측(178문서 볼트, 질문 15개): 관련 있는 질문의 1위 점수는
            # 15.4~66.8, 무관한 질문은 3.9~10.7 로 갈라졌다. 파이 쪽 기본값 13 이
            # 그 사이다. 볼트가 커지면 다시 재보라(scripts/ 의 점수 출력이 있다).
            if minscore > 0:
                hits = [h for h in hits if h["score"] >= minscore]
            # 1위에 비해 한참 처지는 것도 버린다. 예산을 반쯤 쓸모없는 조각에
            # 나눠주느니 좋은 것 하나를 길게 주는 편이 낫다.
            if hits:
                floor = hits[0]["score"] * 0.35
                hits = [h for h in hits if h["score"] >= floor]

            # 컨텍스트 예산을 여기서 지킨다. 파이4 는 프롬프트 토큰이 그대로
            # 대기시간이라, 넘겨받은 쪽이 자르는 것보다 여기서 자르는 게 확실하다.
            # 파이는 "- {title}: {text}" 로 붙이므로 제목도 예산에 넣어 센다.
            # 안 그러면 요청한 500자가 실제로는 600자가 되어 예산이 거짓말이 된다.
            budget = maxchars
            out = []
            for h in hits:
                if budget <= 0:
                    break
                overhead = len(h["title"]) + 4
                share = max(80, budget // max(1, (len(hits) - len(out)))) - overhead
                if share < 60:      # 제목이 길어 남는 게 없으면 이 건은 버린다
                    continue
                text = clean_snippet(h["text"])
                if len(text) > share:
                    text = text[:share].rstrip() + "…"
                budget -= len(text) + overhead
                out.append({**h, "text": text})
            return self._json(200, {"query": q, "results": out})

        return self._json(404, {"error": "no such endpoint"})

    def log_message(self, fmt, *args):
        if os.getenv("VAULT_VERBOSE"):
            sys.stderr.write("[vault] " + (fmt % args) + "\n")


def main():
    global STATE
    if not VAULT.is_dir():
        print(f"[vault] 볼트가 없다: {VAULT}\n"
              f"        VAULT_DIR 로 경로를 지정하라.", file=sys.stderr)
        return 1
    STATE = State()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    srv.daemon_threads = True
    print(f"[vault] 준비 완료 — http://{HOST}:{PORT}/search?q=...", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
