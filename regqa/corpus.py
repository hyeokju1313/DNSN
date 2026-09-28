"""검색 코퍼스(학부 규정 본문 조문), 한국어 인코더 임베딩, BM25 색인.

코퍼스 범위(proposal_1 3절 + 이번 결정)
- 학부 규정 본문 조문만. 대학원 규정과 부칙은 뺀다
- 학칙(산업대)과 학칙시행세칙(산업대)(SEQ 33, 34)은 뺀다. 산업대 시절 입학생에게만 적용되는 옛 규정이라
  현행 학칙과 거의 같은 문장이 많아 검색에서 현행 조문과 경쟁한다
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REG = ROOT / "data" / "regulations"
DER = ROOT / "data" / "derived"
ENCODER = "nlpai-lab/KURE-v1"
# huggingface_hub 기본 다운로더가 멈춰서 curl로 받은 로컬 사본을 먼저 쓴다(같은 파일)
ENCODER_LOCAL = Path.home() / ".cache" / "aix_models" / "KURE-v1"
EXCLUDE_SEQ = {33, 34}


def load_corpus():
    arts = [json.loads(l) for l in open(REG / "articles.jsonl", encoding="utf-8")]
    return [a for a in arts if a["addenda"] is None and "대학원" not in a["path"] and "대학원" not in a["law"]
            and a["law_seq"] not in EXCLUDE_SEQ and a["text_clean"].strip()]


def doc_text(a) -> str:
    head = f"{a['law']} {a['article']}" + (f"({a['title']})" if a["title"] else "")
    return head + "\n" + a["text_clean"]


def load_eval(name="eval_v0_claude.jsonl"):
    return [json.loads(l) for l in open(REG / "qa" / name, encoding="utf-8")]


@lru_cache(1)
def encoder():
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    src = str(ENCODER_LOCAL) if (ENCODER_LOCAL / "model.safetensors").exists() else ENCODER
    return SentenceTransformer(src, device=dev)


def embed(texts, cache_name: str | None = None, batch_size=16):
    if cache_name:
        p = DER / f"emb_{cache_name}.npy"
        if p.exists():
            e = np.load(p)
            if len(e) == len(texts):
                return e
    e = encoder().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                         convert_to_numpy=True).astype(np.float32)
    if cache_name:
        DER.mkdir(parents=True, exist_ok=True)
        np.save(DER / f"emb_{cache_name}.npy", e)
    return e


# ---------------- BM25 (Kiwi 형태소) ----------------
_KEEP = ("NNG", "NNP", "NR", "SN", "SL", "VV", "VA", "XR", "MAG", "NNB")


@lru_cache(1)
def _kiwi():
    from kiwipiepy import Kiwi
    return Kiwi()


def tokenize(text: str):
    return [t.form for t in _kiwi().tokenize(text) if t.tag.startswith(_KEEP)]


class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(t) for t in self.toks])
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]
        self.len = [len(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        s = np.zeros(len(self.tf))
        for i, (tf, L) in enumerate(zip(self.tf, self.len)):
            for w in q:
                if w in tf:
                    f = tf[w]
                    s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
        return s


def rrf(*rank_lists, k=60, n=None):
    """reciprocal rank fusion. 각 입력은 점수 배열."""
    n = n or len(rank_lists[0])
    fused = np.zeros(n)
    for s in rank_lists:
        order = np.argsort(-s)
        fused[order] += 1.0 / (k + np.arange(1, n + 1))
    return fused
