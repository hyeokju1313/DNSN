"""재순위 모델 학습·평가용 (질문, 후보 조문) 쌍을 만든다.

- 후보: BM25와 KURE-v1 dense 점수를 RRF로 합친 상위 K개
- 학습 질문: 역클로즈(inverse cloze) 방식. 조문 안의 한 줄을 질문으로 쓰고 그 조문을 정답으로 삼는다.
  조문 단위로 학습 80%, 검증 20%로 나눈다
- 개발 질문(dev_v0_claude.jsonl): 답변 불가 기준값을 정하는 데만 쓴다
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.corpus import BM25, DER, REG, doc_text, embed, load_corpus, load_eval, rrf  # noqa: E402

K = 20
N_NEG = 7

DEV = [
    ("장학금은 중복으로 받을 수 있어?", True), ("휴학하면 받던 장학금은 어떻게 돼?", True),
    ("마이크로디그리 과정은 어떻게 이수해?", True), ("전공트랙 이수 기준이 뭐야?", True),
    ("학생 동아리를 새로 만들려면 어떻게 해?", True), ("장애학생은 어떤 지원을 받을 수 있어?", True),
    ("지도교수 상담은 몇 번 받아야 해?", True), ("원격수업 출석은 어떻게 인정돼?", True),
    ("재학증명서는 누가 발급받을 수 있어?", True), ("학적부는 아무나 볼 수 있어?", True),
    ("학교 축제에 어떤 가수가 와?", False), ("도서관 스터디룸 예약은 어떻게 해?", False),
    ("학교 앞 버스 정류장이 어디야?", False), ("교직원 식당 가격 알려줘", False),
    ("교내 헬스장 회원권 있어?", False), ("교내 편의점은 몇 시까지 해?", False),
    ("택배 보관함 위치가 어디야?", False), ("우산 빌려주는 곳 있어?", False),
    ("학교 로고 파일은 어디서 받아?", False), ("시험기간에 열람실 24시간 운영해?", False),
]


def write_dev():
    p = REG / "qa" / "dev_v0_claude.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for i, (q, ins) in enumerate(DEV):
            f.write(json.dumps({"id": f"dev-{i:02d}", "question": q, "in_scope": ins, "source": "claude"},
                               ensure_ascii=False) + "\n")
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def pseudo_queries(C, seed=0, per_article=3):
    rng = np.random.default_rng(seed)
    out = []
    for i, a in enumerate(C):
        lines = [re.sub(r"^\s*(?:[①-⑳➀-➉]|\d+\.|[가-하]\.)\s*", "", ln).strip()
                 for ln in a["text_clean"].split("\n")]
        lines = [ln for ln in lines if len(ln) >= 15 and not ln.startswith("|")]
        for ln in rng.permutation(lines)[:per_article]:
            out.append({"query": ln, "pos": i})
    return out


def main():
    C = load_corpus()
    docs = [doc_text(a) for a in C]
    E = embed(docs, "corpus")
    bm = BM25(docs)
    rng = np.random.default_rng(0)
    val_art = set(rng.choice(len(C), len(C) // 5, replace=False).tolist())

    def candidates(qtexts, qemb):
        out = []
        for qt, qe in zip(qtexts, qemb):
            dense = E @ qe
            bms = bm.scores(qt)
            fused = rrf(dense, bms)
            top = np.argsort(-fused)[:K]
            out.append({"cands": top.tolist(), "dense": dense[top].tolist(), "bm25": bms[top].tolist(),
                        "dense_rank": np.argsort(-dense)[:K].tolist(), "bm25_rank": np.argsort(-bms)[:K].tolist()})
        return out

    # 학습·검증 쌍
    P = pseudo_queries(C)
    PE = embed([p["query"] for p in P], "pseudo")
    cand = candidates([p["query"] for p in P], PE)
    train, val = [], []
    for j, (p, c) in enumerate(zip(P, cand)):
        negs = [x for x in c["cands"] if x != p["pos"]][:N_NEG]
        rec = {"qid": f"ps-{j:05d}", "q_idx": j, "pos": p["pos"], "cands": c["cands"], "negs": negs,
               "dense": c["dense"]}
        (val if p["pos"] in val_art else train).append(rec)
    # 평가·개발 질문
    Q = load_eval()
    QE = embed([q["question"] for q in Q], "eval_v0")
    ev = [{**q, **c} for q, c in zip(Q, candidates([q["question"] for q in Q], QE))]
    D = write_dev()
    DE = embed([d["question"] for d in D], "dev_v0")
    dv = [{**d, **c} for d, c in zip(D, candidates([d["question"] for d in D], DE))]
    DER.mkdir(parents=True, exist_ok=True)
    json.dump({"train": train, "val": val, "eval": ev, "dev": dv, "corpus_ids": [a["id"] for a in C]},
              open(DER / "pairs.json", "w"), ensure_ascii=False)
    print(f"corpus {len(C)}, pseudo train {len(train)} / val {len(val)}, eval {len(ev)}, dev {len(dv)}")


if __name__ == "__main__":
    main()
