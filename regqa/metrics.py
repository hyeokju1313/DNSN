"""검색·재순위 평가 지표. 순위 목록은 코퍼스 인덱스, 정답은 조문 ID."""
from __future__ import annotations

import numpy as np

LEVELS = ["L1", "L2", "L3", "L4", "L5a"]


def score_one(rank_ids: list[str], q: dict) -> dict:
    gold = set(q["gold_any"])
    pos = [i for i, r in enumerate(rank_ids) if r in gold]
    first = pos[0] if pos else None
    out = {"hit1": float(first == 0), "hit5": float(first is not None and first < 5),
           "rr": 0.0 if first is None else 1.0 / (first + 1)}
    if "gold_all" in q:
        top5 = set(rank_ids[:5])
        groups = [any(g in top5 for g in grp) for grp in q["gold_all"]]
        out["all5"] = float(all(groups))
        out["group5"] = float(np.mean(groups))
    return out


def summarize(per_q: list[dict], qs: list[dict]) -> dict:
    res = {}
    for lv in LEVELS + ["ALL"]:
        rows = [r for r, q in zip(per_q, qs) if (lv == "ALL" and q["level"] != "L5b") or q["level"] == lv]
        if not rows:
            continue
        d = {k: float(np.mean([r[k] for r in rows])) for k in ["hit1", "hit5", "rr"]}
        if lv == "L4":
            d["all5"] = float(np.mean([r["all5"] for r in rows]))
            d["group5"] = float(np.mean([r["group5"] for r in rows]))
        d["n"] = len(rows)
        res[lv] = d
    return res


def fmt(res: dict) -> str:
    parts = []
    for lv, d in res.items():
        if lv not in LEVELS + ["ALL"]:
            continue
        s = f"{lv}: R@1 {d['hit1']:.2f} R@5 {d['hit5']:.2f} MRR {d['rr']:.2f}"
        if "all5" in d:
            s += f" 전부@5 {d['all5']:.2f}"
        parts.append(s)
    return " | ".join(parts)
