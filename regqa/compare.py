"""재순위 모델 간 질문 단위 쌍체 부트스트랩 검정.

캐시한 특징으로 rerank.py와 같은 절차(C를 검증 MRR로 선택)를 다시 맞춘 뒤,
- 검증(역클로즈 277개): 역순위(RR) 차이
- 평가(92개, L5b 제외): 1위 정답 여부 차이
의 95% 구간을 구한다. 재순위 점수만 쓴 경우(only)를 비교한다(합친 경우는 모두 dense와 거의 같다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from regqa.corpus import DER  # noqa: E402
from regqa.rerank import CS, K_EVAL, K_VAL, build_pairs, feats_lr_pair  # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def per_question(F, sets, P):
    sc = StandardScaler().fit(F["train"])
    Xtr, y = sc.transform(F["train"]), sets["train"]["y"]
    ids = P["corpus_ids"]
    best = None
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, y)
        S = m.decision_function(sc.transform(F["val"])).reshape(-1, K_VAL)
        rr = []
        for r, meta, s in zip(P["val"], sets["val"]["meta"], S):
            order = [meta["cands"][i] for i in np.argsort(-s)]
            rr.append(1.0 / (order.index(r["pos"]) + 1) if r["pos"] in order else 0.0)
        if best is None or np.mean(rr) > np.mean(best[1]):
            best = (m, rr)
    m, rr_val = best
    S = m.decision_function(sc.transform(F["eval"])).reshape(-1, K_EVAL)
    hit1 = []
    for q, meta, s in zip(P["eval"], sets["eval"]["meta"], S):
        if q["level"] == "L5b":
            continue
        top = ids[meta["cands"][int(np.argmax(s))]]
        hit1.append(float(top in set(q["gold_any"])))
    return np.array(rr_val), np.array(hit1)


def boot(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    P, sets = build_pairs()
    feats = {"lr_pair": {k: feats_lr_pair(v["q"], v["a"]) for k, v in sets.items()}}
    for v in ["connectome", "random", "degree_shuffle"]:
        z = np.load(DER / f"feats_res_{v}_interaction_rho1.2_leak0.5_T8_s0.npz")
        feats[f"res_{v}_ix"] = {k: z[k] for k in z.files}
    out = {name: per_question(F, sets, P) for name, F in feats.items()}
    res = {}
    for other in ["res_random_ix", "res_degree_shuffle_ix", "lr_pair"]:
        a, b = out["res_connectome_ix"], out[other]
        res[f"connectome_ix - {other}"] = {"val_rr": boot(a[0], b[0]), "eval_hit1": boot(a[1], b[1])}
    for k, v in res.items():
        print(k, {m: tuple(round(x, 3) for x in t) for m, t in v.items()})
    (RES / "paired_bootstrap.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
