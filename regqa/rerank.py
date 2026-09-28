"""(질문, 후보 조문) 쌍의 관련성을 매기는 재순위 모델들.

특징 종류
- lr_pair:   [q*a, |q-a|] (2,048차원) -> 로지스틱 회귀. 커넥톰 없는 학습형 기준선
- randfeat:  tanh(R [q; a]) (2,129차원, R 무작위) -> 로지스틱 회귀. '뇌 없이 무작위 비선형 특징만' 대조군
- res_<망>:  질문 벡터 -> 후각 감각 뉴런, 조문 벡터 -> 시각 감각 뉴런에 T스텝 동안 넣고
             하행 뉴런 + 운동 뉴런 2,129개의 마지막 2스텝 평균 활동 -> 로지스틱 회귀
             <망> = connectome | random | degree_shuffle | weight_shuffle
- res_<망>_ix: 입력을 비교 신호 [q*a, |q-a|]로 바꾼 버전(lr_pair와 같은 입력이 뇌를 거친다)
후보는 dense(KURE-v1) 상위 20개. 최종 점수는 (a) 재순위 점수만, (b) dense 코사인 + λ·재순위 점수 두 가지로 보고한다.
사용 예:
  python -u regqa/rerank.py --feats lr_pair randfeat res_connectome res_random
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flycns.connectome import build_matrix, groups, load_neurons  # noqa: E402
from flycns.reservoir import run_static  # noqa: E402
from regqa.corpus import DER, embed  # noqa: E402
from regqa.metrics import fmt, score_one, summarize  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
K_EVAL = 20
K_VAL = 10
CS = [0.01, 0.1, 1.0, 10.0]
LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def build_pairs():
    P = json.load(open(DER / "pairs.json"))
    E = np.load(DER / "emb_corpus.npy")
    PE = np.load(DER / "emb_pseudo.npy")
    QE = np.load(DER / "emb_eval_v0.npy")
    DE = np.load(DER / "emb_dev_v0.npy")
    sets = {}
    # 학습: 정답 1 + 어려운 오답 7
    q, a, y = [], [], []
    for r in P["train"]:
        for c, lab in [(r["pos"], 1)] + [(n, 0) for n in r["negs"]]:
            q.append(PE[r["q_idx"]]); a.append(E[c]); y.append(lab)
    sets["train"] = {"q": np.array(q), "a": np.array(a), "y": np.array(y)}

    def listwise(name, recs, QM, k, qkey):
        q, a, meta = [], [], []
        for r in recs:
            qe = QM[r[qkey]]
            top = np.argsort(-(E @ qe))[:k]
            for c in top:
                q.append(qe); a.append(E[c])
            meta.append({"cands": top.tolist(), "dense": (E[top] @ qe).tolist()})
        sets[name] = {"q": np.array(q), "a": np.array(a), "meta": meta}

    for i, r in enumerate(P["eval"]):
        r["_i"] = i
    for i, r in enumerate(P["dev"]):
        r["_i"] = i
    listwise("val", P["val"], PE, K_VAL, "q_idx")
    listwise("eval", P["eval"], QE, K_EVAL, "_i")
    listwise("dev", P["dev"], DE, K_EVAL, "_i")
    return P, sets


def feats_lr_pair(q, a):
    return np.concatenate([q * a, np.abs(q - a)], 1)


def feats_randfeat(q, a, dim=2129, seed=0):
    R = np.random.default_rng(seed).standard_normal((q.shape[1] * 2, dim)).astype(np.float32)
    return np.tanh(np.concatenate([q, a], 1) @ R)


def reservoir_features(variant, sets, rho=1.2, leak=0.5, steps=8, seed=0, log=print, mode="separate"):
    """mode
    - separate:    질문 q -> 후각 뉴런, 조문 a -> 시각 뉴런 (둘을 비교하는 일은 뇌에 맡긴다)
    - interaction: 비교 신호 q*a -> 후각 뉴런, |q-a| -> 시각 뉴런 (lr_pair와 같은 입력, 학습 쌍 기준 z-score)
    """
    suffix = "" if mode == "separate" else f"_{mode}"
    cache = DER / f"feats_res_{variant}{suffix}_rho{rho}_leak{leak}_T{steps}_s{seed}.npz"
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    neurons = load_neurons()
    g = groups(neurons)
    W, meta = build_matrix(variant, rho=rho, seed=seed)
    n = W.shape[0]
    rng = np.random.default_rng(2000 + seed)
    d = 1024
    olf, vis = g["olfactory"], g["visual"]
    # 질문 1024차원 -> 후각 뉴런, 조문 1024차원 -> 시각 뉴런. 가중치 N(0, 1): 단위 벡터 입력의 전류 표준편차가 1
    rows = np.concatenate([np.repeat(olf, d), np.repeat(vis, d)])
    cols = np.concatenate([np.tile(np.arange(d), len(olf)), np.tile(np.arange(d, 2 * d), len(vis))])
    vals = rng.standard_normal(len(rows)).astype(np.float32)
    if mode == "interaction":
        vals /= np.sqrt(d)            # z-score된 1024차원 입력의 전류 표준편차를 1로
    Win = sp.csr_matrix((vals, (rows, cols)), shape=(n, 2 * d))
    win_path = DER / "inputs" / f"win_regqa{suffix}_s{seed}.npz"
    win_path.parent.mkdir(parents=True, exist_ok=True)
    if not win_path.exists():
        sp.save_npz(win_path, Win)
    out = {}
    if mode == "interaction":
        tr = feats_lr_pair(sets["train"]["q"], sets["train"]["a"])
        mu, sd = tr.mean(0), tr.std(0) + 1e-8
    for name, s in sets.items():
        t0 = time.time()
        if mode == "interaction":
            U = ((feats_lr_pair(s["q"], s["a"]) - mu) / sd).T.astype(np.float32)
        else:
            U = np.concatenate([s["q"], s["a"]], 1).T.astype(np.float32)
        out[name] = run_static(meta["path"], win_path, U, leak, g["readout"], steps=steps, last_k=2, log=log)
        log(f"  [{variant}] {name}: {U.shape[1]} pairs in {time.time() - t0:.0f}s")
    np.savez(cache, **out)
    return out


def fit_and_eval(name, F, sets, P):
    """C와 λ를 검증 질문(역클로즈)의 MRR로 고르고, 평가·개발 질문에 적용한다."""
    ids = P["corpus_ids"]
    sc = StandardScaler().fit(F["train"])
    Xtr = sc.transform(F["train"])
    ytr = sets["train"]["y"]

    def rank_scores(model, split, k):
        s = model.decision_function(sc.transform(F[split]))
        return s.reshape(-1, k)

    def val_mrr(S, lam, comb):
        rr = []
        for r, meta, s in zip(P["val"], sets["val"]["meta"], S):
            score = np.array(meta["dense"]) + lam * s if comb else s
            order = [meta["cands"][i] for i in np.argsort(-score)]
            rr.append(1.0 / (order.index(r["pos"]) + 1) if r["pos"] in order else 0.0)
        return float(np.mean(rr))

    best = None
    for C in CS:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr)
        S = rank_scores(m, "val", K_VAL)
        sd = S.std() + 1e-9
        v_only = val_mrr(S, 0, False)
        v_comb = max((val_mrr(S / sd, lam, True), lam) for lam in LAMBDAS)
        if best is None or v_only > best["val_mrr_only"]:
            best = {"C": C, "model": m, "sd": sd, "val_mrr_only": v_only,
                    "val_mrr_comb": v_comb[0], "lambda": v_comb[1]}
    m, sd, lam = best["model"], best["sd"], best["lambda"]
    S_eval = rank_scores(m, "eval", K_EVAL) / sd
    S_dev = rank_scores(m, "dev", K_EVAL) / sd
    qs = [q for q in P["eval"] if q["level"] != "L5b"]
    res = {"C": best["C"], "lambda": lam, "val_mrr_only": best["val_mrr_only"], "val_mrr_comb": best["val_mrr_comb"]}
    for mode in ["only", "comb"]:
        per, conf_eval = [], []
        for q, meta, s in zip(P["eval"], sets["eval"]["meta"], S_eval):
            score = np.array(meta["dense"]) + lam * s if mode == "comb" else s
            o = np.argsort(-score)
            conf_eval.append(float(score[o[0]]))
            if q["level"] != "L5b":
                per.append(score_one([ids[meta["cands"][i]] for i in o], q))
        res[mode] = summarize(per, qs)
        # 답변 불가: 개발 질문으로 기준값 τ를 정한다(균형 정확도 최대)
        conf_dev = [float(np.max(np.array(meta["dense"]) + lam * s if mode == "comb" else s))
                    for meta, s in zip(sets["dev"]["meta"], S_dev)]
        ins = np.array([d["in_scope"] for d in P["dev"]])
        cands = sorted(set(conf_dev))
        tau = max(cands, key=lambda t: ((np.array(conf_dev)[ins] >= t).mean() + (np.array(conf_dev)[~ins] < t).mean()))
        lv = np.array([q["level"] for q in P["eval"]])
        ce = np.array(conf_eval)
        res[mode]["abstain"] = {"tau": tau, "reject_out_of_scope": float((ce[lv == "L5b"] < tau).mean()),
                                "false_reject_in_scope": float((ce[lv != "L5b"] < tau).mean())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["lr_pair", "randfeat", "res_connectome"])
    ap.add_argument("--rho", type=float, default=1.2)
    ap.add_argument("--leak", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=8)
    a = ap.parse_args()
    P, sets = build_pairs()
    print({k: len(v["q"]) for k, v in sets.items()}, flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    # 재순위 없는 dense 기준(같은 후보, 같은 평가 코드)
    ids = P["corpus_ids"]
    qs = [q for q in P["eval"] if q["level"] != "L5b"]
    per = [score_one([ids[c] for c in m["cands"]], q) for q, m in zip(P["eval"], sets["eval"]["meta"]) if q["level"] != "L5b"]
    print(f"[dense only] {fmt(summarize(per, qs))}", flush=True)
    for name in a.feats:
        t0 = time.time()
        if name == "lr_pair":
            F = {k: feats_lr_pair(v["q"], v["a"]) for k, v in sets.items()}
            tag = name
        elif name == "randfeat":
            F = {k: feats_randfeat(v["q"], v["a"]) for k, v in sets.items()}
            tag = name
        else:
            mode = "interaction" if name.endswith("_ix") else "separate"
            variant = name.replace("res_", "").replace("_ix", "")
            F = reservoir_features(variant, sets, rho=a.rho, leak=a.leak, steps=a.steps,
                                   log=lambda s: print(s, flush=True), mode=mode)
            tag = f"{name}_rho{a.rho}_leak{a.leak}_T{a.steps}"
        res = fit_and_eval(tag, F, sets, P)
        res["feature"] = tag
        res["feature_dim"] = int(F["train"].shape[1])
        (RES / f"rerank_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        ab = res["comb"]["abstain"]
        print(f"[{tag}] C={res['C']} λ={res['lambda']} val MRR only {res['val_mrr_only']:.3f} comb {res['val_mrr_comb']:.3f} "
              f"({time.time() - t0:.0f}s)\n   only: {fmt(res['only'])}\n   comb: {fmt(res['comb'])}\n"
              f"   abstain(comb): out-of-scope reject {ab['reject_out_of_scope']:.2f}, in-scope false reject {ab['false_reject_in_scope']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()
