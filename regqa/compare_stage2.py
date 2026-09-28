"""Stage 2 커넥톰 vs 무작위 망: 저장한 가중치로 질문별 점수를 다시 계산해 쌍체 부트스트랩.

재순위 점수만 쓴 경우(only)를 비교한다. 검증(역클로즈 277개)은 역순위, 평가(92개)는 1위 정답 여부.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flycns.connectome import build_matrix, groups, load_neurons  # noqa: E402
from regqa.compare import boot  # noqa: E402
from regqa.rerank import K_EVAL, K_VAL, build_pairs, feats_lr_pair  # noqa: E402
from regqa.stage2_train import FlyReranker  # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def per_question(variant, P, sets, F, dev="mps"):
    neurons = load_neurons()
    g = groups(neurons)
    keys, type_idx = np.unique(neurons["type_key"].values, return_inverse=True)
    W, _ = build_matrix(variant, rho=1.2, seed=0)
    n, d = W.shape[0], 1024
    rng = np.random.default_rng(2000)
    olf, vis = g["olfactory"], g["visual"]
    rows = np.concatenate([np.repeat(olf, d), np.repeat(vis, d)])
    cols = np.concatenate([np.tile(np.arange(d), len(olf)), np.tile(np.arange(d, 2 * d), len(vis))])
    win = sp.csr_matrix(((rng.standard_normal(len(rows)) / np.sqrt(d)).astype(np.float32), (rows, cols)), shape=(n, 2 * d))
    m = FlyReranker(W, win, type_idx, len(keys), g["readout"], dev=dev).to(dev)
    m.load_state_dict(torch.load(RES / "stage2" / f"stage2_{variant}_rho1.2_s0.pt"))
    m.eval()

    def scores(split):
        X = torch.from_numpy(F[split])
        with torch.no_grad():
            return np.concatenate([m(X[i:i + 256].T.contiguous().to(dev)).cpu().numpy() for i in range(0, len(X), 256)])

    ids = P["corpus_ids"]
    rr = []
    for r, meta, s in zip(P["val"], sets["val"]["meta"], scores("val").reshape(-1, K_VAL)):
        order = [meta["cands"][i] for i in np.argsort(-s)]
        rr.append(1.0 / (order.index(r["pos"]) + 1) if r["pos"] in order else 0.0)
    hit = []
    for q, meta, s in zip(P["eval"], sets["eval"]["meta"], scores("eval").reshape(-1, K_EVAL)):
        if q["level"] != "L5b":
            hit.append(float(ids[meta["cands"][int(np.argmax(s))]] in set(q["gold_any"])))
    return np.array(rr), np.array(hit)


def main():
    P, sets = build_pairs()
    tr = feats_lr_pair(sets["train"]["q"], sets["train"]["a"])
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    F = {k: ((feats_lr_pair(v["q"], v["a"]) - mu) / sd).astype(np.float32) for k, v in sets.items()}
    c = per_question("connectome", P, sets, F)
    r = per_question("random", P, sets, F)
    res = {"stage2 connectome - random": {"val_rr": boot(c[0], r[0]), "eval_hit1": boot(c[1], r[1]),
                                          "means": {"connectome": [float(c[0].mean()), float(c[1].mean())],
                                                    "random": [float(r[0].mean()), float(r[1].mean())]}}}
    print(json.dumps(res, ensure_ascii=False, indent=1))
    (RES / "paired_bootstrap_stage2.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
