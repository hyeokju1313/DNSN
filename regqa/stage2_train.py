"""Stage 2: 커넥톰 배선을 고정하고 세포 유형별 값과 출력층을 역전파로 학습하는 재순위 모델.

입력은 rerank.py의 _ix 버전과 같다: 비교 신호 [q*a, |q-a|](학습 쌍 기준 z-score) -> 고정 무작위 투영 ->
후각 뉴런(q*a), 시각 뉴런(|q-a|). T=8스텝 동안 일정하게 준다.
학습: 세포 유형별 누설률·휴지 상태(약 2.9만), 전체 이득, 출력층(하행+운동 뉴런 2,129 -> 점수)
손실: 역클로즈 질문마다 정답 1 + 오답 7 중 정답을 고르는 softmax 교차 엔트로피
검증 질문(역클로즈) MRR로 epoch과 λ(dense와 합칠 비율)를 고르고, 평가 질문에 적용한다.
사용 예:
  python -u regqa/stage2_train.py --variant connectome
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flycns.connectome import build_matrix, groups, load_neurons  # noqa: E402
from regqa.metrics import fmt, score_one, summarize  # noqa: E402
from regqa.rerank import K_EVAL, K_VAL, LAMBDAS, build_pairs, feats_lr_pair  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "stage2"


class FlyReranker(torch.nn.Module):
    def __init__(self, W, win, type_idx, n_types, readout, leak0=0.5, dev="mps"):
        super().__init__()
        W = W.tocoo()
        self.A = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([W.row, W.col]).astype(np.int64)),
                                         torch.from_numpy(W.data.astype(np.float32)), W.shape).coalesce().to(dev)
        win = win.tocoo()
        self.Win = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([win.row, win.col]).astype(np.int64)),
                                           torch.from_numpy(win.data.astype(np.float32)), win.shape).coalesce().to(dev)
        self.n = W.shape[0]
        self.register_buffer("type_idx", torch.from_numpy(type_idx.astype(np.int64)))
        self.leak_logit = torch.nn.Parameter(torch.full((n_types,), float(np.log(leak0 / (1 - leak0)))))
        self.bias = torch.nn.Parameter(torch.zeros(n_types))
        self.gain = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("readout", torch.from_numpy(readout.astype(np.int64)))
        self.head = torch.nn.Linear(len(readout), 1)
        torch.nn.init.zeros_(self.head.weight)

    def forward(self, U, steps=8):
        """U: (2048, B) z-score된 비교 신호 -> (B,) 점수"""
        a = torch.sigmoid(self.leak_logit)[self.type_idx].unsqueeze(1)
        b = self.bias[self.type_idx].unsqueeze(1)
        inp = torch.sparse.mm(self.Win, U)
        x = torch.zeros(self.n, U.shape[1], device=U.device)
        acc = 0
        for t in range(steps):
            x = (1 - a) * x + a * torch.tanh(self.gain * torch.sparse.mm(self.A, x) + inp + b)
            if t >= steps - 2:
                acc = acc + x[self.readout]
        return self.head((acc / 2).T).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="connectome")
    ap.add_argument("--rho", type=float, default=1.2)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--qbatch", type=int, default=8, help="배치당 질문 수(질문마다 후보 8개)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_batches", type=int, default=0, help="점검용: epoch당 배치 수 제한")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tag = f"stage2_{a.variant}_rho{a.rho}_s{a.seed}" + ("_smoke" if a.max_batches else "")
    RES.mkdir(parents=True, exist_ok=True)

    P, sets = build_pairs()
    tr = feats_lr_pair(sets["train"]["q"], sets["train"]["a"])
    mu, sd = tr.mean(0), tr.std(0) + 1e-8
    F = {k: ((feats_lr_pair(v["q"], v["a"]) - mu) / sd).astype(np.float32) for k, v in sets.items()}

    neurons = load_neurons()
    g = groups(neurons)
    keys, type_idx = np.unique(neurons["type_key"].values, return_inverse=True)
    W, meta = build_matrix(a.variant, rho=a.rho, seed=a.seed)
    # rerank.py _ix와 같은 고정 입력 투영
    n, d = W.shape[0], 1024
    rng = np.random.default_rng(2000 + a.seed)
    olf, vis = g["olfactory"], g["visual"]
    rows = np.concatenate([np.repeat(olf, d), np.repeat(vis, d)])
    cols = np.concatenate([np.tile(np.arange(d), len(olf)), np.tile(np.arange(d, 2 * d), len(vis))])
    win = sp.csr_matrix(((rng.standard_normal(len(rows)) / np.sqrt(d)).astype(np.float32), (rows, cols)), shape=(n, 2 * d))
    model = FlyReranker(W, win, type_idx, len(keys), g["readout"], dev=dev).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] trainable params {n_params:,} on {dev}", flush=True)

    Xtr = torch.from_numpy(F["train"])          # 질문마다 8쌍(정답이 첫 번째)
    nq = len(Xtr) // 8
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    def scores(split, bs=256):
        model.eval()
        out = []
        with torch.no_grad():
            X = torch.from_numpy(F[split])
            for i in range(0, len(X), bs):
                out.append(model(X[i:i + bs].T.contiguous().to(dev)).cpu().numpy())
        model.train()
        return np.concatenate(out)

    def val_mrr(S, lam, comb):
        rr = []
        for r, m, s in zip(P["val"], sets["val"]["meta"], S.reshape(-1, K_VAL)):
            sc = np.array(m["dense"]) + lam * s if comb else s
            order = [m["cands"][i] for i in np.argsort(-sc)]
            rr.append(1.0 / (order.index(r["pos"]) + 1) if r["pos"] in order else 0.0)
        return float(np.mean(rr))

    t0 = time.time()
    best, best_state, hist = -1, None, []
    for ep in range(a.epochs):
        perm = torch.randperm(nq)
        losses = []
        nb = int(np.ceil(nq / a.qbatch))
        for bi, i in enumerate(range(0, nq, a.qbatch)):
            if a.max_batches and bi >= a.max_batches:
                break
            qs = perm[i:i + a.qbatch]
            idx = (qs[:, None] * 8 + torch.arange(8)[None, :]).reshape(-1)
            s = model(Xtr[idx].T.contiguous().to(dev)).reshape(-1, 8)
            loss = torch.nn.functional.cross_entropy(s, torch.zeros(len(qs), dtype=torch.long, device=dev))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (bi + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  ep {ep + 1} batch {bi + 1}/{nb} loss {np.mean(losses[-20:]):.3f} ({el:.0f}s, "
                      f"ETA this epoch {el / (ep * nb + bi + 1) * (nb - bi - 1):.0f}s)", flush=True)
        S = scores("val")
        sdv = S.std() + 1e-9
        v_only = val_mrr(S, 0, False)
        v_comb, lam = max((val_mrr(S / sdv, l, True), l) for l in LAMBDAS)
        hist.append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_mrr_only": v_only, "val_mrr_comb": v_comb,
                     "lambda": lam, "gain": float(model.gain.detach()), "seconds": round(time.time() - t0)})
        print(f"[{tag}] epoch {ep + 1}: loss {hist[-1]['loss']:.3f} val MRR only {v_only:.3f} comb {v_comb:.3f} "
              f"(λ={lam}) ({hist[-1]['seconds']}s)", flush=True)
        if v_only > best:
            best = v_only
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            best_lam, best_sd = lam, sdv
    model.load_state_dict(best_state)
    ids = P["corpus_ids"]
    qs = [q for q in P["eval"] if q["level"] != "L5b"]
    S_eval = scores("eval").reshape(-1, K_EVAL) / best_sd
    res = {"tag": tag, "variant": a.variant, "trainable_params": n_params, "history": hist, "lambda": best_lam}
    for mode in ["only", "comb"]:
        per = []
        for q, m, s in zip(P["eval"], sets["eval"]["meta"], S_eval):
            if q["level"] == "L5b":
                continue
            sc = np.array(m["dense"]) + best_lam * s if mode == "comb" else s
            per.append(score_one([ids[m["cands"][i]] for i in np.argsort(-sc)], q))
        res[mode] = summarize(per, qs)
        print(f"[{tag}] {mode}: {fmt(res[mode])}", flush=True)
    (RES / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    torch.save(best_state, RES / f"{tag}.pt")


if __name__ == "__main__":
    main()
