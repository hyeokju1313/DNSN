"""M2: 수컷 초파리 전체 CNS 배선을 고정하고 세포 유형별 값을 역전파로 학습하는 잔여수명 모델.

flyvis(Lappalainen 등 2024)의 방식: 연결 구조, 시냅스 수, 부호는 고정. 학습하는 것은
- 세포 유형별 누설률 a_k(시그모이드), 휴지 상태 b_k       : 유형 11,751개 + 유형 없는 뉴런 2,605개
- 입력 가중치(센서 채널 -> 감각 뉴런, 연결 위치는 M1과 같음)
- 전체 이득 g, 출력층(하행 뉴런 + 운동 뉴런 2,129개 -> 잔여수명)
동역학: x(t+1) = (1 - a) * x(t) + a * tanh(g * W x(t) + Win u(t) + b)
기준 모델(baselines.py)과 같이 최근 30사이클 창을 0 상태에서 시작해 넣는다.
학습 엔진 80대, 조기 종료용 검증 엔진 20대(readout.py 데이터 양 실험과 같은 20대).
Apple GPU(MPS)에서 COO 희소 행렬로 돌린다.
사용 예:
  python -u cmapss/m2_train.py --variant connectome
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmapss.baselines import windows  # noqa: E402
from cmapss.dataset import RUL_CAP, load, nasa_score, normalize, rmse, sequences  # noqa: E402
from cmapss.readout import last_cycle  # noqa: E402
from cmapss.reservoir_states import build_input  # noqa: E402
from flycns.connectome import build_matrix, groups, load_neurons  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "m2"


class FlyCNS(torch.nn.Module):
    def __init__(self, W, win, type_idx, n_types, readout, leak0=0.1, dev="mps"):
        super().__init__()
        W = W.tocoo()
        self.A = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([W.row, W.col]).astype(np.int64)),
                                         torch.from_numpy(W.data.astype(np.float32)), W.shape).coalesce().to(dev)
        self.n = W.shape[0]
        win = win.tocoo()
        self.register_buffer("in_rows", torch.from_numpy(win.row.astype(np.int64)))
        self.register_buffer("in_cols", torch.from_numpy(win.col.astype(np.int64)))
        self.w_in = torch.nn.Parameter(torch.from_numpy(win.data.astype(np.float32)))
        self.register_buffer("type_idx", torch.from_numpy(type_idx.astype(np.int64)))
        self.leak_logit = torch.nn.Parameter(torch.full((n_types,), float(np.log(leak0 / (1 - leak0)))))
        self.bias = torch.nn.Parameter(torch.zeros(n_types))
        self.gain = torch.nn.Parameter(torch.ones(()))
        self.register_buffer("readout", torch.from_numpy(readout.astype(np.int64)))
        self.head = torch.nn.Linear(len(readout), 1)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.constant_(self.head.bias, 0.5)
        # 출력층 입력 표준화(warm-start에서 초기 상태의 평균·표준편차로 채운다). 기본값은 표준화 없음
        self.register_buffer("r_mu", torch.zeros(len(readout)))
        self.register_buffer("r_sd", torch.ones(len(readout)))

    def forward(self, u):
        """u: (B, T, C) -> 잔여수명/125 예측 (B,)"""
        return self.head((self.readout_state(u) - self.r_mu) / self.r_sd).squeeze(-1)

    def readout_state(self, u):
        """u: (B, T, C) -> 마지막 스텝의 하행+운동 뉴런 상태 (B, 2129)"""
        B, T, _ = u.shape
        a = torch.sigmoid(self.leak_logit)[self.type_idx].unsqueeze(1)       # (n, 1)
        b = self.bias[self.type_idx].unsqueeze(1)
        x = torch.zeros(self.n, B, device=u.device)
        for t in range(T):
            # 입력 뉴런(9,453개)은 채널 하나씩만 받는다
            pre = (self.gain * torch.sparse.mm(self.A, x) + b).index_add(
                0, self.in_rows, self.w_in.unsqueeze(1) * u[:, t, :].T[self.in_cols])
            x = (1 - a) * x + a * torch.tanh(pre)
        return x[self.readout].T                                             # (B, 2129)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="connectome")
    ap.add_argument("--rho", type=float, default=1.2)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--stride", type=int, default=5, help="학습 창을 몇 사이클 간격으로 뽑을지")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr_dyn", type=float, default=None,
                    help="동역학 파라미터(누설률·휴지 상태·이득) 학습률. 없으면 --lr와 같다. 입력 가중치는 10배, 출력층은 --lr")
    ap.add_argument("--warm", action="store_true",
                    help="M1과 같은 설정의 창 상태로 출력층을 ridge로 먼저 맞춘 뒤 학습(epoch 0 = M1 창 버전)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tag = f"FD001_m2_{a.variant}_rho{a.rho}_s{a.seed}" + ("_warm" if a.warm else "")
    RES.mkdir(parents=True, exist_ok=True)
    log_rows = []

    neurons = load_neurons()
    g = groups(neurons)
    keys, type_idx = np.unique(neurons["type_key"].values, return_inverse=True)
    W, meta = build_matrix(a.variant, rho=a.rho, seed=a.seed)

    class _A:  # reservoir_states.build_input 인자 흉내
        top = 0
        mapping = "semantic"
        seed = a.seed
    win = build_input(W.shape[0], _A, g)
    model = FlyCNS(W, win, type_idx, len(keys), g["readout"], dev=dev).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] trainable params {n_params:,} (types {len(keys):,}, input weights {win.nnz:,}, "
          f"readout {len(g['readout'])}) on {dev}", flush=True)

    tr, te = load("FD001")
    trn, ten = normalize(tr, te)
    Xw, Y, U = windows(*sequences(trn))
    Xe, Ye, Ue = windows(*sequences(ten))
    units = np.unique(U)
    val_units = np.random.default_rng(12345).choice(units, 20, replace=False)
    va = np.isin(U, val_units)
    # 학습 창은 stride 간격으로 뽑되 각 엔진의 마지막 사이클은 항상 넣는다
    cyc = np.concatenate([np.arange(len(y)) for y in sequences(trn)[1]])
    is_last = np.r_[np.diff(U) != 0, True]
    pick_tr = (~va) & ((cyc % a.stride == 0) | is_last)
    pick_va = va & ((cyc % a.stride == 0) | is_last)
    Xtr, Ytr = torch.from_numpy(Xw[pick_tr]), torch.from_numpy(Y[pick_tr] / RUL_CAP)
    Xva, Yva = torch.from_numpy(Xw[pick_va]), Y[pick_va]
    print(f"[{tag}] train windows {len(Xtr)}, val windows {len(Xva)}, test engines {len(np.unique(Ue))}", flush=True)

    if a.lr_dyn is None:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    else:
        # 뇌 상태의 변동 폭이 0.001 수준이라 동역학 파라미터를 같은 학습률로 바꾸면 출력이 폭발한다
        opt = torch.optim.Adam([
            {"params": [model.leak_logit, model.bias, model.gain], "lr": a.lr_dyn},
            {"params": [model.w_in], "lr": a.lr_dyn * 10},
            {"params": list(model.head.parameters()), "lr": a.lr},
        ])

    def predict(X, bs=128):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(model(torch.as_tensor(X[i:i + bs]).to(dev)).cpu().numpy())
        model.train()
        return np.clip(np.concatenate(out) * RUL_CAP, 0, None)

    best, best_state, bad = 1e9, None, 0
    t0 = time.time()
    init = None
    if a.warm:
        # 1) M1 설정(누설 0.1, 휴지 0, 이득 1) 그대로 창 상태를 모은다  2) ridge로 출력층을 맞춘다
        def states(X, bs=256):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), bs):
                    out.append(model.readout_state(torch.as_tensor(X[i:i + bs]).to(dev)).cpu().numpy())
            return np.concatenate(out).astype(np.float64)
        Rtr, Rva = states(Xtr.numpy()), states(Xva.numpy())
        mu_r, sd_r = Rtr.mean(0), Rtr.std(0) + 1e-12
        Ztr, Zva = (Rtr - mu_r) / sd_r, (Rva - mu_r) / sd_r
        ytr = Ytr.numpy().astype(np.float64)
        cand = []
        for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
            w = np.linalg.solve(Ztr.T @ Ztr + alpha * len(Ztr) / 1000.0 * np.eye(Ztr.shape[1]), Ztr.T @ (ytr - ytr.mean()))
            v = rmse(np.clip((Zva @ w + ytr.mean()) * RUL_CAP, 0, None), Yva)
            cand.append((v, alpha, w))
        v0, alpha0, w0 = min(cand, key=lambda c: c[0])
        with torch.no_grad():
            # 원래 척도로 옮기면 가중치가 1/sd(수천 배)로 커져 학습이 폭발한다. 표준화된 상태를 받게 한다
            model.r_mu.copy_(torch.from_numpy(mu_r.astype(np.float32)))
            model.r_sd.copy_(torch.from_numpy(sd_r.astype(np.float32)))
            model.head.weight.copy_(torch.from_numpy(w0[None, :].astype(np.float32)))
            model.head.bias.fill_(float(ytr.mean()))
        last_idx0 = np.r_[np.where(np.diff(Ue) != 0)[0], len(Ue) - 1]
        p0 = predict(Xe[last_idx0])
        t0_te = last_cycle(Ye, Ue)
        init = {"alpha": alpha0, "val_rmse": v0, "test_rmse": rmse(p0, t0_te), "test_nasa": nasa_score(p0, t0_te)}
        log_rows.append({"epoch": 0, "val_rmse": v0, "note": "M1 창 버전(ridge 초기화)"})
        best = v0
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"[{tag}] epoch 0 (ridge init, alpha={alpha0}): val {v0:.2f} test {init['test_rmse']:.2f} "
              f"nasa {init['test_nasa']:.0f} ({time.time() - t0:.0f}s)", flush=True)
    nb = int(np.ceil(len(Xtr) / a.batch))
    for ep in range(a.epochs):
        perm = torch.randperm(len(Xtr))
        tl = []
        for bi, i in enumerate(range(0, len(perm), a.batch)):
            idx = perm[i:i + a.batch]
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            opt.zero_grad()
            loss = torch.mean((model(xb) - yb) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
            if (bi + 1) % 10 == 0:
                el = time.time() - t0
                done = ep * nb + bi + 1
                print(f"  ep {ep + 1} batch {bi + 1}/{nb} loss {np.mean(tl[-10:]) * RUL_CAP ** 2:.1f} "
                      f"({el:.0f}s, ETA this epoch {el / done * (nb - bi - 1):.0f}s)", flush=True)
        pv = predict(Xva)
        v = rmse(pv, Yva)
        row = {"epoch": ep + 1, "train_rmse_batch": float(np.sqrt(np.mean(tl)) * RUL_CAP), "val_rmse": v,
               "gain": float(model.gain), "seconds": round(time.time() - t0)}
        log_rows.append(row)
        print(f"[{tag}] epoch {ep + 1}: train {row['train_rmse_batch']:.2f} val {v:.2f} gain {row['gain']:.3f} "
              f"({row['seconds']}s)", flush=True)
        if v < best - 0.01:
            best, bad = v, 0
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    # 시험: 각 시험 엔진의 마지막 사이클 창
    last_idx = np.r_[np.where(np.diff(Ue) != 0)[0], len(Ue) - 1]
    p_te = predict(Xe[last_idx])
    t_te = last_cycle(Ye, Ue)
    p_tr = predict(Xw[pick_tr])
    res = {"tag": tag, "variant": a.variant, "rho": a.rho, "trainable_params": n_params, "best_val_rmse": best,
           "warm_init": init,
           "train_rmse": rmse(p_tr, Y[pick_tr]), "test_rmse": rmse(p_te, t_te), "test_nasa": nasa_score(p_te, t_te),
           "epochs": log_rows, "test_pred_last": [float(v) for v in p_te], "args": vars(a)}
    # 학습 후 유형별 누설률 변화가 큰 세포 유형(해석용)
    leak = torch.sigmoid(best_state["leak_logit"]).numpy()
    order = np.argsort(-np.abs(leak - 0.1))[:30]
    res["top_changed_types"] = [{"type": str(keys[i]), "leak": float(leak[i]),
                                 "n_neurons": int((type_idx == i).sum())} for i in order]
    (RES / f"{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    torch.save(best_state, RES / f"{tag}.pt")
    print(f"[{tag}] TEST rmse {res['test_rmse']:.2f} nasa {res['test_nasa']:.0f} (val {best:.2f}, "
          f"train {res['train_rmse']:.2f}) total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
