"""C-MAPSS 기준 모델: 선형 회귀 2종, LSTM, GRU, 1D-CNN.

저장소와 같은 규약으로 평가한다.
- 매 사이클을 표본으로 쓴다(최근 30사이클 창, 앞부분은 첫 행으로 채움)
- 시험은 각 시험 엔진의 마지막 사이클
- 데이터 양 실험은 readout.py와 같은 검증 엔진 20대와 같은 뽑기 시드를 쓴다
사용 예:
  python -u cmapss/baselines.py --models linear_now linear_window lstm gru cnn
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmapss.dataset import load, nasa_score, normalize, rmse, sequences  # noqa: E402
from cmapss.readout import DATA_SIZES, REPEATS, Ridge, choose_alpha, last_cycle  # noqa: E402

RES = Path(__file__).resolve().parent / "results" / "baselines"
WIN = 30
torch.set_num_threads(4)


def windows(xs, ys, units):
    X, Y, U = [], [], []
    for x, y, u in zip(xs, ys, units):
        pad = np.concatenate([np.repeat(x[:1], WIN - 1, 0), x])
        idx = np.arange(len(x))[:, None] + np.arange(WIN)[None, :]
        X.append(pad[idx])
        Y.append(y)
        U.append(np.full(len(y), u))
    return np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(U)


class RNN(nn.Module):
    def __init__(self, cell, d_in=17, h=64):
        super().__init__()
        self.rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell](d_in, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(o[:, -1]).squeeze(-1)


class CNN(nn.Module):
    def __init__(self, d_in=17):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_in, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(-1)


def make(name):
    return RNN("lstm") if name == "lstm" else RNN("gru") if name == "gru" else CNN()


def train_nn(name, X, Y, U, seed, max_epochs=80, patience=10):
    """고른 엔진의 20%(최소 2대)를 조기 종료용으로 떼어 두고 학습한다."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    units = np.unique(U)
    n_val = max(2, int(round(len(units) * 0.2)))
    vu = rng.choice(units, n_val, replace=False)
    va = np.isin(U, vu)
    Xt, Yt = torch.from_numpy(X[~va]), torch.from_numpy(Y[~va])
    Xv, Yv = torch.from_numpy(X[va]), torch.from_numpy(Y[va])
    model = make(name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state, bad = 1e18, None, 0
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = ((model(Xt[b]) - Yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(Xv) - Yv) ** 2).mean())
        if v < best - 1e-3:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1


def predict(model, X):
    with torch.no_grad():
        return np.clip(np.concatenate([model(torch.from_numpy(X[i:i + 4096])).numpy()
                                       for i in range(0, len(X), 4096)]), 0, None)


def feats(name, Xw):
    if name == "linear_now":
        return Xw[:, -1, :]
    if name == "linear_window":
        return Xw.reshape(len(Xw), -1)
    return Xw


def fit_predict(name, Xw, Y, U, Xw_eval_list, seed):
    if name.startswith("linear"):
        F = feats(name, Xw)
        alpha, _ = choose_alpha(F, Y, U, seed=seed)
        m = Ridge(alpha).fit(F, Y)
        n_params = F.shape[1] + 1
        return [m.predict(F)] + [m.predict(feats(name, Xe)) for Xe in Xw_eval_list], n_params, {"alpha": alpha}
    model, epochs = train_nn(name, Xw, Y, U, seed)
    n_params = sum(p.numel() for p in model.parameters())
    return [predict(model, Xw)] + [predict(model, Xe) for Xe in Xw_eval_list], n_params, {"epochs": epochs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--models", nargs="+", default=["linear_now", "linear_window", "lstm", "gru", "cnn"])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    tr, te = load(a.fd)
    trn, ten = normalize(tr, te)
    Xw, Y, U = windows(*sequences(trn))
    Xe, Ye, Ue = windows(*sequences(ten))
    t_te = last_cycle(Ye, Ue)
    RES.mkdir(parents=True, exist_ok=True)
    units = np.unique(U)
    val_units = np.random.default_rng(12345).choice(units, 20, replace=False)
    pool = np.setdiff1d(units, val_units)
    va = np.isin(U, val_units)

    for name in a.models:
        t0 = time.time()
        full = []
        for s in range(a.seeds if not name.startswith("linear") else 1):
            (p_tr, p_te), n_params, extra = fit_predict(name, Xw, Y, U, [Xe], seed=s)
            p_last = last_cycle(p_te, Ue)
            full.append({"seed": s, "train_rmse": rmse(p_tr, Y), "test_rmse": rmse(p_last, t_te),
                         "test_nasa": nasa_score(p_last, t_te), **extra})
            print(f"[{name}] full seed {s}: test rmse {full[-1]['test_rmse']:.2f} nasa {full[-1]['test_nasa']:.0f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        curve = []
        for n in DATA_SIZES:
            for r in range(REPEATS if n < len(pool) else 1):
                pick = np.random.default_rng(1000 * n + r).choice(pool, n, replace=False)
                m = np.isin(U, pick)
                (p_tr, p_va, p_te), _, extra = fit_predict(name, Xw[m], Y[m], U[m], [Xw[va], Xe], seed=r)
                curve.append({"n_engines": n, "repeat": r, "train_rmse": rmse(p_tr, Y[m]),
                              "val_rmse": rmse(p_va, Y[va]), "gap": rmse(p_va, Y[va]) - rmse(p_tr, Y[m]),
                              "test_rmse": rmse(last_cycle(p_te, Ue), t_te), **extra})
            vs = [c for c in curve if c["n_engines"] == n]
            print(f"[{name}] n={n}: val {np.mean([c['val_rmse'] for c in vs]):.2f} gap {np.mean([c['gap'] for c in vs]):.2f} "
                  f"test {np.mean([c['test_rmse'] for c in vs]):.2f} ({time.time() - t0:.0f}s)", flush=True)
        out = {"model": name, "n_params": n_params, "full": full, "data_size": curve}
        (RES / f"{a.fd}_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
