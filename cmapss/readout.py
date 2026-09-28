"""저장소 상태 -> 잔여수명 ridge 출력층. 엔진 단위 교차검증으로 alpha를 고른다.

사용 예:
  python -u cmapss/readout.py FD001_connectome_glu-1_rho1.2_leak0.1_mapsemantic_s0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmapss.dataset import engine_folds, nasa_score, rmse  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
DATA_SIZES = [10, 25, 50, 80]
REPEATS = 5


class Ridge:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, X, y):
        X = X.astype(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.mu) / self.sd
        self.b = y.mean()
        A = Z.T @ Z + self.alpha * len(Z) / 1000.0 * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - self.b))
        return self

    def predict(self, X):
        return np.clip(((X.astype(np.float64) - self.mu) / self.sd) @ self.w + self.b, 0, None)


def last_cycle(pred, unit):
    """엔진마다 마지막 행의 값."""
    idx = np.r_[np.where(np.diff(unit) != 0)[0], len(unit) - 1]
    return pred[idx]


def choose_alpha(X, y, unit, k=5, seed=0):
    units = np.unique(unit)
    folds = engine_folds(units, k=min(k, len(units)), seed=seed)
    best = None
    for a in ALPHAS:
        errs = []
        for f in folds:
            va = np.isin(unit, f)
            m = Ridge(a).fit(X[~va], y[~va])
            errs.append(np.mean((m.predict(X[va]) - y[va]) ** 2))
        score = float(np.sqrt(np.mean(errs)))
        if best is None or score < best[1]:
            best = (a, score)
    return best


def evaluate(X_tr, y_tr, unit_tr, X_te, y_te, unit_te, seed=0):
    alpha, cv = choose_alpha(X_tr, y_tr, unit_tr, seed=seed)
    m = Ridge(alpha).fit(X_tr, y_tr)
    p_te = last_cycle(m.predict(X_te), unit_te)
    t_te = last_cycle(y_te, unit_te)
    return {
        "alpha": alpha, "cv_rmse": cv, "train_rmse": rmse(m.predict(X_tr), y_tr),
        "test_rmse": rmse(p_te, t_te), "test_nasa": nasa_score(p_te, t_te),
    }, m, p_te


def data_size_curve(X_tr, y_tr, unit_tr, X_te, y_te, unit_te):
    """검증 엔진 20대를 고정하고, 나머지 80대에서 n대를 뽑아 학습한다."""
    units = np.unique(unit_tr)
    rng = np.random.default_rng(12345)
    val_units = rng.choice(units, 20, replace=False)
    pool = np.setdiff1d(units, val_units)
    va = np.isin(unit_tr, val_units)
    t_te = last_cycle(y_te, unit_te)
    rows = []
    for n in DATA_SIZES:
        for r in range(REPEATS if n < len(pool) else 1):
            pick = np.random.default_rng(1000 * n + r).choice(pool, n, replace=False)
            tr = np.isin(unit_tr, pick)
            alpha, _ = choose_alpha(X_tr[tr], y_tr[tr], unit_tr[tr], seed=r)
            m = Ridge(alpha).fit(X_tr[tr], y_tr[tr])
            tr_rmse = rmse(m.predict(X_tr[tr]), y_tr[tr])
            va_rmse = rmse(m.predict(X_tr[va]), y_tr[va])
            te_rmse = rmse(last_cycle(m.predict(X_te), unit_te), t_te)
            rows.append({"n_engines": n, "repeat": r, "alpha": alpha, "train_rmse": tr_rmse,
                         "val_rmse": va_rmse, "gap": va_rmse - tr_rmse, "test_rmse": te_rmse})
    return rows


def main(key):
    z = np.load(RES / "states" / f"{key}.npz")
    X_tr, y_tr, unit_tr = z["X_tr"], z["y_tr"], z["unit_tr"]
    X_te, y_te, unit_te = z["X_te"], z["y_te"], z["unit_te"]
    res, _, p_te = evaluate(X_tr, y_tr, unit_tr, X_te, y_te, unit_te)
    res["data_size"] = data_size_curve(X_tr, y_tr, unit_tr, X_te, y_te, unit_te)
    res["key"] = key
    res["test_pred_last"] = [float(v) for v in p_te]
    out = RES / "readout" / f"{key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    ds = {}
    for r in res["data_size"]:
        ds.setdefault(r["n_engines"], []).append((r["val_rmse"], r["gap"], r["test_rmse"]))
    summ = " | ".join(f"n={n}: val {np.mean([v[0] for v in vs]):.2f} gap {np.mean([v[1] for v in vs]):.2f}"
                      for n, vs in ds.items())
    print(f"{key}: alpha={res['alpha']} cv={res['cv_rmse']:.2f} train={res['train_rmse']:.2f} "
          f"TEST rmse={res['test_rmse']:.2f} nasa={res['test_nasa']:.0f} || {summ}", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        main(k)
