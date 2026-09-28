"""NASA C-MAPSS 불러오기와 전처리. 평가 규약은 proposal_2 4.3절을 따른다."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data"
DER = ROOT / "data" / "derived"
COLS = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
# FD001에서 값이 변하지 않거나 거의 변하지 않는 센서(s1, s5, s6, s10, s16, s18, s19)를 뺀 14개
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
SETTINGS = ["os1", "os2", "os3"]
RUL_CAP = 125

# proposal_2 5.2절의 의미 대응(팀 설계): 입력 채널 -> 감각 뉴런 집단
CHANNEL_GROUP = {
    "s2": "thermo", "s3": "thermo", "s4": "thermo",
    "s8": "vnc_proprio", "s9": "vnc_proprio", "s13": "vnc_proprio", "s14": "vnc_proprio",
    "s7": "vnc_tactile", "s11": "vnc_tactile",
    "s12": "head_mechano", "s15": "head_mechano", "s17": "head_mechano", "s20": "head_mechano", "s21": "head_mechano",
    "os1": "visual", "os2": "visual", "os3": "visual",
}


def load(fd: str = "FD001"):
    tr = pd.read_csv(RAW / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(RAW / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_te = pd.read_csv(RAW / f"RUL_{fd}.txt", header=None).iloc[:, 0].values
    life = tr.groupby("unit").cycle.transform("max")
    tr["rul"] = np.minimum(life - tr.cycle, RUL_CAP)
    last = te.groupby("unit").cycle.transform("max")
    te["rul"] = np.minimum(rul_te[te.unit.values - 1] + (last - te.cycle), RUL_CAP)
    return tr, te


def normalize(tr: pd.DataFrame, te: pd.DataFrame, train_units=None):
    """z-score. 통계는 학습 엔진에서만 계산한다(누수 방지)."""
    feats = SENSORS + SETTINGS
    ref = tr if train_units is None else tr[tr.unit.isin(train_units)]
    mu, sd = ref[feats].mean(), ref[feats].std().replace(0, 1.0)
    trn, ten = tr.copy(), te.copy()
    trn[feats] = (tr[feats] - mu) / sd
    ten[feats] = (te[feats] - mu) / sd
    return trn, ten


def sequences(df: pd.DataFrame):
    """엔진별 (입력 시계열 [T, 17], 목표 [T]) 목록. 엔진 번호 순서."""
    feats = SENSORS + SETTINGS
    xs, ys, units = [], [], []
    for u, d in df.groupby("unit", sort=True):
        d = d.sort_values("cycle")
        xs.append(d[feats].values.astype(np.float32))
        ys.append(d["rul"].values.astype(np.float32))
        units.append(u)
    return xs, ys, np.array(units)


def nasa_score(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def engine_folds(units, k=5, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.permutation(units)
    return [u[i::k] for i in range(k)]
