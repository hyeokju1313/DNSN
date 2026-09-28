"""C-MAPSS 엔진 시계열을 커넥톰 저장소에 넣어 readout 뉴런 상태를 계산하고 저장한다.

사용 예:
  python -u cmapss/reservoir_states.py --variant connectome --rho 1.2 --leak 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmapss.dataset import CHANNEL_GROUP, DER, SENSORS, SETTINGS, load, normalize, sequences  # noqa: E402
from flycns.connectome import build_matrix, groups, load_neurons, top_connected  # noqa: E402
from flycns.reservoir import run_sequences  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "states"


def config_key(a) -> str:
    k = f"{a.fd}_{a.variant}_glu{int(a.glu):+d}_rho{a.rho}_leak{a.leak}_map{a.mapping}_s{a.seed}"
    if a.ablate:
        k += "_abl" + "-".join(a.ablate)
    if a.top:
        k += f"_top{a.top}"
    return k


def build_input(n, a, g):
    """입력 채널(센서 14 + 운전 조건 3) -> 감각 뉴런. 가중치 U(-1, 1)."""
    rng = np.random.default_rng(1000 + a.seed)
    chans = SENSORS + SETTINGS
    rows, cols = [], []
    if a.top:
        # 부분 커넥톰: 남긴 뉴런 중 20%를 무작위로 골라 채널을 고르게 배정(Costi 등과 같은 방식)
        sub = top_connected(a.top)
        inp = rng.choice(sub, max(len(chans), len(sub) // 5), replace=False)
        rows, cols = list(inp), list(rng.integers(0, len(chans), len(inp)))
    else:
        grp_of = dict(CHANNEL_GROUP)
        if a.mapping == "random":
            names = [grp_of[c] for c in chans]
            rng.shuffle(names)
            grp_of = dict(zip(chans, names))
        # 같은 감각 뉴런 집단을 쓰는 채널끼리는 집단을 나눠 갖는다
        by_group: dict[str, list[int]] = {}
        for ci, c in enumerate(chans):
            by_group.setdefault(grp_of[c], []).append(ci)
        for gname, cis in by_group.items():
            neurons = rng.permutation(g[gname])
            for j, part in enumerate(np.array_split(neurons, len(cis))):
                rows += list(part)
                cols += [cis[j]] * len(part)
    vals = rng.uniform(-1, 1, len(rows)).astype(np.float32)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(chans)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fd", default="FD001")
    ap.add_argument("--variant", default="connectome")
    ap.add_argument("--glu", type=float, default=-1.0)
    ap.add_argument("--rho", type=float, default=1.2)
    ap.add_argument("--leak", type=float, default=0.1)
    ap.add_argument("--mapping", default="semantic", choices=["semantic", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablate", nargs="*", default=[])
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    key = config_key(a)
    out = OUT / f"{key}.npz"
    if out.exists():
        print(f"[skip] {out.name} exists", flush=True)
        return
    t0 = time.time()
    neurons = load_neurons()
    g = groups(neurons)
    subset = top_connected(a.top) if a.top else None
    W, meta = build_matrix(a.variant, glu_sign=a.glu, rho=a.rho, seed=a.seed, ablate=tuple(a.ablate), subset=subset)
    readout = subset if a.top else g["readout"]
    Win = build_input(W.shape[0], a, g)
    win_path = DER / "inputs" / f"win_{key}.npz"
    win_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(win_path, Win)
    print(f"[{key}] matrix nnz={meta['nnz']} raw_sr={meta['raw_spectral_radius']:.3f}, readout={len(readout)}, "
          f"input neurons={Win.nnz} ({time.time() - t0:.0f}s)", flush=True)

    tr, te = load(a.fd)
    trn, ten = normalize(tr, te)
    xs_tr, ys_tr, u_tr = sequences(trn)
    xs_te, ys_te, u_te = sequences(ten)
    log = lambda s: print(s, flush=True)  # noqa: E731
    print(f"[{key}] train {len(xs_tr)} engines (max {max(map(len, xs_tr))} cycles)", flush=True)
    st_tr = run_sequences(meta["path"], win_path, xs_tr, a.leak, readout, workers=a.workers, log=log)
    print(f"[{key}] test {len(xs_te)} engines (max {max(map(len, xs_te))} cycles)", flush=True)
    st_te = run_sequences(meta["path"], win_path, xs_te, a.leak, readout, workers=a.workers, log=log)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_tr=np.concatenate(st_tr).astype(np.float32), y_tr=np.concatenate(ys_tr),
        unit_tr=np.concatenate([[u] * len(y) for u, y in zip(u_tr, ys_tr)]),
        X_te=np.concatenate(st_te).astype(np.float32), y_te=np.concatenate(ys_te),
        unit_te=np.concatenate([[u] * len(y) for u, y in zip(u_te, ys_te)]),
    )
    info = {"key": key, "args": vars(a), "matrix": meta, "readout_neurons": int(len(readout)),
            "seconds": round(time.time() - t0, 1)}
    out.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"[{key}] saved {out.name} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
