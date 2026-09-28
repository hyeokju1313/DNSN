"""항공엔진 잔여수명 시연. 시험 엔진 하나의 센서 기록을 사이클마다 뇌 전체에 흘려보내며 잔여수명을 예측한다.

1. 학습 엔진 100대의 저장소 상태(저장본)로 ridge 출력층을 맞춘다(alpha는 교차검증으로 고른 값)
2. 시험 엔진의 센서 17개를 사이클마다 감각 뉴런에 넣고, 뉴런 165,122개를 한 스텝씩 갱신한다
3. 하행·운동 뉴런 2,129개 상태로 그 사이클의 잔여수명을 읽는다
4. 새로 돌린 뉴런 상태를 실험 때 저장한 상태와 비교해 같은 모델인지 확인한다
사용 예:
  python -u cmapss/simulate.py --engines 34 49 93
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
from cmapss.dataset import DER, load, normalize, sequences  # noqa: E402
from cmapss.readout import Ridge  # noqa: E402
from flycns.connectome import DERIVED, groups, load_neurons  # noqa: E402
from flycns.plotstyle import GRID, INK, INK2, SERIES, SURFACE  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
# 망마다 교차검증으로 고른 설정(시드 0)
NETS = {
    "커넥톰": "FD001_connectome_glu-1_rho1.2_leak0.1_mapsemantic_s0",
    "무작위 망": "FD001_random_glu-1_rho0.9_leak0.1_mapsemantic_s0",
    "배선 섞은 망": "FD001_degree_shuffle_glu-1_rho1.2_leak0.5_mapsemantic_s0",
}


def run_engine(W, Win, xs, leak, readout, log=None):
    """엔진 하나를 사이클마다 한 스텝씩 돌리고 readout 뉴런 상태를 돌려준다."""
    x = np.zeros(W.shape[0], dtype=np.float32)
    out = np.zeros((len(xs), len(readout)), dtype=np.float32)
    t0 = time.time()
    for t, u in enumerate(xs):
        x = (1.0 - leak) * x + leak * np.tanh(W @ x + Win @ u)
        out[t] = x[readout]
        if log and ((t + 1) % 50 == 0 or t + 1 == len(xs)):
            el = time.time() - t0
            log(f"    사이클 {t + 1}/{len(xs)}, {el:.0f}s 경과, 남은 시간 약 {el / (t + 1) * (len(xs) - t - 1):.0f}s")
    return out


def simulate(name, key, engines, xs_te, u_te):
    info = json.loads((RES / "states" / f"{key}.json").read_text())
    z = np.load(RES / "states" / f"{key}.npz")
    alpha = json.loads((RES / "readout" / f"{key}.json").read_text())["alpha"]
    m = Ridge(alpha).fit(z["X_tr"], z["y_tr"])
    W = sp.load_npz(DERIVED / "matrices" / Path(info["matrix"]["path"]).name).tocsr().astype(np.float32)
    Win = sp.load_npz(DER / "inputs" / f"win_{key}.npz").tocsr().astype(np.float32)
    readout = groups(load_neurons())["readout"]
    print(f"[{name}] 연결 {W.nnz:,}개, 누설률 {info['args']['leak']}, ridge alpha {alpha}", flush=True)
    res = {}
    for e in engines:
        i = int(np.where(u_te == e)[0][0])
        print(f"  엔진 {e}: {len(xs_te[i])}사이클", flush=True)
        st = run_engine(W, Win, xs_te[i], info["args"]["leak"], readout, log=lambda s: print(s, flush=True))
        saved = z["X_te"][z["unit_te"] == e]
        diff = float(np.abs(st - saved).max())
        pred = m.predict(st)
        print(f"    저장된 상태와 최대 차이 {diff:.2e}, 마지막 사이클 예측 {pred[-1]:.1f}", flush=True)
        res[e] = {"pred": pred.tolist(), "max_state_diff": diff}
    return res


def plot(path, engines, ys_te, u_te, sims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, len(engines), figsize=(4.0 * len(engines), 3.9), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, e in zip(np.atleast_1d(axes), engines):
        ax.set_facecolor(SURFACE)
        y = ys_te[int(np.where(u_te == e)[0][0])]
        t = np.arange(1, len(y) + 1)
        ax.plot(t, y, color=INK, lw=1.6, ls=(0, (4, 3)), label="실제 잔여수명(125에서 자름)")
        for (name, r), col in zip(sims.items(), SERIES):
            ax.plot(t, r[e]["pred"], color=col, lw=1.4, label=name)
        # 시험 점수는 마지막 사이클만 본다
        ax.axvline(len(y), color=GRID, lw=1)
        lines = [f"실제 {y[-1]:.0f}"] + [f"{n} {r[e]['pred'][-1]:.0f}" for n, r in sims.items()]
        ax.text(0.03, 0.04, "마지막 사이클\n" + "\n".join(lines), transform=ax.transAxes,
                fontsize=7.5, color=INK2, va="bottom")
        ax.set_title(f"시험 엔진 {e}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("사이클", fontsize=8.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    np.atleast_1d(axes)[0].set_ylabel("잔여수명(사이클)", fontsize=8.5, color=INK2)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8.5,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", type=int, nargs="+", default=[34, 49, 93])
    a = ap.parse_args()
    tr, te = load("FD001")
    _, ten = normalize(tr, te)
    xs_te, ys_te, u_te = sequences(ten)
    sims = {name: simulate(name, key, a.engines, xs_te, u_te) for name, key in NETS.items()}

    print("\n엔진별 마지막 사이클 예측(시험 점수에 쓰는 값)")
    print("| 엔진 | 사이클 | 실제 | " + " | ".join(NETS) + " |")
    print("|---|---|---|" + "---|" * len(NETS))
    for e in a.engines:
        y = ys_te[int(np.where(u_te == e)[0][0])]
        print(f"| {e} | {len(y)} | {y[-1]:.0f} | " + " | ".join(f"{sims[n][e]['pred'][-1]:.1f}" for n in NETS) + " |")

    tag = "_".join(map(str, a.engines))
    out = RES / "simulate" / f"engines_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"nets": NETS, "engines": a.engines,
                               "truth": {e: ys_te[int(np.where(u_te == e)[0][0])].tolist() for e in a.engines},
                               "sims": sims}, ensure_ascii=False))
    fig = RES / "figures" / f"simulate_engines_{tag}.png"
    plot(fig, a.engines, ys_te, u_te, sims)
    print(f"\n저장: {out.relative_to(RES.parents[1])}, {fig.relative_to(RES.parents[1])}")


if __name__ == "__main__":
    main()
