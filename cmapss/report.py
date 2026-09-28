"""C-MAPSS 실험 결과를 모아 표·그래프·부트스트랩 검정을 만든다.

출력: results/cmapss/summary.json, results/cmapss/figures/*.png
(보고서 본문 results/REPORT_cmapss.md 는 이 요약을 보고 따로 쓴다)
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmapss.dataset import load, nasa_score, rmse  # noqa: E402
from cmapss.readout import last_cycle  # noqa: E402
from flycns.plotstyle import GRID, INK, INK2, SERIES, SURFACE  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
FIG = RES / "figures"


def test_truth():
    _, te = load("FD001")
    return last_cycle(te.sort_values(["unit", "cycle"])["rul"].values, te.sort_values(["unit", "cycle"])["unit"].values)


def readouts():
    out = {}
    for f in glob.glob(str(RES / "readout" / "*.json")):
        r = json.load(open(f))
        out[r["key"]] = r
    return out


def baselines():
    out = {}
    for f in glob.glob(str(RES / "baselines" / "FD001_*.json")):
        r = json.load(open(f))
        out[r["model"]] = r
    return out


def m2s():
    """키: connectome, random(처음부터 학습), connectome_warm, random_warm(M1 창 버전에서 출발)"""
    out = {}
    for f in glob.glob(str(RES / "m2" / "*_s0*.json")):
        r = json.load(open(f))
        out[r["variant"] + ("_warm" if r["tag"].endswith("_warm") else "")] = r
    return out


def curve(rows, key="val_rmse"):
    ns = sorted({r["n_engines"] for r in rows})
    m = [float(np.mean([r[key] for r in rows if r["n_engines"] == n])) for n in ns]
    s = [float(np.std([r[key] for r in rows if r["n_engines"] == n])) for n in ns]
    return ns, m, s


def boot_diff(pa, pb, truth, n=5000, seed=0):
    """엔진 단위 쌍체 부트스트랩: RMSE(a) - RMSE(b)의 95% 구간."""
    rng = np.random.default_rng(seed)
    ea, eb = (np.asarray(pa) - truth) ** 2, (np.asarray(pb) - truth) ** 2
    idx = rng.integers(0, len(truth), (n, len(truth)))
    d = np.sqrt(ea[idx].mean(1)) - np.sqrt(eb[idx].mean(1))
    return float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def best_key(R, variant, seed=0):
    ks = [k for k in R if k.startswith(f"FD001_{variant}_glu-1_") and k.endswith(f"_mapsemantic_s{seed}")]
    return min(ks, key=lambda k: R[k]["cv_rmse"]) if ks else None


def line_chart(path, title, ylabel, series):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for (label, xs, ys, sd), col in zip(series, SERIES):
        ax.fill_between(xs, np.array(ys) - np.array(sd), np.array(ys) + np.array(sd), color=col, alpha=0.12, lw=0)
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    # 오른쪽 끝 이름표: 겹치지 않게 최소 간격을 둔다
    lo = min(min(np.array(s[2]) - np.array(s[3])) for s in series)
    hi = max(max(np.array(s[2]) + np.array(s[3])) for s in series)
    gap = (hi - lo) * 0.06
    ends = sorted([(s[2][-1], s[0], col) for s, col in zip(series, SERIES)])
    placed = []
    for y, label, col in ends:
        y_adj = max(y, placed[-1] + gap) if placed else y
        placed.append(y_adj)
        ax.annotate(label, (series[0][1][-1], y), xytext=(series[0][1][-1] + 2.5, y_adj), textcoords="data",
                    va="center", fontsize=8.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.8) if abs(y_adj - y) > gap * 0.3 else None)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("학습에 쓴 엔진 수", fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_xticks(series[0][1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
    ax.set_xlim(series[0][1][0] - 3, series[0][1][-1] + 22)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def bar_chart(path, title, labels, values, xlabel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(labels) + 1.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(labels))
    ax.barh(y, values, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=2)
    for yi, v in zip(y, values):
        ax.annotate(f"{v:+.2f}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(y, labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    lo, hi = min(values + [0]), max(values + [0])
    pad = (hi - lo) * 0.25 + 0.2
    ax.set_xlim(lo - pad, hi + pad)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main():
    R, B, M = readouts(), baselines(), m2s()
    truth = test_truth()
    S = {"phaseA": {}, "best": {}, "phaseB": {}, "baselines": {}, "m2": {}, "tests": {}}
    for k, r in sorted(R.items()):
        if k.endswith("_mapsemantic_s0") and "_glu-1_" in k:
            S["phaseA"][k] = {kk: r[kk] for kk in ["alpha", "cv_rmse", "train_rmse", "test_rmse", "test_nasa"]}
    for v in ["connectome", "random", "degree_shuffle", "weight_shuffle"]:
        k = best_key(R, v)
        if k:
            r = R[k]
            S["best"][v] = {"key": k, "cv_rmse": r["cv_rmse"], "test_rmse": r["test_rmse"], "test_nasa": r["test_nasa"],
                            "curve_val": curve(r["data_size"]), "curve_gap": curve(r["data_size"], "gap"),
                            "curve_test": curve(r["data_size"], "test_rmse")}
    # 시드 편차(무작위·섞은 망은 망과 입력 가중치, 커넥톰은 입력 가중치만 바뀐다)
    for v in S["best"]:
        base = S["best"][v]["key"][:-3]
        seeds = [R[f"{base}_s{s}"] for s in range(3) if f"{base}_s{s}" in R]
        S["best"][v]["seeds_test_rmse"] = [r["test_rmse"] for r in seeds]
        S["best"][v]["seeds_cv_rmse"] = [r["cv_rmse"] for r in seeds]
        S["best"][v]["seeds_test_nasa"] = [r["test_nasa"] for r in seeds]
        S["best"][v]["seeds_gap10"] = [float(np.mean([d["gap"] for d in r["data_size"] if d["n_engines"] == 10]))
                                      for r in seeds]
        S["best"][v]["seeds_val10"] = [float(np.mean([d["val_rmse"] for d in r["data_size"] if d["n_engines"] == 10]))
                                      for r in seeds]
    ck = S["best"].get("connectome", {}).get("key")
    if ck:
        stem = ck
        for k, r in R.items():
            if k != stem and (k.startswith(stem + "_") or k.replace("glu+1", "glu-1") == stem
                              or k.replace("maprandom", "mapsemantic") == stem):
                S["phaseB"][k] = {"cv_rmse": r["cv_rmse"], "test_rmse": r["test_rmse"], "test_nasa": r["test_nasa"],
                                  "d_cv": r["cv_rmse"] - R[stem]["cv_rmse"],
                                  "d_test": r["test_rmse"] - R[stem]["test_rmse"],
                                  "curve_val": curve(r["data_size"])}
    for name, r in B.items():
        full = r["full"]
        S["baselines"][name] = {"n_params": r["n_params"],
                                "test_rmse": [f["test_rmse"] for f in full], "test_nasa": [f["test_nasa"] for f in full],
                                "curve_val": curve(r["data_size"]), "curve_gap": curve(r["data_size"], "gap"),
                                "curve_test": curve(r["data_size"], "test_rmse")}
    for v, r in M.items():
        S["m2"][v] = {k: r[k] for k in ["trainable_params", "best_val_rmse", "train_rmse", "test_rmse", "test_nasa"]}
        S["m2"][v]["warm_init"] = r.get("warm_init")
        S["m2"][v]["epochs"] = r["epochs"]
        S["m2"][v]["top_changed_types"] = r["top_changed_types"][:10]
    # 쌍체 부트스트랩(시험 엔진 100대)
    pairs = []
    if ck and "random" in S["best"]:
        pairs.append(("M1 커넥톰 - M1 무작위", R[ck]["test_pred_last"], R[S["best"]["random"]["key"]]["test_pred_last"]))
    for v in ["degree_shuffle", "weight_shuffle"]:
        if ck and v in S["best"]:
            pairs.append((f"M1 커넥톰 - M1 {v}", R[ck]["test_pred_last"], R[S["best"][v]["key"]]["test_pred_last"]))
    for suf in ["", "_warm"]:
        if f"connectome{suf}" in M and f"random{suf}" in M:
            pairs.append((f"M2{suf} 커넥톰 - M2{suf} 무작위", M[f"connectome{suf}"]["test_pred_last"],
                          M[f"random{suf}"]["test_pred_last"]))
    for name, pa, pb in pairs:
        d, lo, hi = boot_diff(pa, pb, truth)
        S["tests"][name] = {"rmse_diff": d, "ci95": [lo, hi]}
    (RES / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))

    # 그래프
    lab = {"connectome": "커넥톰 저장소", "random": "무작위 망 저장소", "degree_shuffle": "배선 섞은 저장소", "lstm": "LSTM"}
    ser_val, ser_gap = [], []
    for v in ["connectome", "random", "degree_shuffle"]:
        if v in S["best"]:
            ser_val.append((lab[v], *S["best"][v]["curve_val"]))
            ser_gap.append((lab[v], *S["best"][v]["curve_gap"]))
    if "lstm" in S["baselines"]:
        ser_val.append((lab["lstm"], *S["baselines"]["lstm"]["curve_val"]))
        ser_gap.append((lab["lstm"], *S["baselines"]["lstm"]["curve_gap"]))
    if ser_val:
        line_chart(FIG / "data_size_val_rmse.png", "학습 엔진 수에 따른 검증 RMSE (낮을수록 좋음)", "검증 RMSE (엔진 20대, 모든 사이클)", ser_val)
        line_chart(FIG / "data_size_gap.png", "학습 엔진 수에 따른 과적합 격차 (검증 RMSE - 학습 RMSE)", "과적합 격차", ser_gap)
    abl = [(k.split("_abl")[1], v["d_cv"]) for k, v in S["phaseB"].items() if "_abl" in k]
    if abl:
        names = {"optic_lobe": "시각엽 제거", "mushroom_body": "버섯체 제거", "central_complex": "중심복합체 제거",
                 "vnc": "신경삭 제거", "neck": "목 절단(하행·상행 뉴런 제거)"}
        bar_chart(FIG / "ablation_cv.png", "부위 절제에 따른 교차검증 RMSE 변화 (양수 = 나빠짐)",
                  [names.get(a, a) for a, _ in abl], [d for _, d in abl], "교차검증 RMSE 변화")
    print(json.dumps({k: v for k, v in S.items() if k in ("best", "tests")}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
