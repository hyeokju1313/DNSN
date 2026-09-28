"""학사규정 챗봇 실험 결과를 모아 요약 JSON과 그래프를 만든다.

출력: results/regqa/summary.json, results/regqa/figures/r1_by_level.png
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flycns.plotstyle import GRID, INK, INK2, SERIES, SURFACE  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
LEVELS = ["L1", "L2", "L3", "L4", "L5a", "ALL"]


def main():
    base = json.load(open(RES / "retrieval_baselines.json"))
    rr = {}
    for f in sorted(glob.glob(str(RES / "rerank_*.json"))):
        r = json.load(open(f))
        rr[r["feature"]] = r
    S = {"retrieval": base, "rerank": rr}
    (RES / "summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False
    systems = [("BM25(키워드)", base["bm25"]), ("dense(KURE-v1)", base["dense"])]
    for key, lab in [("lr_pair", "dense + 로지스틱 재순위"), ("res_connectome_ix_rho1.2_leak0.5_T8", "dense + 커넥톰 재순위")]:
        if key in rr:
            systems.append((lab, rr[key]["comb"]))
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(LEVELS))
    w = 0.8 / len(systems)
    for i, ((lab, res), col) in enumerate(zip(systems, SERIES)):
        vals = [res[lv]["hit1"] for lv in LEVELS]
        ax.bar(x + (i - (len(systems) - 1) / 2) * w, vals, width=w, color=col, edgecolor=SURFACE, linewidth=2, label=lab)
    ax.set_xticks(x, ["L1 단일", "L2 구어체", "L3 조건", "L4 여러 조문", "L5a 예외", "전체"], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R@1 (1위가 정답 조문인 비율)", fontsize=9, color=INK2)
    ax.set_title("질문 난이도별 R@1 (평가 질문 92개, Claude 작성)", loc="left", fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), labelcolor=INK2)
    fig.tight_layout()
    (RES / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RES / "figures" / "r1_by_level.png", facecolor=SURFACE)
    plt.close(fig)
    for k, r in rr.items():
        print(k, "only", round(r["only"]["ALL"]["hit1"], 3), "comb", round(r["comb"]["ALL"]["hit1"], 3),
              "abstain", r["comb"]["abstain"])


if __name__ == "__main__":
    main()
