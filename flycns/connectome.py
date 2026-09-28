"""수컷 초파리 CNS 커넥톰(male CNS v1.0)을 불러와 순환망 가중치 행렬로 만든다.

행렬 W는 (post, pre) 방향의 CSR이다. 즉 x_post = W @ x_pre.
가중치 = 시냅스 전 뉴런의 부호 × 시냅스 수, 그 뒤 spectral radius로 나눠 rho에 맞춘다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "male_cns_v1.0"
DERIVED = ROOT / "data" / "derived"

NT_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
VNC_SUPERCLASSES = ["vnc_intrinsic", "vnc_sensory", "vnc_motor", "vnc_efferent",
                    "vnc_endocrine", "vnc_sensory_tbc", "vnc_tbc"]


def load_neurons() -> pd.DataFrame:
    """추적이 끝난 뉴런 표. 행 순서가 곧 행렬 인덱스다."""
    cache = DERIVED / "neurons.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    a = pd.read_feather(RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt = pd.read_feather(RAW / "body-neurotransmitters-male-cns-v1.0.feather")[["body", "consensus_nt"]]
    tr = a[a.status == "Traced"][["bodyId", "type", "class", "superclass", "somaSide"]].copy()
    tr = tr.merge(nt, left_on="bodyId", right_on="body", how="left").drop(columns="body")
    tr = tr.sort_values("bodyId").reset_index(drop=True)
    # 세포 유형이 없는 뉴런은 뉴런 하나를 유형 하나로 본다
    tr["type_key"] = tr["type"].where(tr["type"].notna(), "untyped_" + tr["bodyId"].astype(str))
    DERIVED.mkdir(parents=True, exist_ok=True)
    tr.to_parquet(cache)
    return tr


def load_edges(neurons: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """추적 뉴런끼리의 연결: (pre 인덱스, post 인덱스, 시냅스 수)."""
    cache = DERIVED / "edges_traced.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pre"], z["post"], z["w"]
    idx = pd.Series(np.arange(len(neurons)), index=neurons.bodyId.values)
    w = pd.read_feather(RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
    w = w[w.body_pre.isin(idx.index) & w.body_post.isin(idx.index)]
    pre = idx[w.body_pre.values].values.astype(np.int32)
    post = idx[w.body_post.values].values.astype(np.int32)
    ww = w.weight.values.astype(np.float32)
    np.savez(cache, pre=pre, post=post, w=ww)
    return pre, post, ww


def neuron_signs(neurons: pd.DataFrame, glu_sign: float = -1.0) -> np.ndarray:
    m = dict(NT_SIGN)
    m["glutamate"] = glu_sign
    return neurons.consensus_nt.map(m).fillna(0.0).values.astype(np.float32)


def groups(neurons: pd.DataFrame) -> dict[str, np.ndarray]:
    """입력·출력·절제에 쓰는 뉴런 집합(행렬 인덱스)."""
    sc = neurons.superclass.fillna("")
    cl = neurons["class"].fillna("")
    head_sensory = sc.isin(["cb_sensory", "ol_sensory"])
    g = {
        "olfactory": np.where(head_sensory & (cl == "olfactory"))[0],
        "visual": np.where(head_sensory & (cl == "visual"))[0],
        "head_mechano": np.where(head_sensory & (cl == "mechanosensory"))[0],
        "thermo": np.where(head_sensory & (cl == "thermosensory"))[0],
        "vnc_proprio": np.where((sc == "vnc_sensory") & (cl == "mechanosensory_proprioceptive"))[0],
        "vnc_tactile": np.where((sc == "vnc_sensory") & (cl == "mechanosensory_tactile"))[0],
        "descending": np.where(sc == "descending_neuron")[0],
        "ascending": np.where(sc == "ascending_neuron")[0],
        "motor": np.where(sc.isin(["vnc_motor", "cb_motor"]))[0],
        "mbon": np.where(cl == "MBON")[0],
        # 절제 대상
        "optic_lobe": np.where(sc == "ol_intrinsic")[0],
        "mushroom_body": np.where(cl == "Kenyon_Cell")[0],
        "central_complex": np.where(cl == "CX")[0],
        "vnc": np.where(sc.isin(VNC_SUPERCLASSES))[0],
    }
    g["readout"] = np.concatenate([g["descending"], g["motor"]])
    g["neck"] = np.concatenate([g["descending"], g["ascending"]])
    return g


def _dedup_csr(post, pre, val, n) -> sp.csr_matrix:
    W = sp.csr_matrix((val, (post, pre)), shape=(n, n), dtype=np.float32)
    W.sum_duplicates()
    W.eliminate_zeros()
    return W


def spectral_radius(W: sp.csr_matrix) -> float:
    lam = eigs(W.astype(np.float64), k=1, which="LM", return_eigenvectors=False, maxiter=20000, tol=1e-5)
    return float(abs(lam[0]))


def build_matrix(variant: str = "connectome", glu_sign: float = -1.0, rho: float = 0.99,
                 seed: int = 0, ablate: tuple[str, ...] = (), subset: np.ndarray | None = None,
                 norm: str = "input_fraction", use_cache: bool = True) -> tuple[sp.csr_matrix, dict]:
    """variant:
    - connectome: 실제 배선, 실제 시냅스 수, 신경전달물질 부호
    - random: 같은 연결 수의 무작위 배선. 시냅스 수 분포와 뉴런 부호는 그대로
    - degree_shuffle: 뉴런마다 들어오고 나가는 연결 수는 유지하고 연결 상대만 섞음
    - weight_shuffle: 배선은 유지하고 연결마다 시냅스 수만 섞음
    ablate: groups()의 이름. 해당 뉴런의 들어오고 나가는 연결을 모두 끊는다.
    subset: 이 뉴런들만 남긴 부분 커넥톰(인덱스는 원래 N 크기를 유지).
    norm:
    - input_fraction: 뉴런마다 받는 시냅스 절댓값 합을 1로 맞춘다(연결 가중치 = 입력 비율).
      시냅스 수천 개짜리 소수 연결이 spectral radius를 지배해 나머지 신호가 사라지는 것을 막는다.
    - raw: 시냅스 수를 그대로 쓰고 spectral radius로만 나눈다.
    """
    key = f"W_{variant}_{norm}_glu{int(glu_sign):+d}_rho{rho}_s{seed}_abl{'-'.join(ablate) or 'none'}"
    if subset is not None:
        key += f"_sub{len(subset)}_{int(np.asarray(subset).sum()) % 100000}"
    cache = DERIVED / "matrices" / f"{key}.npz"
    meta_path = cache.with_suffix(".json")
    if use_cache and cache.exists():
        meta = json.loads(meta_path.read_text())
        meta["path"] = str(cache)
        return sp.load_npz(cache).tocsr(), meta

    neurons = load_neurons()
    n = len(neurons)
    pre, post, w = load_edges(neurons)
    sign = neuron_signs(neurons, glu_sign)
    rng = np.random.default_rng(seed)

    if variant == "connectome":
        p, q, ww = pre, post, w
    elif variant == "random":
        m = len(pre)
        p = rng.integers(0, n, m, dtype=np.int32)
        q = rng.integers(0, n, m, dtype=np.int32)
        ww = rng.permutation(w)
    elif variant == "degree_shuffle":
        # pre 쪽(나가는 연결 수, 나가는 시냅스 합)은 그대로 두고 post만 섞는다 -> 들어오는 연결 수도 유지
        p, ww = pre, w
        q = rng.permutation(post)
    elif variant == "weight_shuffle":
        p, q = pre, post
        ww = rng.permutation(w)
    else:
        raise ValueError(variant)

    keep = np.ones(len(p), dtype=bool)
    if ablate or subset is not None:
        g = groups(neurons)
        dead = np.zeros(n, dtype=bool)
        for name in ablate:
            dead[g[name]] = True
        if subset is not None:
            alive = np.zeros(n, dtype=bool)
            alive[subset] = True
            dead |= ~alive
        keep = ~dead[p] & ~dead[q]
    val = (ww[keep] * sign[p[keep]]).astype(np.float32)
    W = _dedup_csr(q[keep], p[keep], val, n)
    if norm == "input_fraction":
        s = np.asarray(abs(W).sum(axis=1)).ravel()
        s[s == 0] = 1.0
        W = sp.diags((1.0 / s).astype(np.float32)) @ W
        W = W.tocsr()
    elif norm != "raw":
        raise ValueError(norm)
    sr = spectral_radius(W)
    W = (W * (rho / sr)).astype(np.float32).tocsr()
    meta = {"variant": variant, "norm": norm, "glu_sign": glu_sign, "rho": rho, "seed": seed, "ablate": list(ablate),
            "subset": None if subset is None else int(len(subset)),
            "n": n, "nnz": int(W.nnz), "raw_spectral_radius": sr}
    cache.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(cache, W)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    meta["path"] = str(cache)
    return W, meta


def top_connected(n_keep: int) -> np.ndarray:
    """Costi 등(2025)의 'most connected' 기준: 들어오고 나가는 연결 수 합이 큰 뉴런 n_keep개."""
    neurons = load_neurons()
    pre, post, _ = load_edges(neurons)
    deg = np.bincount(pre, minlength=len(neurons)) + np.bincount(post, minlength=len(neurons))
    return np.argsort(-deg)[:n_keep]
