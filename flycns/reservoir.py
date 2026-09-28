"""커넥톰 저장소(reservoir)를 여러 프로세스로 나눠 돌린다.

동역학 (누설 적분 tanh 저장소):
    x(t+1) = (1 - a) * x(t) + a * tanh(W @ x(t) + I(t))
W는 (post, pre) CSR, I(t)는 입력 전류(Win @ u(t)).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

_W = None
_WIN = None


def _init(w_path: str, win_path: str):
    global _W, _WIN
    os.environ["OMP_NUM_THREADS"] = "1"
    _W = sp.load_npz(w_path).tocsr().astype(np.float32)
    _WIN = sp.load_npz(win_path).tocsr().astype(np.float32)


def _run_seq_chunk(args):
    """시계열 여러 개를 열로 묶어 함께 돌리고, readout 뉴런의 상태를 매 스텝 기록한다."""
    seqs, leak, readout, substeps = args
    n = _W.shape[0]
    lens = [len(s) for s in seqs]
    tmax = max(lens)
    b = len(seqs)
    x = np.zeros((n, b), dtype=np.float32)
    out = [np.zeros((L, len(readout)), dtype=np.float32) for L in lens]
    u = np.zeros((tmax, _WIN.shape[1], b), dtype=np.float32)
    for j, s in enumerate(seqs):
        u[: len(s), :, j] = s
    for t in range(tmax):
        inp = _WIN @ u[t]
        for _ in range(substeps):
            x = (1.0 - leak) * x + leak * np.tanh(_W @ x + inp)
        r = x[readout]
        for j, L in enumerate(lens):
            if t < L:
                out[j][t] = r[:, j]
    return out


def _run_static_chunk(args):
    """입력을 T스텝 동안 일정하게 주고, 마지막 k스텝의 readout 평균을 돌려준다."""
    u, leak, readout, steps, last_k = args
    n = _W.shape[0]
    inp = _WIN @ u                      # (n, b)
    x = np.zeros((n, u.shape[1]), dtype=np.float32)
    acc = np.zeros((len(readout), u.shape[1]), dtype=np.float32)
    for t in range(steps):
        x = (1.0 - leak) * x + leak * np.tanh(_W @ x + inp)
        if t >= steps - last_k:
            acc += x[readout]
    return (acc / last_k).T


def run_sequences(w_path, win_path, seqs, leak, readout, substeps=1, workers=12, chunk=None, log=None):
    chunk = chunk or max(1, int(np.ceil(len(seqs) / workers)))
    order = np.argsort([-len(s) for s in seqs])           # 긴 것부터 나눠 부하를 맞춘다
    chunks = [order[i:i + chunk] for i in range(0, len(order), chunk)]
    t0 = time.time()
    res = [None] * len(seqs)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(w_path), str(win_path))) as ex:
        futs = [ex.submit(_run_seq_chunk, ([seqs[i] for i in c], leak, readout, substeps)) for c in chunks]
        for k, (c, f) in enumerate(zip(chunks, futs)):
            for i, o in zip(c, f.result()):
                res[i] = o
            if log:
                el = time.time() - t0
                log(f"  chunk {k + 1}/{len(chunks)} done, {el:.0f}s elapsed, ETA {el / (k + 1) * (len(chunks) - k - 1):.0f}s")
    return res


def run_static(w_path, win_path, U, leak, readout, steps=8, last_k=2, workers=12, chunk=64, log=None):
    """U: (입력 차원, 표본 수). 표본마다 readout 특징 벡터를 돌려준다."""
    idx = np.arange(U.shape[1])
    chunks = [idx[i:i + chunk] for i in range(0, len(idx), chunk)]
    t0 = time.time()
    out = np.zeros((U.shape[1], len(readout)), dtype=np.float32)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(str(w_path), str(win_path))) as ex:
        futs = [ex.submit(_run_static_chunk, (U[:, c], leak, readout, steps, last_k)) for c in chunks]
        for k, (c, f) in enumerate(zip(chunks, futs)):
            out[c] = f.result()
            if log and (k + 1) % max(1, len(chunks) // 10) == 0:
                el = time.time() - t0
                log(f"  {k + 1}/{len(chunks)} chunks, {el:.0f}s elapsed, ETA {el / (k + 1) * (len(chunks) - k - 1):.0f}s")
    return out
