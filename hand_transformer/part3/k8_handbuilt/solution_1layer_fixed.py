"""K8 の発展：N が固定なら、1層・1ヘッドでも「位置だけ」で2回目を当てられることを、重みを手で作って確かめる。

2回目のクエリ位置 i の答えは、位置 i − N + 1 のトークン（コピー元）。N が決まっていれば、これは「N − 1 個前」という
固定のずれなので、previous-token head と同じ作り方（POS どうしをずらして結ぶ）で、ずれを 1 から N − 1 に変えればよい。
トークンの中身を一度も見ずに位置だけで答えを探すので、N が系列ごとに変わるデータでは外れる（check.py --eval-varied）。

    区画   中身
    TOK    今のトークンの one-hot（E）
    POS    今の位置の one-hot（P）
    OUT    答えの one-hot（層0 のヘッドが書く）
"""
import numpy as np

S0 = 10.0
S2 = 10.0


def build(V, T, n=8, sink=False):
    nv = V + 2
    TOK, POS = 0, nv
    OUT = POS + T
    D = OUT + nv
    E = np.zeros((nv, D))
    E[:, TOK:TOK + nv] = np.eye(nv)
    P = np.zeros((T, D))
    P[:, POS:POS + T] = np.eye(T)
    dk = T
    Wq = np.zeros((D, dk))
    Wk = np.zeros((D, dk))
    for i in range(T):
        Wq[POS + i, i] = S0 * np.sqrt(dk)
    for j in range(T):
        if j + n - 1 < T:
            Wk[POS + j, j + n - 1] = 1.0         # キー位置 j = u_{j+N−1} → q_i·k_j = [j = i − (N − 1)]
    Wv = np.zeros((D, nv))
    Wv[TOK:TOK + nv, :] = np.eye(nv)
    Wo = np.zeros((nv, D))
    Wo[:, OUT:OUT + nv] = np.eye(nv)
    Wout = np.zeros((D, nv))
    Wout[OUT:OUT + nv, :] = S2 * np.eye(nv)
    return {"E": E, "P": P, "Wout": Wout,
            "L0H0.Wq": Wq, "L0H0.Wk": Wk, "L0H0.Wv": Wv, "L0H0.Wo": Wo}
