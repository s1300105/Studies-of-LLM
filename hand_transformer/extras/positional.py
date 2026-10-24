"""発展課題 C2：sin/cos 位置埋め込み（元論文）と RoPE（回転位置埋め込み）。

本編は学習型の位置埋め込み P（GPT 型）を使い、U1 = LN(X0) が ±1 になるよう値を選んだ。
元論文（Vaswani et al. 2017）の固定の位置埋め込みは
    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
d=4 なら i=0 で周波数 1、i=1 で周波数 1/100。T=3 なら 12 個の値を関数電卓で求められる（ラジアンモードで！）。

RoPE（Su et al. 2021、LLaMA などが採用）は X0 に足すのではなく、各ヘッドの q, k の成分ペアを
位置 m に比例した角度 mθ だけ回転させる。回転行列の性質 R(mθ)ᵀR(nθ) = R((n−m)θ) から
    (R_m q)·(R_n k) = qᵀ R((n−m)θ) k
となり、スコアは**位置の差 n−m だけ**で決まる。これを数値で確かめる。

実行:  python3 extras/positional.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402


def sinusoidal(T, d):
    pos = np.arange(T)[:, None]
    freq = 1.0 / 10000 ** (np.arange(0, d, 2) / d)
    pe = np.zeros((T, d))
    pe[:, 0::2] = np.sin(pos * freq)
    pe[:, 1::2] = np.cos(pos * freq)
    return pe


def rotate(x, pos, theta):
    """x の成分ペア (x0, x1), (x2, x3), ... を角度 pos·θ_i だけ回転する（RoPE）。"""
    out = x.copy()
    for i, th in enumerate(theta):
        c, s = np.cos(pos * th), np.sin(pos * th)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i], out[2 * i + 1] = c * a - s * b, s * a + c * b
    return out


def main():
    pe = sinusoidal(mg.T, mg.D)
    print("[1] sin/cos 位置埋め込み（T=3, d=4）")
    print("            d0        d1        d2        d3")
    for t in range(mg.T):
        print(f"  pos{t} " + "".join(f"{v:10.5f}" for v in pe[t]))
    print("  内積 PE[m]·PE[n]（位置の差だけで決まるか）:")
    for m, n in ((0, 1), (1, 2), (0, 2)):
        print(f"    PE[{m}]·PE[{n}] = {pe[m] @ pe[n]:.5f}   （差 {n - m}）")

    p, _ = mg.load_params()
    X0 = p["E"][mg.INP] + pe
    U1, st = mg.layernorm(X0)
    print("\n  本編の E に P の代わりに PE を足すと、U1 = LN(E[ids] + PE) は（本編では全成分 ±1）:")
    for t in range(mg.T):
        print(f"    {mg.POS[t]:<8}" + "".join(f"{v:9.4f}" for v in U1[t]) + f"    σ = {st['sigma'][t, 0]:.4f}")

    print("\n[2] RoPE：スコアが位置の差だけで決まることの確認（d_k=4、θ = 1, 1/100）")
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=4), rng.normal(size=4)
    theta = 1.0 / 10000 ** (np.arange(0, 4, 2) / 4)
    for m, n in ((0, 1), (5, 6), (100, 101), (0, 3), (7, 10)):
        s = rotate(q, m, theta) @ rotate(k, n, theta)
        print(f"  クエリ位置 m={m:<3d} キー位置 n={n:<3d}（差 {n - m}）  スコア = {s:.6f}")
    print(f"  参考: 回転なしのスコア q·k = {q @ k:.6f}")
    print(f"  回転してもベクトルの長さは変わらない: |q| = {np.linalg.norm(q):.6f}, |R_5 q| = {np.linalg.norm(rotate(q, 5, theta)):.6f}")


if __name__ == "__main__":
    main()
