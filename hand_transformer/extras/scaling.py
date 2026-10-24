"""発展課題 S4：なぜスコアを √d_k で割るのか。

q, k の各成分が平均0・分散1で独立なら、内積 q·k = Σ_i q_i k_i は d_k 個の（分散1の）項の和なので
分散 d_k、標準偏差 √d_k になる。d_k が大きいとスコアの差が大きくなりすぎ、softmax がほぼ one-hot に張り付く。
張り付いた softmax は勾配も小さい（∂a_i/∂s_j = a_i(δ_ij − a_j) がほぼ 0）。√d_k で割ると d_k によらず標準偏差 1 に戻る。

ここでは乱数の q, k（キー16個）で
  - スコアの標準偏差
  - Attention 重みのエントロピー（log 16 = 2.77 なら一様、0 なら one-hot）
  - 最大の重み
  - softmax の勾配の大きさの目安 1 − Σ a_i²（= Σ_i ∂a_i/∂s_i、one-hot なら 0）
を、割る場合と割らない場合で比べる。本編は d_k = 2 なので √2 ≈ 1.41 倍の違いしかない。

実行:  python3 extras/scaling.py
"""
import numpy as np

N_KEYS, TRIALS = 16, 2000


def stats(dk, scale, rng):
    q = rng.normal(size=(TRIALS, 1, dk))
    k = rng.normal(size=(TRIALS, N_KEYS, dk))
    s = (q @ k.transpose(0, 2, 1))[:, 0, :]
    if scale:
        s = s / np.sqrt(dk)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    ent = -(a * np.log(np.maximum(a, 1e-300))).sum(-1)
    return s.std(), ent.mean(), a.max(-1).mean(), (1 - (a ** 2).sum(-1)).mean()


def main():
    rng = np.random.default_rng(0)
    print(f"キー {N_KEYS} 個、試行 {TRIALS} 回の平均。一様分布のエントロピー = log{N_KEYS} = {np.log(N_KEYS):.3f}")
    print("  d_k   割り方      スコアの標準偏差   エントロピー   最大の重み   1−Σa²（勾配の目安）")
    for dk in (2, 16, 128, 1024):
        for scale in (False, True):
            sd, ent, mx, gr = stats(dk, scale, rng)
            print(f"  {dk:4d}  {'÷√d_k' if scale else 'そのまま':<8}  {sd:12.3f}   {ent:12.3f}   {mx:10.3f}   {gr:12.3f}")


if __name__ == "__main__":
    main()
