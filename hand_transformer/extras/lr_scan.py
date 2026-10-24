"""発展課題 M3：本編と同じ1文で何ステップも学習し、学習率 lr を振る。

手計算では lr=1 で1ステップだけ進めて 1.3949 → 0.1056 を見た。ここでは
  1. lr ∈ {0.25, 0.5, 1, 2, 4} で100ステップ学習したときの損失曲線
  2. 1次近似 ΔL ≈ −lr·‖g‖²（7.2）が、lr をどこまで上げると外れるか
  3. 最初の勾配方向に沿って η だけ進んだときの損失 L(θ − η·g)（「1ステップ目」の全体像）
  4. lr をさらに上げたときに本当に発散するか
を確かめる。

損失は mini_gpt.py と同じモデルで計算するが、ここでは nll = logsumexp − logit_正解 の形で計算する
（mini_gpt.py の −log(softmax) は、確率が 0 にアンダーフローすると inf になるため）。

実行:  python3 extras/lr_scan.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # 発散時の overflow 警告を黙らせる


def stable_loss(logits):
    """nll = logsumexp(z) − z_正解。確率を経由しないので inf になりにくい。"""
    m = logits.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(logits - m).sum(-1, keepdims=True)))[:, 0]
    return float((lse - logits[np.arange(mg.T), mg.TGT]).mean())


def run(p, lr, steps):
    """SGD を steps 回。各ステップの損失と、全位置正解になった最初のステップを返す。"""
    hist, first_ok = [], None
    for s in range(steps + 1):
        c = mg.forward(p)
        hist.append(stable_loss(c["logits"]))
        if first_ok is None and np.array_equal(c["logits"].argmax(-1), mg.TGT):
            first_ok = s
        if s < steps:
            if not np.isfinite(hist[-1]):
                hist += [np.nan] * (steps - s)
                break
            g, _ = mg.backward(p, c)
            p = mg.sgd(p, g, lr)
    return hist, first_ok


def fmt(x):
    if not np.isfinite(x):
        return "発散"
    return f"{x:.4f}" if 1e-4 <= x < 1e4 else f"{x:.1e}"


def main():
    p0, _ = mg.load_params()
    c0 = mg.forward(p0)
    g0, _ = mg.backward(p0, c0)
    L0 = c0["loss"]
    gsq = sum(float((g0[k] ** 2).sum()) for k in mg.PARAMS)
    print(f"初期損失 L = {L0:.4f},  ‖g‖² = {gsq:.4f}")

    print("\n[1] 損失曲線（SGD、params.json から）")
    show = [0, 1, 2, 3, 5, 10, 20, 50, 100]
    print("   lr  " + "".join(f"{f'step{s}':>9}" for s in show) + "   全問正解になったステップ")
    for lr in (0.25, 0.5, 1, 2, 4):
        hist, ok = run(p0, lr, 100)
        print(f"{lr:5g}  " + "".join(f"{fmt(hist[s]):>9}" for s in show) + f"   {ok}")

    print("\n[2] 1ステップ目：1次近似 ΔL ≈ −lr‖g‖² と実際の ΔL")
    print("   lr    予測ΔL    実際ΔL   実際/予測   予測した L'")
    for lr in (0.01, 0.1, 0.25, 0.5, 1, 2, 4):
        L1 = stable_loss(mg.forward(mg.sgd(p0, g0, lr))["logits"])
        print(f"{lr:5g} {-lr * gsq:9.4f} {L1 - L0:9.4f} {(L1 - L0) / (-lr * gsq):9.3f} {L0 - lr * gsq:11.4f}")
    print(f"  予測した L' が負（ありえない値）になる境目: lr = L/‖g‖² = {L0 / gsq:.3f}")
    f = lambda eta: stable_loss(mg.forward(mg.sgd(p0, g0, eta))["logits"])
    h = 1e-3
    print(f"  勾配方向の傾き dL/dη = {(f(h) - f(-h)) / (2 * h):.4f}（= −‖g‖²）, "
          f"曲率 d²L/dη² = {(f(h) - 2 * f(0) + f(-h)) / h ** 2:.4f}")

    print("\n[3] 最初の勾配方向に沿った損失 L(θ − η·g)（η = その lr で1ステップ進んだ点）")
    for eta in (0, 0.5, 1, 2, 4, 6, 8, 9, 10, 12, 16):
        c = mg.forward(mg.sgd(p0, g0, eta))
        pred = " ".join(mg.VOCAB[i] for i in c["logits"].argmax(-1))
        print(f"  η={eta:<4g} L={fmt(stable_loss(c['logits'])):>9}  予測: {pred}")

    print("\n[4] lr を上げていくと発散するか（最初の6ステップと100ステップ後）")
    for lr in (4, 8, 9, 10, 16):
        hist, _ = run(p0, lr, 100)
        print(f"  lr={lr:<3g} " + " → ".join(fmt(x) for x in hist[:6]) + f"  …  step100: {fmt(hist[100])}")


if __name__ == "__main__":
    main()
