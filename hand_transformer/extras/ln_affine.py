"""発展課題 S5：LayerNorm の γ, β も学習するとしたら、その勾配はいくつか。

本編では γ=1, β=0 に固定した。実際の LayerNorm は  U = γ ⊙ Û + β （Û = (X − μ)/σ、γ, β は4次元）。
勾配は（位置 t について足し合わせる。重みを3位置で共有しているから）
    dγ = Σ_t dU[t] ⊙ Û[t],     dβ = Σ_t dU[t]
γ=1, β=0 では Û = U なので、本編の dU1（6.7）と U1（2.3）、dU2（6.2）と U2（4.2）だけで手計算できる。
特に U1 は全成分 ±1 なので、dγ1 は dU1 の列を符号付きで足すだけ。
また γ=1 なら Û への勾配は dU のままなので、本編の他の勾配は1つも変わらない。

このスクリプトは
  1. dγ1, dβ1, dγ2, dβ2 を上の式で計算し、数値微分と照合する
  2. γ, β も含めて lr=1 で1ステップ更新したら L' が 0.1056 からどう変わるかを計算する

実行:  python3 extras/ln_affine.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402


def loss_with_affine(p, gb):
    """gb = [(γ1, β1), (γ2, β2)] を LN1, LN2 に入れて損失を計算する（mini_gpt.forward の LN を一時的に差し替え）。"""
    calls = iter(gb)
    orig = mg.layernorm

    def ln_affine(x):
        y, st = orig(x)
        g, b = next(calls)
        return g * y + b, st

    mg.layernorm = ln_affine
    try:
        return mg.forward(p)["loss"]
    finally:
        mg.layernorm = orig


def main():
    p, lr = mg.load_params()
    c = mg.forward(p)
    g, d = mg.backward(p, c)
    grads = {
        "dγ1": (d["U1"] * c["U1"]).sum(0), "dβ1": d["U1"].sum(0),
        "dγ2": (d["U2"] * c["U2"]).sum(0), "dβ2": d["U2"].sum(0),
    }
    one, zero = np.ones(mg.D), np.zeros(mg.D)
    base = [(one, zero), (one, zero)]
    h = 1e-6
    print("              d0        d1        d2        d3     数値微分との最大差")
    for name, val in grads.items():
        layer, which = int(name[2]) - 1, 0 if name[1] == "γ" else 1
        num = np.zeros(mg.D)
        for i in range(mg.D):
            plus = [list(t) for t in base]
            minus = [list(t) for t in base]
            plus[layer][which] = plus[layer][which] + h * np.eye(mg.D)[i]
            minus[layer][which] = minus[layer][which] - h * np.eye(mg.D)[i]
            num[i] = (loss_with_affine(p, plus) - loss_with_affine(p, minus)) / (2 * h)
        print(f"  {name}  " + "".join(f"{v:9.4f} " for v in val) + f"   {np.abs(num - val).max():.1e}")

    p1 = mg.sgd(p, g, lr)
    gsq = sum(float((g[k] ** 2).sum()) for k in mg.PARAMS)
    gsq_ln = sum(float((v ** 2).sum()) for v in grads.values())
    upd = [(one - lr * grads["dγ1"], zero - lr * grads["dβ1"]), (one - lr * grads["dγ2"], zero - lr * grads["dβ2"])]
    print(f"\n‖g‖²: 本編の180個 {gsq:.4f} + γβ の16個 {gsq_ln:.4f}")
    print(f"lr={lr:g} で1ステップ:  γβ固定 L' = {mg.forward(p1)['loss']:.4f}   γβも更新 L' = {loss_with_affine(p1, upd):.4f}")


if __name__ == "__main__":
    main()
