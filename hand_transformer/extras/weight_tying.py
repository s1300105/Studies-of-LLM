"""発展課題 C3：重み共有（weight tying）Wout = Eᵀ。

GPT-2 などは出力層に埋め込み行列の転置を使う（Press & Wolf 2017）。本編では共有しなかったので
「入力に出てこない 鳴く・。 の行は dE = 0」（6.9）だった。共有すると E は2か所で使われるので、
分岐の規則（derivations.md 問2）どおり勾配は2つの経路の和になる：
    dE（共有） = dE（入力側, 6.9） + dWoutᵀ（出力側, 6.1）
出力側の dWout には全語彙の列があるので、鳴く・。 の行にも勾配が来る。

このスクリプトは
  1. 本編の E をそのまま使い、Wout を Eᵀ に置き換えたモデルで、勾配の2つの成分を表示する
  2. 合計が数値微分と一致することを確かめる
  3. 共有あり / なしで SGD を回し、損失を比べる（パラメータ数は 180 → 160）
本編の初期値は共有を前提に作っていない（Wout を Eᵀ に置き換えると出発点そのものが変わる）ので、
3 は「同じ lr で同じように学習できるとは限らない」ことを見るための比較。

実行:  python3 extras/weight_tying.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)   # 発散時の overflow 警告を黙らせる


def tie(p):
    q = {k: v.copy() for k, v in p.items()}
    q["Wout"] = q["E"].T.copy()
    return q


def tied_grads(p):
    c = mg.forward(p)
    g, _ = mg.backward(p, c)
    g = dict(g)
    parts = (g["E"].copy(), g["Wout"].T.copy())
    g["E"] = parts[0] + parts[1]
    g["Wout"] = g["E"].T
    return c["loss"], g, parts


def train(p, lr, steps, tied):
    hist = []
    for _ in range(steps + 1):
        if tied:
            loss, g, _ = tied_grads(p)
        else:
            c = mg.forward(p)
            loss, (g, _) = c["loss"], mg.backward(p, c)
        hist.append(loss)
        p = mg.sgd(p, g, lr)
        if tied:
            p["Wout"] = p["E"].T.copy()
    return hist


def main():
    p0, lr = mg.load_params()
    pt = tie(p0)
    loss, g, (d_in, d_out) = tied_grads(pt)
    print(f"Wout = Eᵀ にしたモデルの初期損失 L = {loss:.4f}（本編は {mg.forward(p0)['loss']:.4f}）")
    print("\ndE の2つの成分（行 = トークン）")
    print(f"  {'':<6}{'入力側 dE（6.9 と同じ形）':<40}{'出力側 dWoutᵀ':<40}")
    for i, w in enumerate(mg.VOCAB):
        print(f"  {w:<5}" + "".join(f"{v:9.4f}" for v in d_in[i]) + "   " + "".join(f"{v:9.4f}" for v in d_out[i]))

    h, worst = 1e-6, 0.0
    for idx in np.ndindex(pt["E"].shape):
        q = {k: v.copy() for k, v in pt.items()}
        q["E"][idx] += h
        lp = mg.forward(tie(q))["loss"]
        q["E"][idx] -= 2 * h
        lm = mg.forward(tie(q))["loss"]
        worst = max(worst, abs((lp - lm) / (2 * h) - g["E"][idx]))
    print(f"\n合計 dE と数値微分（E を動かすと Wout も一緒に動く）の最大差: {worst:.1e}")
    c0 = mg.forward(p0)
    g0, _ = mg.backward(p0, c0)
    print(f"‖g‖²: 共有なし {sum(float((g0[k] ** 2).sum()) for k in mg.PARAMS):.4f}  "
          f"共有あり {sum(float((g[k] ** 2).sum()) for k in mg.PARAMS if k != 'Wout'):.4f}")

    steps = (0, 1, 2, 5, 10, 50, 200)
    print("\nSGD の損失（step " + ", ".join(map(str, steps)) + "）")
    for lr_ in (0.1, 0.3, 1.0):
        for tied, start in ((False, p0), (True, pt)):
            hist = train(start, lr_, steps[-1], tied)
            print(f"  lr={lr_:<4g} {'共有あり（160個）' if tied else '共有なし（180個）'}  "
                  + " ".join(f"{hist[s]:.4f}" if np.isfinite(hist[s]) else "発散" for s in steps))


if __name__ == "__main__":
    main()
