"""発展課題 M2：Wout の1成分の勾配を、手で数値微分して 6.1 の dWout と照合する。

選ぶ成分は Wout[d0, 鳴く]（初期値 0、6.1 の解析的な勾配 dWout[d0, 鳴く] = −0.3373）。
Wout[i, j] を h だけ動かすと、変わるのは logits の j 列だけで、その変化量は h·X2[:, i]（4.5 の X2 の i 列）。
したがって順伝播をやり直す必要はなく、5.1 の logits の1列を書き換えて 5.2〜5.3 だけ計算し直せばよい（約20分）。
「鳴く」列はもともと全部 0 なので、書き換えた後の値は h·X2[t, d0] そのもの。

    中心差分  (L(+h) − L(−h)) / 2h      誤差は h² に比例
    前進差分  (L(+h) − L) / h            誤差は h に比例

手計算では L を小数4桁（有効数字4〜5桁）までしか持てないので、h を小さくしすぎると引き算で桁が消える。
下の表の「L を小数4桁に丸めた場合」の列で、手計算に向く h を確かめる。

実行:  python3 extras/numgrad_by_hand.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402

ROW, COL = 0, mg.VOCAB.index("鳴く")


def loss_with(p, delta):
    q = {k: v.copy() for k, v in p.items()}
    q["Wout"][ROW, COL] += delta
    return mg.forward(q)


def main():
    p, _ = mg.load_params()
    c = mg.forward(p)
    g, _ = mg.backward(p, c)
    ana = g["Wout"][ROW, COL]
    print(f"対象: Wout[d{ROW}, {mg.VOCAB[COL]}] = {p['Wout'][ROW, COL]:g}、解析的な勾配（6.1）= {ana:.4f}")
    print(f"X2 の d{ROW} 列（4.5）= {np.round(c['X2'][:, ROW], 4)}  → h だけ動かすと logits の「{mg.VOCAB[COL]}」列がこの h 倍になる")

    print("\n手計算用の途中の値（h = 0.1）")
    for sign in (+1, -1):
        cc = loss_with(p, sign * 0.1)
        print(f"  h = {sign * 0.1:+.1f}:  logits[:, {mg.VOCAB[COL]}] = {np.round(cc['logits'][:, COL], 4)}  "
              f"各位置の nll = {np.round(cc['nll'], 4)}  L = {cc['loss']:.4f}")

    print("\n   h      中心差分   誤差       前進差分   誤差      | L を小数4桁に丸めた場合: 中心差分   誤差")
    for h in (0.1, 0.05, 0.01, 0.001):
        lp, lm, l0 = loss_with(p, h)["loss"], loss_with(p, -h)["loss"], c["loss"]
        cen, fwd = (lp - lm) / (2 * h), (lp - l0) / h
        cen4 = (round(lp, 4) - round(lm, 4)) / (2 * h)
        print(f"  {h:<6g} {cen:9.4f} {cen - ana:+.1e}   {fwd:9.4f} {fwd - ana:+.1e}  |"
              f"                 {cen4:9.4f} {cen4 - ana:+.1e}")


if __name__ == "__main__":
    main()
