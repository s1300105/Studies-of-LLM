"""発展課題 S3：ロジットを「誰の寄与か」に分解し、1ステップの更新で何が変わったかを見る。

このモデルには最終 LayerNorm が無いので、残差ストリームは単なる足し算
    X2 = X0 + (ヘッド0 の出力) + (ヘッド1 の出力) + F
で、logits = X2 Wout も厳密に4つの寄与の和に分かれる（mini_gpt.py --decompose）。
    直接経路  X0 Wout                     … 埋め込みだけで決まる「バイグラム」的な予測
    ヘッド h  O_h Wo[2h:2h+2, :] Wout       … Attention で他の位置から運んできた情報
    FFN       F Wout

softmax は全ロジットに同じ数を足しても変わらないので、意味があるのは「正解トークンのロジットが
他より相対的にどれだけ高いか」。ここでは
    中心化ロジット = 正解のロジット − その位置の全5語のロジットの平均
も分解する（平均は線形なので、これも厳密に各寄与の和になる）。

実行:  python3 extras/decompose_update.py
（更新前の行列全体は  python3 mini_gpt.py --decompose  でも見られる）
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402


def table(p):
    """{寄与の名前: (3位置分の正解ロジット, 3位置分の中心化ロジット)}"""
    c = mg.forward(p)
    out = {}
    for name, z in mg.logit_decomposition(p, c).items():
        raw = z[np.arange(mg.T), mg.TGT]
        out[name] = (raw, raw - z.mean(-1))
    out["合計"] = tuple(sum(v[i] for v in out.values()) for i in range(2))
    return out, c


def main():
    p0, lr = mg.load_params()
    c0 = mg.forward(p0)
    g, _ = mg.backward(p0, c0)
    p1 = mg.sgd(p0, g, lr)
    before, cb = table(p0)
    after, ca = table(p1)
    assert np.allclose(before["合計"][0], cb["logits"][np.arange(mg.T), mg.TGT])
    assert np.allclose(after["合計"][0], ca["logits"][np.arange(mg.T), mg.TGT])

    heads = [f"{mg.POS[t]}→{mg.VOCAB[mg.TGT[t]]}" for t in range(mg.T)]
    for k, title in ((0, "正解トークンのロジット"), (1, "中心化ロジット（正解 − 5語の平均）")):
        print(f"\n=== {title}：更新前 → 更新後（変化）===")
        print(f"  {'寄与':<10}" + "".join(f"{h:>26}" for h in heads))
        for name in before:
            cells = "".join(f"{before[name][k][t]:>8.4f} →{after[name][k][t]:>8.4f} ({after[name][k][t] - before[name][k][t]:+.4f})"
                            for t in range(mg.T))
            print(f"  {name:<10}{cells}")

    print("\n=== 中心化ロジットの増分のうち、各寄与が占める割合 ===")
    tot = after["合計"][1] - before["合計"][1]
    for name in list(before)[:-1]:
        d = after[name][1] - before[name][1]
        print(f"  {name:<10}" + "".join(f"{100 * d[t] / tot[t]:>10.1f}%" for t in range(mg.T)))


if __name__ == "__main__":
    main()
