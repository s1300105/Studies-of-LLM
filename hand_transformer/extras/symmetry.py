"""発展課題：2つのヘッドを「全く同じ」初期値にするとどうなるか（対称性の破れ）。

ヘッド1 の Wq, Wk, Wv の列と Wo の行をヘッド0 のコピーにすると、2つのヘッドは
順伝播で同じ出力を出し、逆伝播でも同じ勾配を受け取るので、**何ステップ学習しても同一のまま**。
マルチヘッドが「別々のことを学ぶ」には初期値がランダム（非対称）であることが必要、という話。
ほんの少し（1e-3）ずらすと同一ではなくなるが、このタスクは1ヘッドで足りるので差はあまり広がらない
（2文タスクで Adam を使っても10倍程度）。「対称性は勾配降下では絶対に破れない」のが要点。

実行:  python extras/symmetry.py
"""
import numpy as np

from general_model import forward, init_params, train

INP = np.array([[0, 1, 2]])   # <s> 猫 が
TGT = np.array([[1, 2, 3]])   # 猫 が 鳴く


def tie_heads(p, noise=0.0, seed=1):
    p = {k: v.copy() for k, v in p.items()}
    rng = np.random.default_rng(seed)
    for w in ("Wq", "Wk", "Wv"):
        p[w][:, 2:4] = p[w][:, 0:2] + noise * rng.normal(size=(4, 2))
    p["Wo"][2:4, :] = p["Wo"][0:2, :] + noise * rng.normal(size=(2, 4))
    return p


def head_gap(p):
    return max(np.abs(p[w][:, 2:4] - p[w][:, 0:2]).max() for w in ("Wq", "Wk", "Wv"))


def main():
    base = init_params(5, 3, seed=0)
    for noise in (0.0, 1e-3):
        p = tie_heads(base, noise)
        print(f"\n初期のヘッド間の差 = {head_gap(p):.1e}")
        for steps in (10, 100, 1000):
            q, c, _ = train(p, INP, TGT, steps=steps, lr=0.5, optimizer="sgd")
            print(f"  {steps:5d}ステップ後  損失 {c['loss']:.4f}  ヘッド間の差 {head_gap(q):.1e}  "
                  f"Attention重みの差 {np.abs(c['A'][0] - c['A'][1]).max():.1e}")


if __name__ == "__main__":
    main()
