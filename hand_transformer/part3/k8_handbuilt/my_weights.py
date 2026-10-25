"""K8：ここに自分で重みを書く（2層 Attention-only Transformer で induction head を作る）。

python3 check.py my_weights.py で採点する。手順と考え方は README.md。

約束（../k7_induction/attn_only.py の順伝播）:
    X0 = E[ids] + P[位置]
    層 l のヘッド h:  S = (X Wq)(X Wk)ᵀ/√dk（因果マスク）、A = softmax(S)、X ← X + A (X Wv) Wo
    logits = X_最後 Wout
    トークン番号：0..V−1 が内容、V が <s>、V+1 が <pad>（語彙 nv = V + 2）

最初は全部 0 になっている（このまま採点すると acc_2nd は 0.06 前後）。TODO の所を埋める。
"""
import numpy as np


def build(V, T, sink=False):
    nv = V + 2
    # ---- 1. 残差ストリームの区画を決める（README の手順1）
    TOK = 0              # 今のトークンの one-hot を置く区画の先頭（幅 nv）
    POS = nv             # 今の位置の one-hot を置く区画の先頭（幅 T）
    PREV = POS + T       # 1つ前のトークンを置く区画の先頭（幅 nv）。層0 のヘッドが書く
    OUT = PREV + nv      # 答えを置く区画の先頭（幅 nv）。層1 のヘッドが書き、Wout が読む
    D = OUT + nv

    E = np.zeros((nv, D))
    P = np.zeros((T, D))
    # TODO（手順2）: E の行 t の TOK 区画に one-hot、P の行 i の POS 区画に one-hot を置く

    # ---- 層0 ヘッド0：previous-token head（手順3）
    dk0, dv0 = T, nv
    Wq0, Wk0 = np.zeros((D, dk0)), np.zeros((D, dk0))
    Wv0, Wo0 = np.zeros((D, dv0)), np.zeros((dv0, D))
    # TODO: 位置 i のクエリと位置 j のキーの内積が「j = i − 1 のときだけ大きい」ようにする（√dk で割られることに注意）
    # TODO: 値はトークン（TOK 区画）を読み、出力は PREV 区画に書く

    # ---- 層1 ヘッド0：induction head（手順4）
    dk1, dv1 = nv, nv
    Wq1, Wk1 = np.zeros((D, dk1)), np.zeros((D, dk1))
    Wv1, Wo1 = np.zeros((D, dv1)), np.zeros((dv1, D))
    # TODO: クエリは今のトークン（TOK）、キーは PREV を読み、「同じトークンなら大きい」ようにする
    # TODO: 値はトークン（TOK）を読み、出力は OUT 区画に書く

    # ---- 出力（手順5）
    Wout = np.zeros((D, nv))
    # TODO: OUT 区画のトークン t を、logit の t 列に（十分大きな倍率で）写す

    return {"E": E, "P": P, "Wout": Wout,
            "L0H0.Wq": Wq0, "L0H0.Wk": Wk0, "L0H0.Wv": Wv0, "L0H0.Wo": Wo0,
            "L1H0.Wq": Wq1, "L1H0.Wk": Wk1, "L1H0.Wv": Wv1, "L1H0.Wo": Wo1}
