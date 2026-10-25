"""K8 の参考解：2層 Attention-only Transformer の重みを手で決めて induction head を作る。

自分の my_weights.py で一度は試してから読むこと。

残差ストリームを4つの「区画」（部分空間）に分け、各区画を1つの変数として使う（RASP / tracr の考え方）:

    区画   次元                 中身                                  書く人
    TOK    0 .. V+1             今のトークンの one-hot                 E
    POS    V+2 .. V+1+T         今の位置の one-hot                     P
    PREV   V+2+T .. 2V+3+T      1つ前のトークンの one-hot              層0 のヘッド（previous-token head）
    OUT    2V+4+T .. 3V+5+T     答え（コピー元のトークン）の one-hot   層1 のヘッド（induction head）

層0 ヘッド0（previous-token head）
    QK: クエリ位置 i とキー位置 j の内積が [j = i−1] になるよう、POS どうしをずらして結ぶ
    OV: キー位置のトークン（TOK）を読んで PREV に書く
層1 ヘッド0（induction head）
    QK: クエリの TOK とキーの PREV を結ぶ → 「1つ前のトークンが今のトークンと同じ位置」が高得点（K-composition）
    OV: その位置の TOK を読んで OUT に書く
Wout: OUT を読んで logit にする

S = Q·Kᵀ/√dk なので、欲しい点差 s を出すには Q か K を √dk 倍しておく。
"""
import numpy as np

S0 = 10.0     # 層0：正しいキー（i−1）と他のキーの点差（softmax の前）
S1 = 10.0     # 層1：コピー元と他のキーの点差
S2 = 10.0     # 出力：答えの logit と他の logit の差


def build(V, T, sink=False):
    nv = V + 2                                   # 語彙（内容 V 個 + <s> + <pad>）
    BOS = V
    TOK, POS = 0, nv
    PREV = POS + T
    OUT = PREV + nv
    D = OUT + nv
    I = np.eye

    E = np.zeros((nv, D))
    E[:, TOK:TOK + nv] = I(nv)
    P = np.zeros((T, D))
    P[:, POS:POS + T] = I(T)

    # 層0：previous-token head（dk = T、dv = nv）
    dk0 = T
    Wq0 = np.zeros((D, dk0))
    Wk0 = np.zeros((D, dk0))
    for i in range(T):
        Wq0[POS + i, i] = S0 * np.sqrt(dk0)      # 位置 i のクエリ = u_i（√dk 倍）
    for j in range(T - 1):
        Wk0[POS + j, j + 1] = 1.0                # 位置 j のキー = u_{j+1} → q_i·k_j = [j + 1 = i]
    Wv0 = np.zeros((D, nv))
    Wv0[TOK:TOK + nv, :] = I(nv)                 # 値 = そのトークンの one-hot
    Wo0 = np.zeros((nv, D))
    Wo0[:, PREV:PREV + nv] = I(nv)               # PREV 区画に書く

    # 層1：induction head（dk = nv、改良版は +1。dv = nv）
    dk1 = nv + (1 if sink else 0)
    Wq1 = np.zeros((D, dk1))
    Wk1 = np.zeros((D, dk1))
    Wq1[TOK:TOK + nv, :nv] = S1 * np.sqrt(dk1) * I(nv)  # クエリ = 今のトークン（√dk 倍）
    Wk1[PREV:PREV + nv, :nv] = I(nv)                     # キー = その位置の1つ前のトークン → 一致なら S1 点
    if sink:
        # 改良版：一致するキーが無いとき（1回目）は <s>（位置0）を見る。
        # 余分な1次元で「どのクエリも、位置0 のキーとだけ S1/2 点で合う」ようにする（一致すれば S1 > S1/2 で一致が勝つ）。
        Wq1[TOK:TOK + nv, nv] = S1 / 2 * np.sqrt(dk1)
        Wk1[POS + 0, nv] = 1.0
    Wv1 = np.zeros((D, nv))
    Wv1[TOK:TOK + nv, :] = I(nv)
    Wo1 = np.zeros((nv, D))
    Wo1[:, OUT:OUT + nv] = I(nv)

    Wout = np.zeros((D, nv))
    Wout[OUT:OUT + nv, :] = S2 * I(nv)
    if sink:
        Wout[OUT + BOS, :] = 0.0                 # <s> をコピーしたら「分からない」＝ logit は全部同じ
    return {"E": E, "P": P, "Wout": Wout,
            "L0H0.Wq": Wq0, "L0H0.Wk": Wk0, "L0H0.Wv": Wv0, "L0H0.Wo": Wo0,
            "L1H0.Wq": Wq1, "L1H0.Wk": Wk1, "L1H0.Wv": Wv1, "L1H0.Wo": Wo1}
