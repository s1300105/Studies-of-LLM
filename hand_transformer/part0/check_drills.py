#!/usr/bin/env python3
"""Part 0 コンポーネント・ドリル（D1〜D9）の正解と検算。

ドリル本文（d1_*.md 〜 d9_*.md）に出てくる数値は、すべてこのスクリプトの出力です。
各ドリルについて次の4つを行います。

  1. 答え（順伝播・逆伝播の値）を表示する
  2. 不変量（行和0、内積の一致など）を確かめる
  3. 勾配の答えすべてを、中心差分（h = 1e-6）の数値微分と照合する
  4. 手でやる数値微分（h = 0.1 など）の途中の値を表示する

記号は Part 1（../README.md §3）と同じ：行ベクトル規約（y = x @ W）、dX = ∂L/∂X、添字は0始まり。

使い方:
    python3 check_drills.py              # 全ドリル
    python3 check_drills.py --only D5    # 1つだけ（D1〜D9）
    python3 check_drills.py --quiet      # 検査結果だけ

終了コード: すべての検査に通れば 0、1つでも外れれば 1。
"""
import argparse
import sys

import numpy as np

H_NUM = 1e-6          # 照合用の数値微分の刻み
TOL = 1e-6            # 照合の許容差（相対）
RESULTS = []          # (ドリル, 名前, 合否)
QUIET = False
CURRENT = ""
SQ2 = np.sqrt(2.0)


# ---------------------------------------------------------------- 表示

def fnum(v):
    """数値を1つ表示用の文字列にする。

    小数第9位までで割り切れる値（0.375, 0.132 など）はそのまま全桁、
    それ以外は |v| ≥ 0.01 なら小数第4位まで、もっと小さければ有効数字4桁。
    """
    v = float(v)
    if not np.isfinite(v):
        return "-inf" if v < 0 else "inf"
    if abs(v) < 5e-13:
        return "0"
    r9 = round(v, 9)
    if abs(v - r9) < 1e-12:
        return f"{r9:.9f}".rstrip("0").rstrip(".")
    if abs(v) >= 0.01:
        return f"{v:.4f}"
    return f"{v:.4g}"


def fx(v, digits=6):
    """手計算の途中の値：割り切れる値はそのまま、そうでなければ小数第 digits 位まで。"""
    v = float(v)
    if abs(v) < 5e-13:
        return "0"
    r9 = round(v, 9)
    if abs(v - r9) < 1e-12 and len(f"{r9:.9f}".rstrip("0").split(".")[-1]) <= digits + 1:
        return fnum(v)
    return f"{v:.{digits}f}"


def fmt(a):
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return fnum(a)
    return "[" + ", ".join(fnum(v) for v in a.ravel()) + "]"


def say(*args):
    if not QUIET:
        print(*args)


def show(label, value, rows=None):
    """スカラー・ベクトルは1行、行列は行ラベルつきで表示する。"""
    a = np.asarray(value, dtype=float)
    if a.ndim < 2:
        say(f"  {label} = {fmt(a)}")
        return
    say(f"  {label} =")
    rows = rows or [f"r{i}" for i in range(a.shape[0])]
    for name, row in zip(rows, a):
        say(f"      {name:>4}: {fmt(row)}")


def section(title):
    say(f"\n--- {title}")


def check(name, ok):
    RESULTS.append((CURRENT, name, bool(ok)))
    if not ok or not QUIET:
        print(f"  [{'OK' if ok else 'NG'}] {name}")


def close(a, b, tol=1e-9):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol, rtol=0)


# ---------------------------------------------------------------- 数値微分

def numgrad(f, x, h=H_NUM):
    """f(x) の勾配を中心差分で求める（x は配列。全成分を1つずつ動かす）。"""
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += h
        xm[idx] -= h
        g[idx] = (f(xp) - f(xm)) / (2 * h)
    return g


def gradcheck(name, f, x, analytic):
    """解析的な勾配 analytic を、f の数値微分と照合する。"""
    g = numgrad(f, x)
    err = float(np.max(np.abs(g - analytic))) if np.size(g) else 0.0
    scale = 1.0 + float(np.max(np.abs(analytic))) if np.size(g) else 1.0
    check(f"数値微分（h=1e-6）と一致: {name}  max|差| = {err:.1e}", err <= TOL * scale)


def handdiff(name, f, x, idx, h, analytic, digits=6, forward=False):
    """手でやる数値微分の値を表示する。f はスカラーを返す関数、idx は動かす成分。"""
    x = np.array(x, dtype=float)
    xp, xm = x.copy(), x.copy()
    xp[idx] += h
    xm[idx] -= h
    lp, lm, l0 = f(xp), f(xm), f(x)
    cen = (lp - lm) / (2 * h)
    say(f"  手で数値微分 {name}（h = {h:g}）: 元の値 {fnum(x[idx])} → {fnum(xp[idx])} / {fnum(xm[idx])}")
    say(f"      L(元) = {fx(l0, digits)}   L(+h) = {fx(lp, digits)}   L(−h) = {fx(lm, digits)}")
    say(f"      中心差分 (L(+h) − L(−h)) / 2h = {fx(cen, digits)}   解析解 = {fnum(analytic)}   差 = {cen - analytic:+.2e}")
    if forward:
        fwd = (lp - l0) / h
        say(f"      前進差分 (L(+h) − L) / h     = {fx(fwd, digits)}   差 = {fwd - analytic:+.2e}")
    return lp, lm, cen


# ---------------------------------------------------------------- 共通の部品

def softmax(s):
    """最後の軸の softmax。-inf（マスク）は 0 になる。"""
    s = np.asarray(s, dtype=float)
    m = np.max(s, axis=-1, keepdims=True)
    e = np.exp(s - m)
    return e / e.sum(axis=-1, keepdims=True)


def layernorm(x):
    """γ=1, β=0, ε=0、母分散（÷d）の LayerNorm（1行）。"""
    mu = x.mean()
    sigma = np.sqrt(((x - mu) ** 2).mean())
    return (x - mu) / sigma, mu, sigma


def layernorm_backward(y, sigma, dy):
    m1 = dy.mean()
    m2 = (dy * y).mean()
    return (dy - m1 - y * m2) / sigma, m1, m2


# ================================================================= D1

def d1():
    say("D1 行列積の連鎖律と形：y = xW、L = ½‖y − t‖²")
    x = np.array([1.0, 0.5])
    W = np.array([[0.5, -1.0], [1.0, 0.5]])
    t = np.array([0.5, 0.0])

    def loss(x, W):
        y = x @ W
        return 0.5 * np.sum((y - t) ** 2)

    section("順伝播")
    y = x @ W
    L = loss(x, W)
    show("x", x)
    show("W", W, ["d0", "d1"])
    show("t", t)
    show("y = xW", y)
    show("L", L)

    section("逆伝播")
    dy = y - t
    dW = np.outer(x, dy)
    dx = dy @ W.T
    show("dy = y − t", dy)
    show("dW = xᵀ dy", dW, ["d0", "d1"])
    show("dx = dy Wᵀ", dx)
    say("  成分ごと: dW[k, j] = x[k]·dy[j]、dx[k] = Σ_j dy[j]·W[k, j]")
    for k in range(2):
        terms = [dy[j] * W[k, j] for j in range(2)]
        say(f"      dx[d{k}] = {fnum(dy[0])}·{fnum(W[k, 0])} + {fnum(dy[1])}·{fnum(W[k, 1])} = "
            f"{fnum(terms[0])} + {fnum(terms[1])} = {fnum(dx[k])}")

    section("形の表（Part 1 の行列積に当てはめる）")
    shapes = [("logits = X2·Wout (6.1)", (3, 4), (4, 5)),
              ("Z = U2·W1 (6.2)", (3, 4), (4, 8)),
              ("F = R·W2 (6.2)", (3, 8), (8, 4)),
              ("Attn = Ocat·Wo (6.4)", (3, 4), (4, 4)),
              ("Q = U1·Wq (6.7)", (3, 4), (4, 4))]
    for name, xs, ws in shapes:
        say(f"      {name:<24} X {xs[0]}×{xs[1]}, W {ws[0]}×{ws[1]} → Y {xs[0]}×{ws[1]}, "
            f"dW = Xᵀ dY: ({xs[1]}×{xs[0]})({xs[0]}×{ws[1]}) = {ws[0]}×{ws[1]}, "
            f"dX = dY Wᵀ: ({xs[0]}×{ws[1]})({ws[1]}×{ws[0]}) = {xs[0]}×{xs[1]}")

    section("検算")
    show("W の行和 W·1", W.sum(1))
    show("W の列和 1ᵀW", W.sum(0))
    show("y の和", y.sum())
    show("x·(W·1)", x @ W.sum(1))
    check("行和チェックサム: Σy = x·(W·1)", close(y.sum(), x @ W.sum(1)))
    show("dx の和", dx.sum())
    show("dy·(1ᵀW)", dy @ W.sum(0))
    check("列和チェックサム: Σdx = dy·(1ᵀW)", close(dx.sum(), dy @ W.sum(0)))
    show("x·dx", x @ dx)
    show("y·dy", y @ dy)
    show("⟨W, dW⟩ = Σ W⊙dW", np.sum(W * dW))
    check("内積の一致: x·dx = y·dy = ⟨W, dW⟩", close(x @ dx, y @ dy) and close(y @ dy, np.sum(W * dW)))
    check("dW の各行は dy に比例（d1 行 = 0.5 × d0 行）", close(dW[1], 0.5 * dW[0]))
    show("dW の総和", dW.sum())
    show("(Σx)(Σdy)", x.sum() * dy.sum())
    check("dW の総和 = (Σx)(Σdy)", close(dW.sum(), x.sum() * dy.sum()))
    gradcheck("dW", lambda Wv: loss(x, Wv), W, dW)
    gradcheck("dx", lambda xv: loss(xv, W), x, dx)
    gradcheck("dy", lambda yv: 0.5 * np.sum((yv - t) ** 2), y, dy)

    section("手で数値微分")
    handdiff("W[d0, d1]", lambda Wv: loss(x, Wv), W, (0, 1), 0.1, dW[0, 1], digits=6, forward=True)
    say(f"      前進差分の誤差 = ½·h·x[d0]² = {fnum(0.5 * 0.1 * x[0] ** 2)}（L は W について2次式）")
    for s in (+1, -1):
        Wp = W.copy()
        Wp[0, 1] += s * 0.1
        show(f"y（W[d0, d1] = {fnum(Wp[0, 1])}）", x @ Wp)
    handdiff("x[d1]", lambda xv: loss(xv, W), x, 1, 0.1, dx[1])
    for s in (+1, -1):
        xp = x.copy()
        xp[1] += s * 0.1
        show(f"y（x[d1] = {fnum(xp[1])}）", xp @ W)

    section("発展：lr = 1 で W だけを1回更新する")
    W1 = W - 1.0 * dW
    y1 = x @ W1
    nx2 = x @ x
    show("W' = W − dW", W1, ["d0", "d1"])
    show("y' = xW'", y1)
    show("‖x‖²", nx2)
    show("y' − y", y1 - y)
    show("−‖x‖²·dy", -nx2 * dy)
    check("y' − y = −lr‖x‖²·dy", close(y1 - y, -nx2 * dy))
    show("y' − t", y1 - t)
    show("1 − lr‖x‖²", 1 - nx2)
    check("y' − t = (1 − lr‖x‖²)(y − t)", close(y1 - t, (1 - nx2) * (y - t)))
    show("L'", loss(x, W1))
    show("(1 − lr‖x‖²)²", (1 - nx2) ** 2)
    show("(1 − lr‖x‖²)²·L", (1 - nx2) ** 2 * L)
    check("L' = (1 − lr‖x‖²)²·L", close(loss(x, W1), (1 - nx2) ** 2 * L))
    show("ちょうど t に届く lr = 1/‖x‖²", 1 / nx2)


# ================================================================= D2

def d2():
    say("D2 分岐：同じ値を2か所で使ったら勾配を足す")
    section("(0) ウォームアップ：f = u·u を a = u, b = u の2本に分ける")
    u = 1.5
    a = b = u
    show("u", u)
    show("f = a·b", a * b)
    show("da = b", b)
    show("db = a", a)
    show("du = da + db", a + b)
    show("2u", 2 * u)
    check("du = 2u", close(a + b, 2 * u))

    section("A 残差：h = xW、y = x + h、L = ½‖y − t‖²")
    x = np.array([1.0, -1.0])
    W = np.array([[0.5, 0.5], [0.0, 1.0]])
    t = np.array([1.0, -0.5])

    def lossA(x, W):
        y = x + x @ W
        return 0.5 * np.sum((y - t) ** 2)

    h = x @ W
    y = x + h
    dy = y - t
    LA = lossA(x, W)
    show("x", x)
    show("W", W, ["d0", "d1"])
    show("t", t)
    show("h = xW", h)
    show("y = x + h", y)
    show("L", LA)
    dh = dy
    dW = np.outer(x, dh)
    dx_res = dy
    dx_W = dh @ W.T
    dx = dx_res + dx_W
    show("dy = y − t", dy)
    show("dh = dy（足し算はコピー）", dh)
    show("dW = xᵀ dh", dW, ["d0", "d1"])
    show("残差経由 dx = dy", dx_res)
    show("W 経由 dx = dh Wᵀ", dx_W)
    show("dx = 残差経由 + W 経由", dx)
    IW = np.eye(2) + W
    show("I + W", IW, ["d0", "d1"])
    show("dy (I + W)ᵀ", dy @ IW.T)
    check("dx = dy (I + W)ᵀ（残差 = 単位行列を足した重み）", close(dx, dy @ IW.T))
    gradcheck("A の dx", lambda xv: lossA(xv, W), x, dx)
    gradcheck("A の dW", lambda Wv: lossA(x, Wv), W, dW)
    gradcheck("A の dy", lambda yv: 0.5 * np.sum((yv - t) ** 2), y, dy)
    handdiff("A の x[d0]", lambda xv: lossA(xv, W), x, 0, 0.1, dx[0])
    for s in (+1, -1):
        xp = x.copy()
        xp[0] += s * 0.1
        show(f"x = {fmt(xp)} のとき h, y", np.stack([xp @ W, xp + xp @ W]), ["h", "y"])

    section("B 2本の枝：q = xWq、k = xWk、L = s = q·k")
    Wq = np.array([[1.0, 0.0], [0.5, 1.0]])
    Wk = np.array([[0.5, 1.0], [0.0, -0.5]])

    def lossB(x, Wq=Wq, Wk=Wk):
        return float((x @ Wq) @ (x @ Wk))

    q = x @ Wq
    k = x @ Wk
    s = q @ k
    show("Wq", Wq, ["d0", "d1"])
    show("Wk", Wk, ["d0", "d1"])
    show("q = xWq", q)
    show("k = xWk", k)
    show("s = q·k", s)
    dq = k
    dk = q
    dx_q = dq @ Wq.T
    dx_k = dk @ Wk.T
    dxB = dx_q + dx_k
    show("dq = k", dq)
    show("dk = q", dk)
    show("q 経由 dx = dq Wqᵀ", dx_q)
    show("k 経由 dx = dk Wkᵀ", dx_k)
    show("dx = q 経由 + k 経由", dxB)
    M = Wq @ Wk.T
    show("M = Wq Wkᵀ", M, ["d0", "d1"])
    show("M + Mᵀ", M + M.T, ["d0", "d1"])
    show("x (M + Mᵀ)", x @ (M + M.T))
    check("dx = x(M + Mᵀ)（s = x M xᵀ と見ても同じ）", close(dxB, x @ (M + M.T)))
    show("x·dx", x @ dxB)
    show("2s", 2 * s)
    check("x·dx = 2s（s は x の2次式）", close(x @ dxB, 2 * s))
    dWq = np.outer(x, dq)
    dWk = np.outer(x, dk)
    show("dWq = xᵀ dq", dWq, ["d0", "d1"])
    show("dWk = xᵀ dk", dWk, ["d0", "d1"])
    gradcheck("B の dx", lambda xv: lossB(xv), x, dxB)
    gradcheck("B の dq", lambda qv: float(qv @ k), q, dq)
    gradcheck("B の dk", lambda kv: float(q @ kv), k, dk)
    gradcheck("B の dWq", lambda Wv: lossB(x, Wq=Wv), Wq, dWq)
    gradcheck("B の dWk", lambda Wv: lossB(x, Wk=Wv), Wk, dWk)
    handdiff("B の x[d0]", lambda xv: lossB(xv), x, 0, 0.1, dxB[0])
    for sgn in (+1, -1):
        xp = x.copy()
        xp[0] += sgn * 0.1
        show(f"x = {fmt(xp)} のとき q, k", np.stack([xp @ Wq, xp @ Wk]), ["q", "k"])


# ================================================================= D3

def d3():
    say("D3 埋め込みと重みの共有：X = E[ids] + P、Y = XW、L = Σ G⊙Y")
    E = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    P = np.array([[0.0, 0.5], [0.5, 0.0], [-0.5, 0.0]])
    W = np.array([[1.0, -0.5], [0.5, 1.0]])
    G = np.array([[1.0, 0.0], [0.0, -1.0], [0.5, 1.0]])
    ids = [0, 1, 0]
    tok = ["a", "b", "c"]
    pos = ["t0", "t1", "t2"]

    def loss(E, P, W):
        X = E[ids] + P
        return float(np.sum(G * (X @ W)))

    section("順伝播")
    X = E[ids] + P
    Y = X @ W
    L = loss(E, P, W)
    show("E", E, tok)
    show("P", P, pos)
    show("W", W, ["d0", "d1"])
    show("G = dY", G, pos)
    show("X = E[ids] + P", X, pos)
    show("Y = XW", Y, pos)
    show("L = Σ G⊙Y", L)

    section("逆伝播")
    terms = [np.outer(X[i], G[i]) for i in range(3)]
    for i, tm in enumerate(terms):
        show(f"位置 t{i} の外積 x_t{i}ᵀ g_t{i}", tm, ["d0", "d1"])
    dW = sum(terms)
    show("dW = 3つの外積の和", dW, ["d0", "d1"])
    check("dW = Xᵀ G（外積の和 = 行列積）", close(dW, X.T @ G))
    dX = G @ W.T
    dP = dX.copy()
    dE = np.zeros_like(E)
    np.add.at(dE, ids, dX)
    show("dX = G Wᵀ", dX, pos)
    show("dP = dX", dP, pos)
    show("dE（ids の行に足し込む）", dE, tok)

    section("検算")
    show("dE の列和", dE.sum(0))
    show("dX の列和", dX.sum(0))
    check("dE の全行の和 = dX の全行の和 = dP の全行の和", close(dE.sum(0), dX.sum(0)) and close(dP.sum(0), dX.sum(0)))
    check("使われないトークン c の dE 行 = 0", close(dE[2], 0))
    check("a の dE 行 = dX[t0] + dX[t2]", close(dE[0], dX[0] + dX[2]))
    check("b の dE 行 = dX[t1] = dP[t1]", close(dE[1], dP[1]))
    show("⟨W, dW⟩", np.sum(W * dW))
    show("⟨X, dX⟩", np.sum(X * dX))
    check("⟨W, dW⟩ = ⟨X, dX⟩ = L（L は W にも X にも1次）", close(np.sum(W * dW), L) and close(np.sum(X * dX), L))
    gradcheck("dE", lambda Ev: loss(Ev, P, W), E, dE)
    gradcheck("dP", lambda Pv: loss(E, Pv, W), P, dP)
    gradcheck("dW", lambda Wv: loss(E, P, Wv), W, dW)
    gradcheck("dX", lambda Xv: float(np.sum(G * (Xv @ W))), X, dX)

    section("手で数値微分")
    handdiff("E[a, d1]", lambda Ev: loss(Ev, P, W), E, (0, 1), 0.1, dE[0, 1])
    Ep = E.copy()
    Ep[0, 1] += 0.1
    show("E[a, d1] = 0.1 のときの X", Ep[ids] + P, pos)
    show("そのときの Y", (Ep[ids] + P) @ W, pos)
    handdiff("W[d0, d0]", lambda Wv: loss(E, P, Wv), W, (0, 0), 0.1, dW[0, 0])
    show("位置ごとの寄与 X[t, d0]·G[t, d0]", X[:, 0] * G[:, 0])

    section("SGD（lr = 1）と Part 1 の近道 X' = X − 2·lr·dX")
    E1 = E - dE
    P1 = P - dP
    X1 = E1[ids] + P1
    show("E' = E − dE", E1, tok)
    show("P' = P − dP", P1, pos)
    show("X' = E'[ids] + P'", X1, pos)
    show("近道 X − 2dX", X - 2 * dX, pos)
    show("正しい式 X − dP − dE[ids]", X - dP - dE[ids], pos)
    check("X' = X − dP − dE[ids]", close(X1, X - dP - dE[ids]))
    check("近道は t1 では正しく、t0, t2 では外れる",
          close(X1[1], (X - 2 * dX)[1]) and not close(X1[0], (X - 2 * dX)[0]) and not close(X1[2], (X - 2 * dX)[2]))


# ================================================================= D4

def d4():
    say("D4 ReLU の門と死んだユニット：Z = XW1、R = ReLU(Z)、F = RW2、L = ½‖F − T‖²")
    X = np.array([[1.0, 0.5], [-0.5, 1.0]])
    W1 = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0]])
    W2 = np.array([[1.0, 0.0], [0.5, -1.0], [1.0, -0.5]])
    T = np.array([[1.0, 0.5], [1.0, -1.0]])
    pos = ["t0", "t1"]
    units = ["h0", "h1", "h2"]

    def loss(X, W1, W2):
        R = np.maximum(X @ W1, 0.0)
        return 0.5 * float(np.sum((R @ W2 - T) ** 2))

    section("順伝播")
    Z = X @ W1
    R = np.maximum(Z, 0.0)
    F = R @ W2
    L = loss(X, W1, W2)
    show("X", X, pos)
    show("W1", W1, ["d0", "d1"])
    show("W2", W2, units)
    show("T", T, pos)
    show("Z = XW1", Z, pos)
    show("R = ReLU(Z)", R, pos)
    show("F = RW2", F, pos)
    show("L", L)
    show("X の行和（Z の h2 列 = −行和）", X.sum(1))
    check("Z に 0 ちょうどの成分がない（ReLU′(0) を避ける）", np.min(np.abs(Z)) > 0.1)
    check("h2 は2位置とも Z < 0（死んだユニット）", np.all(Z[:, 2] < 0))

    section("逆伝播")
    dF = F - T
    dW2 = R.T @ dF
    dR = dF @ W2.T
    gate = (Z > 0).astype(float)
    dZ = dR * gate
    dW1 = X.T @ dZ
    dX = dZ @ W1.T
    show("dF = F − T", dF, pos)
    show("dW2 = Rᵀ dF", dW2, units)
    show("dR = dF W2ᵀ", dR, pos)
    show("門 [Z > 0]", gate, pos)
    show("dZ = dR ⊙ [Z > 0]", dZ, pos)
    show("dW1 = Xᵀ dZ", dW1, ["d0", "d1"])
    show("dX = dZ W1ᵀ", dX, pos)

    section("検算")
    check("dR の h2 列は 0 でない（勾配は門の手前まで来ている）", np.all(np.abs(dR[:, 2]) > 0))
    check("dZ の h2 列 = 0、dW1 の h2 列 = 0、dW2 の h2 行 = 0", close(dZ[:, 2], 0) and close(dW1[:, 2], 0) and close(dW2[2], 0))
    check("W2 の h2 行は 0 でない", np.all(np.abs(W2[2]) > 0))
    show("W1 の列和 1ᵀW1", W1.sum(0))
    show("dX の行和", dX.sum(1))
    show("dZ·(1ᵀW1)（行ごと）", dZ @ W1.sum(0))
    check("dX の行和 = dZ[t]·(1ᵀW1)", close(dX.sum(1), dZ @ W1.sum(0)))
    gradcheck("dW1", lambda Wv: loss(X, Wv, W2), W1, dW1)
    gradcheck("dW2", lambda Wv: loss(X, W1, Wv), W2, dW2)
    gradcheck("dX", lambda Xv: loss(Xv, W1, W2), X, dX)
    gradcheck("dZ", lambda Zv: 0.5 * float(np.sum((np.maximum(Zv, 0) @ W2 - T) ** 2)), Z, dZ)
    gradcheck("dR", lambda Rv: 0.5 * float(np.sum((Rv @ W2 - T) ** 2)), R, dR)

    section("手で数値微分")
    handdiff("W1[d0, h2]（死んだユニット）", lambda Wv: loss(X, Wv, W2), W1, (0, 2), 0.1, dW1[0, 2])
    for s in (+1, -1):
        Wp = W1.copy()
        Wp[0, 2] += s * 0.1
        show(f"W1[d0, h2] = {fnum(Wp[0, 2])} のとき Z の h2 列", (X @ Wp)[:, 2])
    handdiff("W1[d0, h1]", lambda Wv: loss(X, Wv, W2), W1, (0, 1), 0.1, dW1[0, 1])
    for s in (+1, -1):
        Wp = W1.copy()
        Wp[0, 1] += s * 0.1
        Zp = X @ Wp
        show(f"W1[d0, h1] = {fnum(Wp[0, 1])} のとき Z の h1 列", Zp[:, 1])
        show(f"  そのときの F", np.maximum(Zp, 0) @ W2, pos)
    say("  h を大きくして折れ目（Z = 0）をまたぐと：")
    _, _, cen06 = handdiff("W1[d0, h1]", lambda Wv: loss(X, Wv, W2), W1, (0, 1), 0.6, dW1[0, 1])
    for s in (+1, -1):
        Wp = W1.copy()
        Wp[0, 1] += s * 0.6
        show(f"W1[d0, h1] = {fnum(Wp[0, 1])} のとき Z の h1 列", (X @ Wp)[:, 1])
    check("h = 0.6 では折れ目をまたぐので中心差分が解析解からずれる", abs(cen06 - dW1[0, 1]) > 0.01)

    section("SGD（lr = 1）のあと")
    W1n = W1 - dW1
    W2n = W2 - dW2
    Zn = X @ W1n
    show("W1' = W1 − dW1", W1n, ["d0", "d1"])
    show("W2' = W2 − dW2", W2n, units)
    show("Z' = XW1'", Zn, pos)
    check("W1 の h2 列は変わらない → Z' の h2 列も同じで、まだ負", close(W1n[:, 2], W1[:, 2]) and np.all(Zn[:, 2] < 0))
    for lr in (0.25, 0.5, 1.0):
        Wa, Wb = W1 - lr * dW1, W2 - lr * dW2
        show(f"寄り道: lr = {lr:g} で両方更新したときの L'", loss(X, Wa, Wb))
        check(f"lr = {lr:g}: h2 は死んだまま", np.all((X @ Wa)[:, 2] < 0))


# ================================================================= D5

def d5():
    say("D5 softmax + 交差エントロピー：2行（2位置）、3クラス、損失は2行の平均")
    Zl = np.array([[1.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    y = [1, 2]
    Y = np.eye(3)[y]
    rows = ["行a", "行b"]

    def nll_rows(Zv):
        m = Zv.max(1, keepdims=True)
        lse = (m + np.log(np.exp(Zv - m).sum(1, keepdims=True)))[:, 0]
        return lse - Zv[np.arange(len(y)), y]

    def loss(Zv):
        return float(nll_rows(Zv).mean())

    section("順伝播")
    show("z", Zl, rows)
    show("正解クラス", np.array(y, float))
    ea = np.exp(Zl[0])
    show("行a の exp(z) = (e¹, e⁰, e⁻¹)", ea)
    show("行a の和", ea.sum())
    show("ln(行a の和) = logsumexp", np.log(ea.sum()))
    p = softmax(Zl)
    show("p = softmax(z)", p, rows)
    nll = nll_rows(Zl)
    show("各行の nll = logsumexp − z[正解]", nll)
    show("ln 3", np.log(3.0))
    L = loss(Zl)
    show("L = 平均", L)

    section("逆伝播")
    dz = (p - Y) / 2
    show("p − y", p - Y, rows)
    show("dz = (p − y)/2", dz, rows)

    section("検算")
    show("p の行和", p.sum(1))
    show("dz の行和", dz.sum(1))
    check("p の行和 = 1", close(p.sum(1), 1))
    check("dz の行和 = 0", close(dz.sum(1), 0))
    check("dz で負になるのは正解の成分だけ", np.all((dz < 0) == (Y > 0)))
    gradcheck("dz（平均の損失）", loss, Zl, dz)
    gradcheck("行a だけの損失の勾配 p − y", lambda zv: float(nll_rows(np.stack([zv, Zl[1]]))[0]), Zl[0], p[0] - Y[0])

    section("ずらしても変わらない")
    p5 = softmax(Zl[0] + 5)
    pbig = softmax(np.array([101.0, 100.0, 99.0]))
    show("softmax(z_a + 5)", p5)
    show("softmax(101, 100, 99)", pbig)
    check("softmax(z + c) = softmax(z)", close(p5, p[0]) and close(pbig, p[0]))

    section("手で数値微分（行a だけの損失 L_a で）")
    for name, v in (("e^1", np.e), ("e^−1", np.exp(-1)), ("e^0.1", np.exp(0.1)), ("e^−0.1", np.exp(-0.1))):
        say(f"  {name} = {v:.6f}")
    for zc, name in ((0.1, "+h"), (-0.1, "−h")):
        zz = Zl[0].copy()
        zz[1] = zc
        ssum = np.exp(zz).sum()
        say(f"  z[正解] = {fnum(zc)}（{name}）: 和 = {ssum:.6f}  ln(和) = {np.log(ssum):.6f}  L_a = {np.log(ssum) - zc:.6f}")
    handdiff("z_a[1]（正解の logit）", lambda zv: float(nll_rows(np.stack([zv, Zl[1]]))[0]), Zl[0], 1, 0.1, p[0, 1] - 1)

    section("logit を1回動かす（行a、L_a について z ← z − 1·(p − y)）")
    za1 = Zl[0] - (p[0] - Y[0])
    pa1 = softmax(za1)
    show("z' = z − (p − y)", za1)
    show("z' − z = −(p − y)", za1 - Zl[0])
    show("p'", pa1)
    show("L_a'", float(nll_rows(np.stack([za1, Zl[1]]))[0]))
    show("L_a（元）", nll[0])
    check("正解の確率が上がる", pa1[1] > p[0, 1])


# ================================================================= D6

def d6():
    say("D6 softmax のヤコビアン：a = softmax(s)、da から ds = a ⊙ (da − Σ a·da)")

    def sm_back(a, da):
        r = float(a @ da)
        return a * (da - r), r

    section("ケース1：a = (0.5, 0.25, 0.25)")
    a = np.array([0.5, 0.25, 0.25])
    da = np.array([1.0, -1.0, 0.5])
    ds, r = sm_back(a, da)
    show("a", a)
    show("da", da)
    show("r = Σ a·da", r)
    show("da − r", da - r)
    show("ds = a ⊙ (da − r)", ds)
    J = np.diag(a) - np.outer(a, a)
    show("J = diag(a) − aᵀa（J[i, j] = ∂a_j/∂s_i）", J, ["s0", "s1", "s2"])
    show("J の行和", J.sum(1))
    show("J da", J @ da)
    check("ds = J da", close(J @ da, ds))
    check("J の行和 = 0、J は対称", close(J.sum(1), 0) and close(J, J.T))
    check("ds の和 = 0", close(ds.sum(), 0))
    ds2, r2 = sm_back(a, da + 2)
    show("da + 2 での r", r2)
    show("da + 2 での ds", ds2)
    check("da に定数を足しても ds は変わらない", close(ds2, ds))
    s = np.log(a) - np.log(a[1])
    show("この a を作る s（s1 = s2 = 0）: s0 = ln 2", s[0])
    check("softmax(ln 2, 0, 0) = a", close(softmax(s), a))
    lossf = lambda sv: float(da @ softmax(sv))  # noqa: E731
    gradcheck("ケース1 の ds", lossf, s, ds)
    show("L = da·a（元）", lossf(s))
    for h in (+0.1, -0.1):
        sp = s.copy()
        sp[0] += h
        ap_ = softmax(sp)
        say(f"  s0 = ln 2 {'+' if h > 0 else '−'} 0.1: 分子 2e^(±0.1) = {2 * np.exp(h):.6f}  分母 = {2 * np.exp(h) + 2:.6f}  "
            f"a = [{', '.join(f'{v:.6f}' for v in ap_)}]  L = {lossf(sp):.6f}")
    say(f"  e^0.1 = {np.exp(0.1):.6f}   e^−0.1 = {np.exp(-0.1):.6f}")
    handdiff("s0", lossf, s, 0, 0.1, ds[0])

    section("ケース2：マスクのある行 a = (0.4, 0.6, 0)、da2 は何でもよい")
    am = np.array([0.4, 0.6, 0.0])
    dam = np.array([0.5, -0.5, 7.0])
    dsm, rm = sm_back(am, dam)
    show("a", am)
    show("da", dam)
    show("r", rm)
    show("ds", dsm)
    show("2要素の公式 a0·a1·(da0 − da1)", am[0] * am[1] * (dam[0] - dam[1]))
    check("ds0 = a0 a1 (da0 − da1)、ds1 = −ds0、ds2 = 0",
          close(dsm[0], am[0] * am[1] * (dam[0] - dam[1])) and close(dsm[1], -dsm[0]) and close(dsm[2], 0))
    sm_ = np.array([np.log(0.4), np.log(0.6), -np.inf])

    def lossm(sv):
        return float(dam @ softmax(sv))

    g_un = numgrad(lambda v: lossm(np.array([v[0], v[1], -np.inf])), sm_[:2])
    check(f"数値微分（h=1e-6）と一致: ケース2 の ds（マスクしない2成分） max|差| = {np.max(np.abs(g_un - dsm[:2])):.1e}",
          close(g_un, dsm[:2], 1e-6))
    show("マスク位置のスコアを有限の −30 にしたときの ds2（数値微分）",
         numgrad(lambda v: lossm(np.array([sm_[0], sm_[1], v[0]])), np.array([-30.0]))[0])

    section("ケース3：要素1つ（t0 の行）")
    a1 = np.array([1.0])
    da1 = np.array([0.8])
    ds1, r1 = sm_back(a1, da1)
    show("a", a1)
    show("da", da1)
    show("r", r1)
    show("ds", ds1)
    check("要素1つの softmax の ds = 0", close(ds1, 0))
    gradcheck("ケース3 の ds", lambda sv: float(da1 @ softmax(sv)), np.array([0.3]), ds1)

    section("ケース4：飽和した行 a = (0.98, 0.01, 0.01)、da はケース1と同じ")
    asat = np.array([0.98, 0.01, 0.01])
    dss, rs = sm_back(asat, da)
    show("r", rs)
    show("ds", dss)
    show("max|ds|（ケース1）", np.max(np.abs(ds)))
    show("max|ds|（ケース4）", np.max(np.abs(dss)))
    show("a_i(1 − a_i)（J の対角）", asat * (1 - asat))
    grid = np.linspace(0, 1, 1001)
    show("a(1 − a) の最大値（a = 0.5 のとき）", np.max(grid * (1 - grid)))
    check("ds の和 = 0", close(dss.sum(), 0))
    ssat = np.log(asat)
    gradcheck("ケース4 の ds", lambda sv: float(da @ softmax(sv)), ssat, dss)


# ================================================================= D7

def attention(Q, K, V, mask=True):
    raw = Q @ K.T
    S = raw / SQ2
    if mask:
        S = np.where(np.triu(np.ones_like(S, dtype=bool), 1), -np.inf, S)
    A = softmax(S)
    return raw, S, A, A @ V


def d7():
    say("D7 1ヘッド Attention（2トークン、d = 2、因果マスク）：L = Σ G⊙O")
    Q = np.array([[0.5, -1.0], [1.0, 1.0]])
    K = np.array([[1.0, -1.0], [0.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 2.0]])
    G = np.array([[2.0, 0.0], [1.0, -1.0]])
    pos = ["t0", "t1"]
    keys = ["k0", "k1"]

    def loss(Q, K, V, mask=True):
        return float(np.sum(G * attention(Q, K, V, mask)[3]))

    section("順伝播")
    raw, S, A, O = attention(Q, K, V)
    show("Q", Q, pos)
    show("K", K, keys)
    show("V", V, keys)
    show("G = dO", G, pos)
    show("素点 QKᵀ（マスク前）", raw, pos)
    show("S = QKᵀ/√2（マスク後）", S, pos)
    show("1/√2", 1 / SQ2)
    say(f"  1/√2（Part 1 のルール2 の5桁） = {1 / SQ2:.5f}")
    say(f"  e^(1/√2) = {np.exp(1 / SQ2):.6f}")
    say(f"  A[t1, k0] = 1/(1 + e^(1/√2)) = {A[1, 0]:.6f}")
    show("A = softmax(S)", A, pos)
    show("O = AV", O, pos)
    L = loss(Q, K, V)
    show("L = Σ G⊙O", L)
    show("3·A[t1, k0]", 3 * A[1, 0])
    check("この G では L = 3·A[t1, k0]（G[t0]·O[t0] = 2、G[t1]·O[t1] = 3·A[t1,k0] − 2）", close(L, 3 * A[1, 0]))

    section("逆伝播")
    dO = G
    dA = dO @ V.T
    dA = np.where(np.triu(np.ones_like(dA, dtype=bool), 1), 0.0, dA)
    r = np.sum(A * dA, axis=1)
    dS = A * (dA - r[:, None])
    dQ = dS @ K / SQ2
    dK = dS.T @ Q / SQ2
    dV = A.T @ dO
    show("dA = dO Vᵀ（マスク位置は 0）", dA, pos)
    show("r = 行ごとの Σ A·dA", r)
    show("dS = A ⊙ (dA − r)", dS, pos)
    show("2要素の公式 A[t1,k0]·A[t1,k1]·(dA[t1,k0] − dA[t1,k1])", A[1, 0] * A[1, 1] * (dA[1, 0] - dA[1, 1]))
    show("dS K（÷√2 の前）", dS @ K, pos)
    show("dQ = dS K / √2", dQ, pos)
    show("dSᵀ Q（÷√2 の前）", dS.T @ Q, keys)
    show("dK = dSᵀ Q / √2", dK, keys)
    show("dV = Aᵀ dO", dV, keys)

    section("検算")
    check("dS の各行の和 = 0", close(dS.sum(1), 0))
    show("G[t]·O[t]（行ごと）", np.sum(G * O, axis=1))
    check("r_t = dO[t]·O[t]（行 t の損失への寄与と同じ）", close(r, np.sum(G * O, axis=1)))
    check("dS の t0 行 = 0 → dQ の t0 行 = 0", close(dS[0], 0) and close(dQ[0], 0))
    show("dK の列和", dK.sum(0))
    check("dK の列和 = 0", close(dK.sum(0), 0))
    show("dV の列和", dV.sum(0))
    show("dO の列和", dO.sum(0))
    check("dV の列和 = dO の列和（A の行和が1）", close(dV.sum(0), dO.sum(0)))
    check("dK の2行はどちらも q1 = (1, 1) に比例", close(dK[0], dK[0, 0] * Q[1]) and close(dK[1], dK[1, 0] * Q[1]))
    check("dV の k1 行 = A[t1, k1]·dO[t1]（k1 を見たのは t1 だけ）", close(dV[1], A[1, 1] * dO[1]))
    gradcheck("dQ", lambda Qv: loss(Qv, K, V), Q, dQ)
    gradcheck("dK", lambda Kv: loss(Q, Kv, V), K, dK)
    gradcheck("dV", lambda Vv: loss(Q, K, Vv), V, dV)
    gradcheck("dS（スコアを直接動かす）",
              lambda Sv: float(np.sum(G * (softmax(np.where(np.triu(np.ones((2, 2), bool), 1), -np.inf, Sv)) @ V))),
              np.where(np.isfinite(S), S, 0.0), dS)

    section("手で数値微分")
    for h in (+0.1, -0.1):
        Qp = Q.copy()
        Qp[1, 0] += h
        rp, Sp, Ap, Op = attention(Qp, K, V)
        diff = Sp[1, 1] - Sp[1, 0]
        say(f"  Q[t1, d0] = {fnum(Qp[1, 0])}: 素点 t1 行 = {fmt(rp[1])}  S[t1,k1] − S[t1,k0] = {diff:.6f}  "
            f"e^差 = {np.exp(diff):.6f}  A[t1, k0] = {Ap[1, 0]:.6f}  O[t1] = [{Op[1, 0]:.6f}, {Op[1, 1]:.6f}]  "
            f"L = {loss(Qp, K, V):.6f}")
    handdiff("Q[t1, d0]", lambda Qv: loss(Qv, K, V), Q, (1, 0), 0.1, dQ[1, 0])
    handdiff("Q[t0, d0]（`<s>` 側のクエリ）", lambda Qv: loss(Qv, K, V), Q, (0, 0), 0.1, dQ[0, 0])
    c = np.array([0.1, 0.1])
    show("全キーに c = (0.1, 0.1) を足したときの L", loss(Q, K + c, V))
    check("全キーに同じベクトルを足しても L は変わらない", close(loss(Q, K + c, V), L))

    section("発展：マスクを外すと")
    rawn, Sn, An, On = attention(Q, K, V, mask=False)
    show("S（マスクなし）", Sn, pos)
    show("A（マスクなし）", An, pos)
    dAn = dO @ V.T
    rn = np.sum(An * dAn, axis=1)
    dSn = An * (dAn - rn[:, None])
    dQn = dSn @ K / SQ2
    show("dA（マスクなし）", dAn, pos)
    show("dS（マスクなし）", dSn, pos)
    show("dQ（マスクなし）", dQn, pos)
    gradcheck("dQ（マスクなし）", lambda Qv: loss(Qv, K, V, mask=False), Q, dQn)
    check("マスクを外すと dQ の t0 行は 0 でなくなる", np.max(np.abs(dQn[0])) > 0.01)


# ================================================================= D8

def d8():
    say("D8 LayerNorm（γ=1, β=0, ε=0、4次元の1行）")
    x = np.array([4.0, -3.0, 1.0, 0.0])
    dy = np.array([1.0, 0.0, -0.5, 0.5])

    section("順伝播")
    y, mu, sigma = layernorm(x)
    show("x", x)
    show("μ", mu)
    show("x − μ", x - mu)
    show("(x − μ)²", (x - mu) ** 2)
    show("σ²", sigma ** 2)
    show("σ", sigma)
    show("y = (x − μ)/σ", y)
    show("mean(y)", y.mean())
    show("mean(y²)", (y ** 2).mean())
    check("mean(y) = 0、mean(y²) = 1", close(y.mean(), 0) and close((y ** 2).mean(), 1))
    show("Σy²（= d）", np.sum(y ** 2))
    show("(1,1,1,1)·y", y.sum())
    x2 = 2 * x + 3
    y2, mu2, sigma2 = layernorm(x2)
    show("x'' = 2x + 3", x2)
    show("μ''", mu2)
    show("σ''", sigma2)
    show("LN(2x + 3)", y2)
    check("LN(2x + 3) = LN(x)", close(y2, y))
    yneg, _, _ = layernorm(-x)
    show("LN(−x)", yneg)
    check("LN(−x) = −LN(x)（負の倍率では符号が反転）", close(yneg, -y))

    section("逆伝播（L = dy·LN(x) とする）")
    dx, m1, m2 = layernorm_backward(y, sigma, dy)
    show("dy", dy)
    show("m1 = mean(dy)", m1)
    show("dy ⊙ y", dy * y)
    show("m2 = mean(dy ⊙ y)", m2)
    show("dy − m1", dy - m1)
    show("y·m2", y * m2)
    show("dy − m1 − y·m2", dy - m1 - y * m2)
    show("dx = (dy − m1 − y·m2)/σ", dx)

    section("途中の微分（x0 を動かす）")
    dmu = 0.25
    dsig = y[0] / 4
    col0 = (np.eye(4)[0] - 0.25 - y[0] * y / 4) / sigma
    show("∂μ/∂x0 = 1/4", dmu)
    show("∂σ/∂x0 = y0/4", dsig)
    show("∂y/∂x0 = (e0 − 1/4 − y0·y/4)/σ", col0)
    show("dy·(∂y/∂x0) = dx0", dy @ col0)
    check("dy·(∂y/∂x0) = dx0", close(dy @ col0, dx[0]))
    check("∂y/∂x0 の和 = 0", close(col0.sum(), 0))
    gradcheck("∂σ/∂x", lambda xv: layernorm(xv)[2], x, y / 4)
    g_col = np.array([numgrad(lambda xv: layernorm(xv)[0][j], x)[0] for j in range(4)])
    check(f"数値微分（h=1e-6）と一致: ∂y/∂x0  max|差| = {np.max(np.abs(g_col - col0)):.1e}", close(g_col, col0, 1e-6))

    section("検算")
    show("Σ dx", dx.sum())
    show("dx·y", dx @ y)
    check("Σ dx = 0", close(dx.sum(), 0))
    check("dx·y = 0", close(dx @ y, 0))
    lossf = lambda xv: float(dy @ layernorm(xv)[0])  # noqa: E731
    gradcheck("dx", lossf, x, dx)
    dx2, _, _ = layernorm_backward(y2, sigma2, dy)
    show("入力が 2x + 3 のときの dx''", dx2)
    show("dx/2", dx / 2)
    check("dx'' = dx/2（σ'' = 2σ）", close(dx2, dx / 2))
    gradcheck("dx''", lambda xv: float(dy @ layernorm(xv)[0]), x2, dx2)
    for name, v in (("dy = (1, 1, 1, 1)", np.ones(4)), ("dy = y", y), ("dy = 3 + 2y", 3 + 2 * y)):
        dxv, _, _ = layernorm_backward(y, sigma, v)
        show(f"{name} のときの dx", dxv)
        check(f"{name} → dx = 0", close(dxv, 0))

    section("手で数値微分（x0 を ±0.1）")
    for h in (+0.1, -0.1):
        xp = x.copy()
        xp[0] += h
        yp, mup, sp = layernorm(xp)
        say(f"  x0 = {fnum(xp[0])}: μ = {mup:.6f}  σ² = {sp ** 2:.6f}  σ = {sp:.6f}  "
            f"y = [{', '.join(f'{v:.6f}' for v in yp)}]  L = {lossf(xp):.6f}")
    handdiff("x0", lossf, x, 0, 0.1, dx[0])
    handdiff("σ（x0 を動かす）", lambda xv: layernorm(xv)[2], x, 0, 0.1, dsig)


# ================================================================= D9

def d9():
    say("D9 SGD 1ステップ：1次予測 −η‖g‖² と実際の ΔL")
    section("A 2次関数 L(θ) = ½(θ0² + 4·θ1²)")
    hdiag = np.array([1.0, 4.0])
    th = np.array([1.0, 0.5])

    def LA(t):
        return 0.5 * float(np.sum(hdiag * t ** 2))

    g = hdiag * th
    g2 = float(g @ g)
    gHg = float(g @ (hdiag * g))
    show("θ", th)
    show("L(θ)", LA(th))
    show("g = (θ0, 4θ1)", g)
    show("‖g‖²", g2)
    show("gᵀHg（H = diag(1, 4)）", gHg)
    gradcheck("A の g", LA, th, g)
    say("  η     θ'                L'         実際 ΔL     予測 −η‖g‖²   実際/予測   2次まで −η‖g‖² + ½η²gᵀHg")
    for eta in (0.1, 0.25, 0.5, 0.6):
        t1 = th - eta * g
        dl = LA(t1) - LA(th)
        pred = -eta * g2
        sec = pred + 0.5 * eta ** 2 * gHg
        say(f"  {eta:<5g} {fmt(t1):<17} {fnum(LA(t1)):<10} {fnum(dl):<11} {fnum(pred):<13} {fnum(dl / pred):<11} {fnum(sec)}")
        check(f"η = {eta:g}: 実際 ΔL = −η‖g‖² + ½η²gᵀHg（2次関数なので厳密）", close(dl, sec))
    check("η = 0.6 では L が上がる", LA(th - 0.6 * g) > LA(th))
    show("η = 0.6 の1次予測で求めた L' = L − η‖g‖²", LA(th) - 0.6 * g2)
    show("θ1 = ±0.5 のときの θ1 の分の損失 ½·4·θ1²", 0.5 * 4 * 0.5 ** 2)
    show("1ステップで最も下がる η* = ‖g‖²/gᵀHg", g2 / gHg)
    show("そのときの ΔL = −‖g‖⁴/(2gᵀHg)", -g2 ** 2 / (2 * gHg))
    check("η* での ΔL が表のどの η よりも小さい",
          LA(th - g2 / gHg * g) - LA(th) <= min(LA(th - e * g) - LA(th) for e in (0.1, 0.25, 0.5, 0.6)) + 1e-12)
    show("各方向の倍率 1 − η·h_i（η = 0.25）", 1 - 0.25 * hdiag)
    show("各方向の倍率 1 − η·h_i（η = 0.5）", 1 - 0.5 * hdiag)
    show("各方向の倍率 1 − η·h_i（η = 0.6）", 1 - 0.6 * hdiag)
    show("発散しない η の上限 2/h_max", 2 / hdiag.max())
    for eta in (0.5, 0.6):
        t = th.copy()
        seq_t, seq_l = [t.copy()], [LA(t)]
        for _ in range(3):
            t = t - eta * hdiag * t
            seq_t.append(t.copy())
            seq_l.append(LA(t))
        say(f"  η = {eta:g} で3ステップ: θ1 = {fmt([s[1] for s in seq_t])}  L = {fmt(seq_l)}")
    handdiff("θ0", LA, th, 0, 0.1, g[0])

    section("B 2つの重みの積 L(w) = ½(w0·w1 − 1)²")
    w = np.array([0.5, 0.5])

    def LB(v):
        return 0.5 * float((v[0] * v[1] - 1) ** 2)

    rr = w[0] * w[1] - 1
    gB = rr * np.array([w[1], w[0]])
    gB2 = float(gB @ gB)
    show("w", w)
    show("r = w0·w1 − 1", rr)
    show("L(w)", LB(w))
    show("g = r·(w1, w0)", gB)
    show("‖g‖²", gB2)
    gradcheck("B の g", LB, w, gB)
    say("  η     w'                 w0'·w1'      L'          実際 ΔL      予測 −η‖g‖²   実際/予測")
    ratios = {}
    for eta in (0.25, 0.5, 1.0, 2.0):
        w1 = w - eta * gB
        dl = LB(w1) - LB(w)
        pred = -eta * gB2
        ratios[eta] = dl / pred
        say(f"  {eta:<5g} {fmt(w1):<18} {fnum(w1[0] * w1[1]):<12} {LB(w1):.6f}   {dl:+.6f}    {fnum(pred):<13} {ratios[eta]:.4f}")
    check("小さい η（0.25, 0.5）では 実際/予測 > 1", ratios[0.25] > 1 and ratios[0.5] > 1)
    check("η = 1, 2 では 実際/予測 < 1", ratios[1.0] < 1 and ratios[2.0] < 1)
    HB = np.outer([w[1], w[0]], [w[1], w[0]]) + rr * np.array([[0.0, 1.0], [1.0, 0.0]])
    show("H = ∇r∇rᵀ + r·[[0, 1], [1, 0]]", HB, ["w0", "w1"])
    gHgB = float(gB @ HB @ gB)
    show("gᵀHg", gHgB)
    ev, evec = np.linalg.eigh(HB)
    show("H の固有値", ev)
    show("固有値 −0.25 の固有ベクトル（√2 倍）", evec[:, 0] * SQ2 * np.sign(evec[0, 0]))
    show("g の向き g/|g0|", gB / abs(gB[0]))
    check("勾配の方向の曲率 gᵀHg < 0（下に凸でない）", gHgB < 0)
    Hnum = np.array([numgrad(lambda v: (lambda gg: gg[i])(np.array([(v[0] * v[1] - 1) * v[1], (v[0] * v[1] - 1) * v[0]])), w)
                     for i in range(2)])
    check(f"H が数値微分と一致  max|差| = {np.max(np.abs(Hnum - HB)):.1e}", close(Hnum, HB, 1e-6))
    handdiff("w0", LB, w, 0, 0.1, gB[0])
    for h in (+0.1, -0.1):
        wp = w.copy()
        wp[0] += h
        show(f"w0 = {fnum(wp[0])} のとき w0·w1, L", np.array([wp[0] * wp[1], LB(wp)]))


# ================================================================= main

DRILLS = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5, "D6": d6, "D7": d7, "D8": d8, "D9": d9}


def main():
    global QUIET, CURRENT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="D1〜D9 のどれか1つだけ")
    ap.add_argument("--quiet", action="store_true", help="検査結果だけ表示")
    args = ap.parse_args()
    QUIET = args.quiet
    names = [args.only.upper()] if args.only else list(DRILLS)
    for n in names:
        if n not in DRILLS:
            sys.exit(f"--only は {', '.join(DRILLS)} のどれか")
    for n in names:
        CURRENT = n
        say("\n" + "=" * 78)
        if QUIET:
            print(f"{n}")
        DRILLS[n]()
    ng = [(d, nm) for d, nm, ok in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"検査 {len(RESULTS)} 件中 {len(RESULTS) - len(ng)} 件 OK")
    if ng:
        for d, nm in ng:
            print(f"  NG {d}: {nm}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
