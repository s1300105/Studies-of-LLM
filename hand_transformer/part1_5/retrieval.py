"""Part 1.5 間をあけた想起：小さな Transformer を白紙から1周する問題の正解実装（NumPy）。

Part 1 と同じ部品（Pre-LN・因果マスク付き Attention・残差・出力層・平均の交差エントロピー・SGD）を、
もっと小さな大きさで使う。手計算と同じ順序・同じ記号で、順伝播 → 損失 → 逆伝播 → SGD 更新 → 再順伝播 を行う。

使い方（hand_transformer/ で）:
  python3 part1_5/retrieval.py               # 全チェックポイント（A1〜C4）
  python3 part1_5/retrieval.py --only B5     # 1つだけ
  python3 part1_5/retrieval.py --section B   # 逆伝播だけ
  python3 part1_5/retrieval.py --check       # 不変量の検査と数値微分（すべて通れば ALL CHECKS PASSED）
  python3 part1_5/retrieval.py --gradcheck   # 数値微分との差だけ
  python3 part1_5/retrieval.py --explain     # 説明し直しの問いで使う数値（学習率と1次予測の比など）
  python3 part1_5/retrieval.py --markdown    # 解答の表を Markdown で（worksheet.md の答えはこの出力）
  python3 part1_5/retrieval.py --blank       # 記入用の空欄表を Markdown で

モデル（T = 2, d = 3, 1ヘッド d_k = 3, FFN なし, 出力射影 Wo なし）:
  X0 = E[ids] + P
  U  = LN(X0)                       # γ = 1, β = 0, ε = 0, 母分散（÷3）
  Q, K, V = U Wq, U Wk, U Wv
  S  = Q Kᵀ / √3（因果マスク）, A = softmax(S), O = A V
  X1 = X0 + O                       # 残差
  logits = X1 Wout, p = softmax(logits), L = (1/2) Σ_t −log p[t, y_t]
  θ' = θ − lr·dθ（lr = 1、E, P, Wq, Wk, Wv, Wout の51個すべて）
"""
import argparse
import sys

import numpy as np

VOCAB = ["<s>", "雨", "降る"]
INP = [0, 1]          # <s> 雨
TGT = [1, 2]          # 雨 降る
T, D, V = 2, 3, 3
LR = 1.0
MASK = np.triu(np.ones((T, T), dtype=bool), 1)   # True = 未来（見てはいけない）

PARAMS0 = {
    "E": [[1.0, 0.5, 0.0],        # <s>
          [-0.5, 1.0, 0.5],       # 雨
          [0.0, -1.0, 1.0]],      # 降る（入力に出てこない）
    "P": [[0.5, 0.0, -0.5],       # t0
          [-0.5, 1.0, 0.0]],      # t1
    "Wq": [[-0.5, 0.0, 0.0],
           [0.0, 1.0, 0.0],
           [1.0, 0.0, -1.0]],
    "Wk": [[-1.0, 0.5, 1.0],
           [0.0, 0.0, 0.5],
           [0.0, 1.0, 0.0]],
    "Wv": [[-0.5, 0.5, 0.0],
           [0.5, -0.5, -0.5],
           [0.5, 1.0, -1.0]],
    "Wout": [[0.0, 0.0, -0.5],
             [0.0, 0.5, 0.0],
             [0.0, 0.0, 0.0]],
}
NAMES = list(PARAMS0)
C = np.sqrt(1.5)      # √(3/2)。1回目の U の成分は ±C か 0

POS = [f"t{t} {VOCAB[i]}" for t, i in enumerate(INP)]
DIMS = [f"d{j}" for j in range(D)]
KEYS = [f"k{t}" for t in range(T)]


def load_params():
    return {k: np.array(v, dtype=float) for k, v in PARAMS0.items()}


# ---------------------------------------------------------------- 部品

def layernorm(x):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)      # 母分散（÷3）
    sigma = np.sqrt(var)
    return xc / sigma, mu, var, sigma


def layernorm_backward(dy, y, sigma):
    return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sigma


def softmax(s):
    e = np.exp(s - np.max(s, -1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------- 順伝播・逆伝播

def forward(p):
    c = {}
    c["Eids"] = p["E"][INP]
    c["X0"] = c["Eids"] + p["P"]
    c["U"], c["mu"], c["var"], c["sigma"] = layernorm(c["X0"])
    c["Q"], c["K"], c["V"] = c["U"] @ p["Wq"], c["U"] @ p["Wk"], c["U"] @ p["Wv"]
    c["S"] = np.where(MASK, -np.inf, c["Q"] @ c["K"].T / np.sqrt(D))
    c["A"] = softmax(c["S"])
    c["O"] = c["A"] @ c["V"]
    c["X1"] = c["X0"] + c["O"]
    c["logits"] = c["X1"] @ p["Wout"]
    c["probs"] = softmax(c["logits"])
    m = c["logits"].max(-1)
    lse = np.log(np.exp(c["logits"] - m[:, None]).sum(-1)) + m
    c["nll"] = lse - c["logits"][np.arange(T), TGT]       # logsumexp − 正解の logit
    c["loss"] = c["nll"].mean()
    return c


def backward(p, c):
    g, d = {}, {}
    Y = np.zeros((T, V))
    Y[np.arange(T), TGT] = 1.0
    d["logits"] = (c["probs"] - Y) / T                     # ÷2 は2位置の平均
    g["Wout"] = c["X1"].T @ d["logits"]
    d["X1"] = d["logits"] @ p["Wout"].T
    d["O"] = d["X1"]                                       # 残差：O 側にそのままコピー
    A = c["A"]
    d["A"] = np.where(MASK, 0.0, d["O"] @ c["V"].T)
    d["r"] = (d["A"] * A).sum(-1)
    d["S"] = A * (d["A"] - d["r"][:, None])
    d["Q"] = d["S"] @ c["K"] / np.sqrt(D)
    d["K"] = d["S"].T @ c["Q"] / np.sqrt(D)
    d["V"] = A.T @ d["O"]
    g["Wq"] = c["U"].T @ d["Q"]
    g["Wk"] = c["U"].T @ d["K"]
    g["Wv"] = c["U"].T @ d["V"]
    d["U_q"], d["U_k"], d["U_v"] = d["Q"] @ p["Wq"].T, d["K"] @ p["Wk"].T, d["V"] @ p["Wv"].T
    d["U"] = d["U_q"] + d["U_k"] + d["U_v"]                # 分岐点 U：3本の枝の和
    d["X0_ln"] = layernorm_backward(d["U"], c["U"], c["sigma"])
    d["X0"] = d["X1"] + d["X0_ln"]                         # 分岐点 X0：残差 + LN経由
    g["P"] = d["X0"].copy()
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], INP, d["X0"])
    return g, d


def sgd(p, g, lr=LR):
    return {k: p[k] - lr * g[k] for k in NAMES}


def gsq(g, keys=NAMES):
    return sum(float((g[k] ** 2).sum()) for k in keys)


def grad_check(p, g, eps=1e-6):
    worst = {}
    for k in NAMES:
        err = 0.0
        for idx in np.ndindex(p[k].shape):
            q = {kk: vv.copy() for kk, vv in p.items()}
            q[k][idx] += eps
            lp = forward(q)["loss"]
            q[k][idx] -= 2 * eps
            lm = forward(q)["loss"]
            err = max(err, abs((lp - lm) / (2 * eps) - g[k][idx]))
        worst[k] = err
    return worst


# ---------------------------------------------------------------- チェックポイント

def run(p=None, lr=LR):
    p = load_params() if p is None else p
    c = forward(p)
    g, d = backward(p, c)
    p1 = sgd(p, g, lr)
    c1 = forward(p1)
    return p, c, g, d, p1, c1


def checkpoints(lr=LR):
    p, c, g, d, p1, c1 = run(lr=lr)
    G = gsq(g)
    col = lambda v: np.asarray(v, dtype=float).reshape(-1, 1)
    stats = lambda cc: np.hstack([cc["mu"], cc["var"], cc["sigma"]])
    cp = []
    add = lambda cid, title, items: cp.append((cid, title, items))

    add("A1", "埋め込み + 位置埋め込み", [("E[ids]", c["Eids"], POS, DIMS), ("X0", c["X0"], POS, DIMS)])
    add("A2", "LayerNorm", [("μ, σ², σ", stats(c), POS, ["μ", "σ²", "σ"]), ("U", c["U"], POS, DIMS),
                           ("U / √(3/2)（パターン）", c["U"] / C, POS, DIMS)])
    add("A3", "Q, K, V", [("Q", c["Q"], POS, DIMS), ("K", c["K"], POS, DIMS), ("V", c["V"], POS, DIMS),
                          ("Q / √(3/2)", c["Q"] / C, POS, DIMS), ("K / √(3/2)", c["K"] / C, POS, DIMS),
                          ("V / √(3/2)", c["V"] / C, POS, DIMS)])
    add("A4", "スコアと Attention 重み", [("S", c["S"], POS, KEYS), ("A", c["A"], POS, KEYS)])
    add("A5", "ヘッド出力と残差", [("O", c["O"], POS, DIMS), ("X1", c["X1"], POS, DIMS)])
    add("A6", "出力と損失", [("logits", c["logits"], POS, VOCAB), ("p", c["probs"], POS, VOCAB),
                          ("nll = −log p[正解]", col(c["nll"]), POS, ["nll"]), ("L", np.array([[c["loss"]]]), ["L"], ["値"])])

    add("B1", "出力層の逆伝播", [("dlogits", d["logits"], POS, VOCAB), ("dWout", g["Wout"], DIMS, VOCAB),
                             ("dX1", d["X1"], POS, DIMS)])
    add("B2", "Attention 重み・softmax の逆伝播", [("dO", d["O"], POS, DIMS), ("dA", d["A"], POS, KEYS),
                                              ("r（行ごとの Σ_j A·dA）", col(d["r"]), POS, ["r"]), ("dS", d["S"], POS, KEYS)])
    add("B3", "Q, K, V の勾配", [("dQ", d["Q"], POS, DIMS), ("dK", d["K"], POS, DIMS), ("dV", d["V"], POS, DIMS)])
    add("B4", "射影行列の勾配と dU", [("dWq", g["Wq"], DIMS, DIMS), ("dWk", g["Wk"], DIMS, DIMS), ("dWv", g["Wv"], DIMS, DIMS),
                                  ("dU の Q の枝", d["U_q"], POS, DIMS), ("dU の K の枝", d["U_k"], POS, DIMS),
                                  ("dU の V の枝", d["U_v"], POS, DIMS), ("dU（3本の和）", d["U"], POS, DIMS)])
    add("B5", "LayerNorm の逆伝播と残差の合流", [("LN経由", d["X0_ln"], POS, DIMS), ("dX0", d["X0"], POS, DIMS)])
    add("B6", "埋め込みの勾配", [("dP", g["P"], POS, DIMS), ("dE", g["E"], VOCAB, DIMS)])

    add("C1", "SGD 更新後のパラメータ（lr = 1）", [(f"{k}'", p1[k], *_labels(k)) for k in NAMES])
    add("C2", "群ごとの ‖g‖² と1次予測", [("‖g‖²（群ごと）", np.array([[gsq(g, [k]) for k in NAMES] + [G]]), ["値"], NAMES + ["合計"]),
                                    ("予測 ΔL = −lr‖g‖²", np.array([[-lr * G, c["loss"] - lr * G]]), ["値"], ["予測ΔL", "予測 L'"])])
    add("C3", "t0 だけの再順伝播", [("X0'[t0]（= X0 − 2·lr·dX0）", c1["X0"][:1], POS[:1], DIMS),
                               ("μ', σ²', σ'", stats(c1)[:1], POS[:1], ["μ", "σ²", "σ"]), ("U'[t0]", c1["U"][:1], POS[:1], DIMS),
                               ("V'[t0] = O'[t0]", c1["V"][:1], POS[:1], DIMS), ("X1'[t0]", c1["X1"][:1], POS[:1], DIMS),
                               ("logits'[t0]", c1["logits"][:1], POS[:1], VOCAB), ("p'[t0]", c1["probs"][:1], POS[:1], VOCAB),
                               ("nll'[t0]", col(c1["nll"][:1]), POS[:1], ["nll"])])
    add("C4", "再順伝播（全体）と損失の比較",
        [("U'", c1["U"], POS, DIMS), ("Q'", c1["Q"], POS, DIMS), ("K'", c1["K"], POS, DIMS), ("V'", c1["V"], POS, DIMS),
         ("S'", c1["S"], POS, KEYS), ("A'", c1["A"], POS, KEYS), ("X1'", c1["X1"], POS, DIMS),
         ("logits'", c1["logits"], POS, VOCAB), ("p'", c1["probs"], POS, VOCAB), ("nll'", col(c1["nll"]), POS, ["nll"]),
         ("L → L'", np.array([[c["loss"], c1["loss"], c1["loss"] - c["loss"], -lr * G, (c1["loss"] - c["loss"]) / (-lr * G)]]),
          ["値"], ["L", "L'", "実際ΔL", "予測ΔL", "実際/予測"]),
         ("正解トークンの確率 前→後", np.vstack([c["probs"][np.arange(T), TGT], c1["probs"][np.arange(T), TGT]]).T,
          POS, ["更新前", "更新後"])])
    return cp


def _labels(k):
    rows = {"E": VOCAB, "P": POS}.get(k, DIMS)
    cols = {"Wout": VOCAB}.get(k, DIMS)
    return rows, cols


# ---------------------------------------------------------------- 表示

def fmt(x, dec):
    """厳密に0は "0"、丸めると0になるだけの小さな値は "0.0000" と書き分ける。"""
    if np.isneginf(x):
        return "−∞"
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{dec}f}"
    return s.lstrip("-") if float(s) == 0 else s.replace("-", "−")


def render_text(cp, dec, only=None, section=None):
    out = []
    for cid, title, items in cp:
        if (only and cid != only) or (section and not cid.startswith(section)):
            continue
        out.append(f"\n=== {cid} {title} ===")
        for label, M, rl, cl in items:
            M = np.atleast_2d(M)
            out.append(f"  {label}")
            w = max(dec + 4, *(len(x) for x in cl))
            out.append("    " + " " * 10 + "".join(f"{x:>{w + 1}}" for x in cl))
            for r, row in zip(rl, M):
                out.append(f"    {r:<10}" + "".join(f"{fmt(v, dec):>{w + 1}}" for v in row))
    return "\n".join(out)


def md_escape(x):
    return x.replace("<", "&lt;").replace(">", "&gt;")


def skip_mask(cid, label, M):
    """記入用シートで「—」にするセル（t0 は自分しか見ないので、Q の t0 行と S・dA の (t0, k0) は結果に効かない）。"""
    k = np.zeros(np.atleast_2d(M).shape, dtype=bool)
    name = label.split()[0]
    if name in ("Q", "Q'", "dQ") and cid in ("A3", "C4"):
        k[0] = True
    if name in ("S", "S'", "dA") or name.startswith("r（"):
        k[0, 0] = True                   # t0 行は要素1つの softmax なので dS[t0] は dA に関係なく0
    return k


def md_table(cid, label, M, rl, cl, blank):
    M = np.atleast_2d(M)
    name = label.split()[0]
    masked = np.broadcast_to(MASK, M.shape) if (M.shape == (T, T) and name.rstrip("'") in ("S", "A", "dA", "dS")) \
        else np.zeros(M.shape, dtype=bool)
    skip = skip_mask(cid, label, M)
    out = [f"**{md_escape(label)}**", "", "| | " + " | ".join(map(md_escape, cl)) + " |", "|---|" + "---:|" * len(cl)]
    for r, row, mrow, krow in zip(rl, M, masked, skip):
        if blank:
            cells = [("−∞" if np.isneginf(v) else "0*") if m else ("—" if k else " ") for v, m, k in zip(row, mrow, krow)]
        else:
            cells = [fmt(v, 4) for v in row]
        out.append("| " + md_escape(r) + " | " + " | ".join(cells) + " |")
    return out + [""]


def render_markdown(cp, blank=False, only=None, section=None):
    out = []
    for cid, title, items in cp:
        if (only and cid != only) or (section and not cid.startswith(section)):
            continue
        out += [f"#### {cid} {title}", ""]
        for label, M, rl, cl in items:
            if blank and ("/ √(3/2)" in label):
                continue
            out += md_table(cid, label, M, rl, cl, blank)
    return "\n".join(out)


# ---------------------------------------------------------------- 検査

def checks(verbose=True):
    p, c, g, d, p1, c1 = run()
    res = []

    def ok(name, cond):
        res.append((name, bool(cond)))

    close = lambda a, b, tol=1e-12: np.allclose(a, b, atol=tol, rtol=0)
    w = {0: np.array([1.0, -2.0, 1.0]), 1: np.array([1.0, 1.0, -2.0])}   # 各行で 1 と U に直交する向き

    ok("A2 U の各行：平均 0、二乗平均 1", close(c["U"].mean(1), 0) and close((c["U"] ** 2).mean(1), 1))
    ok("A2 U の成分は ±√(3/2) か 0", close(np.abs(c["U"]) * (np.abs(c["U"]) > 1e-9), C * (np.abs(c["U"]) > 1e-9)))
    ok("A4 A の各行の和 = 1、t0 行 = [1, 0]", close(c["A"].sum(1), 1) and close(c["A"][0], [1, 0]))
    ok("A4 S[t1, k1] は厳密に0（Q[t1]·K[k1] = 0）", abs(c["S"][1, 1]) < 1e-12)
    ok("A6 p の各行の和 = 1", close(c["probs"].sum(1), 1))
    ok("A6 更新前の予測は2位置ともハズレ", (c["probs"].argmax(1) != TGT).all())
    ok("B1 dlogits の各行の和 = 0", close(d["logits"].sum(1), 0))
    ok("B1 dWout の各行の和 = 0", close(g["Wout"].sum(1), 0))
    ok("B1 dX1 の d2 列は厳密に0（Wout の d2 行が0）", np.all(d["X1"][:, 2] == 0))
    ok("B2 dS の各行の和 = 0、t0 行 = 0", close(d["S"].sum(1), 0) and np.all(np.abs(d["S"][0]) < 1e-12))
    ok("B2 r[t1] = dO[t1]·O[t1]", close(d["r"][1], d["O"][1] @ c["O"][1]))
    ok("B3 dQ の t0 行 = 0", np.all(np.abs(d["Q"][0]) < 1e-12))
    ok("B3 dQ[t1, d1] は厳密に0（K の d1 成分が k0 と k1 で等しい）", abs(d["Q"][1, 1]) < 1e-12 and close(c["K"][0, 1], c["K"][1, 1]))
    ok("B3 dK の列和 = 0（dK[k1] = −dK[k0]）", close(d["K"].sum(0), 0))
    ok("B3 dV の列和 = dO の列和", close(d["V"].sum(0), d["O"].sum(0)))
    ok("B4 dWq・dWk・dWv の各列の和 = 0", all(close(g[k].sum(0), 0) for k in ("Wq", "Wk", "Wv")))
    ok("B4 dWq の d2 行と d1 列は厳密に0", np.all(np.abs(g["Wq"][2]) < 1e-12) and np.all(np.abs(g["Wq"][:, 1]) < 1e-12))
    ok("B4 dWk・dWv の d2 列は厳密に0", np.all(np.abs(g["Wk"][:, 2]) < 1e-12) and np.all(np.abs(g["Wv"][:, 2]) < 1e-12))
    ok("B4 dWk = (U[t0] − U[t1])ᵀ dK[k0]（階数1）", close(g["Wk"], np.outer(c["U"][0] - c["U"][1], d["K"][0])))
    ok("B5 LN経由：各行の和 = 0、U との内積 = 0", close(d["X0_ln"].sum(1), 0) and close((d["X0_ln"] * c["U"]).sum(1), 0))
    ok("B5 LN経由は t0 で (1, −2, 1)、t1 で (1, 1, −2) の定数倍",
       all(close(d["X0_ln"][t], (d["U"][t] @ w[t]) / (6 * c["sigma"][t, 0]) * w[t]) for t in range(T)))
    ok("B5 dX0 と dX1 の行和が等しい", close(d["X0"].sum(1), d["X1"].sum(1)))
    ok("B6 dE の 降る 行 = 0、使われた行 = dP", np.all(g["E"][2] == 0) and close(g["E"][:2], g["P"]))
    ok("C1 X0' = X0 − 2·lr·dX0", close(c1["X0"], c["X0"] - 2 * LR * d["X0"]))
    ok("C1 Wout' の各行の和 = Wout の各行の和", close(p1["Wout"].sum(1), p["Wout"].sum(1)))
    ok("C1 ‖dE‖² = ‖dP‖²", abs(gsq(g, ["E"]) - gsq(g, ["P"])) < 1e-15)
    ok("C4 A' の t0 行 = [1, 0]", close(c1["A"][0], [1, 0]))
    ok("C4 L' < L、実際/予測は 0 と 1 の間",
       c1["loss"] < c["loss"] and 0 < (c1["loss"] - c["loss"]) / (-LR * gsq(g)) < 1)
    ok("C4 更新後の予測は2位置とも正解", (c1["probs"].argmax(1) == TGT).all())
    worst = grad_check(p, g)
    ok(f"数値微分との最大誤差 {max(worst.values()):.1e} < 1e-8（51個すべて）", max(worst.values()) < 1e-8)

    if verbose:
        for name, r in res:
            print(f"  [{'OK' if r else 'NG'}] {name}")
    passed = all(r for _, r in res)
    print(f"\n{len(res)} 件中 {sum(r for _, r in res)} 件 OK")
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return passed


# ---------------------------------------------------------------- 説明し直し用の数値

def explain():
    p, c, g, d, p1, c1 = run()
    G, L = gsq(g), c["loss"]
    along = lambda eta: forward({k: p[k] - eta * g[k] for k in NAMES})["loss"]
    h = 1e-4
    curv = lambda eta: (along(eta + h) - 2 * along(eta) + along(eta - h)) / h ** 2

    print(f"L = {L:.4f}, ‖g‖² = {G:.4f}, 予測 ΔL = {-G:.4f}, L/‖g‖² = {L / G:.4f}")
    print("\n[1] 勾配方向 θ − η·g に沿った損失と、1次予測との比（η = その lr で1ステップ進んだ点）")
    print("    η      L(η)     実際ΔL    予測ΔL   実際/予測")
    for eta in [0.1, 0.25, 0.5, 1, 2, 3, 4, 6]:
        Le = along(eta)
        print(f"  {eta:4g}  {Le:8.4f}  {Le - L:8.4f}  {-eta * G:8.4f}   {(Le - L) / (-eta * G):.4f}")
    print(f"  曲率 d²L/dη²：η=0 で {curv(0):.4f}、η=0.5 で {curv(0.5):.4f}、η=1 で {curv(1):.4f}")
    q2 = -G + 0.5 * curv(0)
    print(f"  2次までのテイラー予測（η=1）：{q2:.4f}（実際/予測 {(c1['loss'] - L) / q2:.4f}）")

    print("\n[2] dlogits の ÷2 を忘れた場合（勾配がすべて2倍 = lr 2 と同じ）")
    print(f"  L' = {along(2):.4f}（正しい勾配では {c1['loss']:.4f}）")

    print("\n[3] 1つの群だけを更新したとき")
    print("  群      L      実際ΔL   1次予測  予測（各位置の argmax）")
    tot = 0.0
    for k in NAMES:
        q = dict(p)
        q[k] = p[k] - LR * g[k]
        cq = forward(q)
        tot += cq["loss"] - L
        pred = " ".join(VOCAB[i] for i in cq["probs"].argmax(1))
        print(f"  {k:<5} {cq['loss']:.4f}  {cq['loss'] - L:8.4f}  {-gsq(g, [k]):8.4f}  {pred}")
    print(f"  6つの和            {tot:8.4f}  {-G:8.4f}")
    print(f"  全部同時  {c1['loss']:.4f}  {c1['loss'] - L:8.4f}  {-G:8.4f}  "
          + " ".join(VOCAB[i] for i in c1["probs"].argmax(1)))

    print("\n[4] logits の分解：直接経路 X0·Wout と Attention 経路 O·Wout（FFN も最終 LN も無いので厳密に足し算）")
    for nm, cc, pp in [("更新前", c, p), ("更新後", c1, p1)]:
        print(f"  {nm}  直接経路:\n{np.round(cc['X0'] @ pp['Wout'], 4)}\n        Attention 経路:\n{np.round(cc['O'] @ pp['Wout'], 4)}")
    print(f"  A[t1] = {np.round(c['A'][1], 4)} → A'[t1] = {np.round(c1['A'][1], 4)}")

    print("\n[5] 2ステップ目の勾配で、1ステップ目に厳密に0だったものはどうなるか")
    g1, d1 = backward(p1, c1)
    print(f"  dQ の t0 行：{np.round(d1['Q'][0], 6)}（mask のため、どの重みでも0）")
    print(f"  dWq の d2 行：{np.round(g1['Wq'][2], 4)}（U'[t1, d2] = {c1['U'][1, 2]:.4f} はもう0でない）")
    print(f"  dX1 の d2 列：{np.round(d1['X1'][:, 2], 4)}（Wout' の d2 行 = {np.round(p1['Wout'][2], 4)}）")
    print(f"  dE の 降る 行：{np.round(g1['E'][2], 4)}（入力に出てこない限り0）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--section", choices=["A", "B", "C"])
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--explain", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(0 if checks() else 1)
    if a.gradcheck:
        p, c, g, *_ = run()
        for k, e in grad_check(p, g).items():
            print(f"{k:5s} 数値微分との最大誤差 {e:.2e}")
        return
    if a.explain:
        explain()
        return
    cp = checkpoints()
    if a.markdown or a.blank:
        print(render_markdown(cp, blank=a.blank, only=a.only, section=a.section))
    else:
        print(render_text(cp, a.decimals, a.only, a.section))


if __name__ == "__main__":
    main()
