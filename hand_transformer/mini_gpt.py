"""手計算ワークブック用ミニGPTの NumPy 正解実装。

手計算と同じ順序・同じ記号で、順伝播 → 損失 → 逆伝播 → SGD更新 → 再順伝播 を行い、
ワークブックのチェックポイント番号（2.1, 3.2, 6.5 ...）ごとに中間値を表示する。

使い方:
  python mini_gpt.py                 # 全チェックポイントを表示
  python mini_gpt.py --step 3        # ステップ3（Attention）だけ表示
  python mini_gpt.py --only 6.5      # チェックポイント 6.5 だけ表示
  python mini_gpt.py --markdown      # 解答集（Markdown）を出力
  python mini_gpt.py --gradcheck     # 数値微分で逆伝播を検算
  python mini_gpt.py --train 200     # 同じデータで200ステップ学習（発展課題用）

モデル（README.md の設計書と同一）:
  X0 = E[ids] + P
  U1 = LN(X0)                                  # gamma=1, beta=0, eps=0
  Q, K, V = U1 Wq, U1 Wk, U1 Wv
  S_h = Q_h K_h^T / sqrt(2)（因果マスク）, A_h = softmax(S_h), O_h = A_h V_h
  Attn = [O_0 | O_1] Wo
  X1 = X0 + Attn
  U2 = LN(X1)
  Z = U2 W1, R = ReLU(Z), F = R W2
  X2 = X1 + F
  logits = X2 Wout, p = softmax(logits), L = mean_t -log p[t, y_t]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOCAB = ["<s>", "猫", "が", "鳴く", "。"]
INP = [0, 1, 2]   # <s> 猫 が
TGT = [1, 2, 3]   # 猫 が 鳴く
T, D, H, DK, FF, V = 3, 4, 2, 2, 8, 5
SHAPES = {"E": (5, 4), "P": (3, 4), "Wq": (4, 4), "Wk": (4, 4), "Wv": (4, 4),
          "Wo": (4, 4), "W1": (4, 8), "W2": (8, 4), "Wout": (4, 5)}
PARAMS = list(SHAPES)
MASK = np.triu(np.ones((T, T), dtype=bool), 1)   # True = 未来（見てはいけない）

POS = [f"t{t} {VOCAB[i]}" for t, i in enumerate(INP)]
DIMS = [f"d{j}" for j in range(D)]
HID = [f"h{j}" for j in range(FF)]
KEYS = [f"k{t}" for t in range(T)]


def load_params(path=HERE / "params.json"):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    p = {k: np.array(raw[k], dtype=float) for k in PARAMS}
    for k, s in SHAPES.items():
        assert p[k].shape == s, f"{k}: {p[k].shape} != {s}"
    return p, float(raw["lr"])


# ---------------------------------------------------------------- 部品

def layernorm(x):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)      # 母分散（÷4）
    sigma = np.sqrt(var)
    return xc / sigma, {"mu": mu, "var": var, "sigma": sigma}


def layernorm_backward(dy, y, sigma):
    # dx = (dy - mean(dy) - y * mean(dy * y)) / sigma
    return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sigma


def softmax(s):
    e = np.exp(s - np.max(s, -1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def head_slice(h):
    return slice(h * DK, (h + 1) * DK)


# ---------------------------------------------------------------- 順伝播・逆伝播

def forward(p):
    c = {}
    c["X0"] = p["E"][INP] + p["P"]
    c["U1"], c["ln1"] = layernorm(c["X0"])
    c["Q"], c["K"], c["V"] = c["U1"] @ p["Wq"], c["U1"] @ p["Wk"], c["U1"] @ p["Wv"]
    c["S"], c["A"], c["O"] = [], [], []
    for h in range(H):
        sl = head_slice(h)
        S = c["Q"][:, sl] @ c["K"][:, sl].T / np.sqrt(DK)
        S = np.where(MASK, -np.inf, S)
        c["S"].append(S)
        c["A"].append(softmax(S))
        c["O"].append(c["A"][h] @ c["V"][:, sl])
    c["Ocat"] = np.concatenate(c["O"], axis=1)
    c["Attn"] = c["Ocat"] @ p["Wo"]
    c["X1"] = c["X0"] + c["Attn"]
    c["U2"], c["ln2"] = layernorm(c["X1"])
    c["Z"] = c["U2"] @ p["W1"]
    c["R"] = np.maximum(c["Z"], 0.0)
    c["F"] = c["R"] @ p["W2"]
    c["X2"] = c["X1"] + c["F"]
    c["logits"] = c["X2"] @ p["Wout"]
    c["probs"] = softmax(c["logits"])
    lse = np.log(np.exp(c["logits"] - c["logits"].max(-1, keepdims=True)).sum(-1)) + c["logits"].max(-1)
    c["nll"] = lse - c["logits"][np.arange(T), TGT]          # = -log p[正解]（logsumexp − logit）
    c["loss"] = c["nll"].mean()
    return c


def backward(p, c):
    """手計算と同じ順序で勾配を計算する。g: パラメータ勾配, d: 中間変数の勾配。"""
    g, d = {}, {}
    Y = np.zeros((T, V))
    Y[np.arange(T), TGT] = 1.0
    d["logits"] = (c["probs"] - Y) / T                     # softmax+CE の微分（÷3 は平均）
    g["Wout"] = c["X2"].T @ d["logits"]
    d["X2"] = d["logits"] @ p["Wout"].T
    d["F"] = d["X2"]                                        # 残差: 勾配はそのままコピー
    g["W2"] = c["R"].T @ d["F"]
    d["R"] = d["F"] @ p["W2"].T
    d["Z"] = d["R"] * (c["Z"] > 0)
    g["W1"] = c["U2"].T @ d["Z"]
    d["U2"] = d["Z"] @ p["W1"].T
    d["X1_ln2"] = layernorm_backward(d["U2"], c["U2"], c["ln2"]["sigma"])
    d["X1"] = d["X2"] + d["X1_ln2"]                        # 残差の素通り + LN2経由
    d["Attn"] = d["X1"]
    g["Wo"] = c["Ocat"].T @ d["Attn"]
    d["Ocat"] = d["Attn"] @ p["Wo"].T
    d["A"], d["S"] = [], []
    d["Q"], d["K"], d["V"] = np.zeros((T, D)), np.zeros((T, D)), np.zeros((T, D))
    for h in range(H):
        sl = head_slice(h)
        dO = d["Ocat"][:, sl]
        A = c["A"][h]
        dA = np.where(MASK, 0.0, dO @ c["V"][:, sl].T)
        d["V"][:, sl] = A.T @ dO
        dS = A * (dA - (dA * A).sum(-1, keepdims=True))     # softmax の微分（行ごと）
        d["A"].append(dA)
        d["S"].append(dS)
        d["Q"][:, sl] = dS @ c["K"][:, sl] / np.sqrt(DK)
        d["K"][:, sl] = dS.T @ c["Q"][:, sl] / np.sqrt(DK)
    g["Wq"] = c["U1"].T @ d["Q"]
    g["Wk"] = c["U1"].T @ d["K"]
    g["Wv"] = c["U1"].T @ d["V"]
    d["U1"] = d["Q"] @ p["Wq"].T + d["K"] @ p["Wk"].T + d["V"] @ p["Wv"].T
    d["X0_ln1"] = layernorm_backward(d["U1"], c["U1"], c["ln1"]["sigma"])
    d["X0"] = d["X1"] + d["X0_ln1"]
    g["P"] = d["X0"].copy()
    g["E"] = np.zeros((V, D))
    for t, i in enumerate(INP):
        g["E"][i] += d["X0"][t]
    return g, d


def sgd(p, g, lr):
    return {k: p[k] - lr * g[k] for k in PARAMS}


def grad_check(p, g, eps=1e-6):
    worst = {}
    for k in PARAMS:
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


def logit_decomposition(p, c):
    """最終LNが無いので logits は残差ストリームの各成分の和に厳密に分解できる。"""
    parts = {"直接経路 X0": c["X0"]}
    for h in range(H):
        sl = head_slice(h)
        parts[f"ヘッド{h}"] = c["O"][h] @ p["Wo"][sl, :]
    parts["FFN"] = c["F"]
    return {k: v @ p["Wout"] for k, v in parts.items()}


# ---------------------------------------------------------------- 表示

def checkpoints(p, lr):
    """(id, タイトル, [(ラベル, 行列, 行ラベル, 列ラベル)]) のリストを返す。"""
    c = forward(p)
    g, d = backward(p, c)
    p1 = sgd(p, g, lr)
    c1 = forward(p1)
    gsq = sum(float((g[k] ** 2).sum()) for k in PARAMS)
    col = lambda v: np.asarray(v).reshape(-1, 1)
    cp = []
    add = lambda cid, title, items: cp.append((cid, title, items))

    add("2.1", "埋め込み + 位置埋め込み", [("E[ids]", p["E"][INP], POS, DIMS), ("X0 = E[ids] + P", c["X0"], POS, DIMS)])
    add("2.2", "LayerNorm1 の統計量", [("μ, σ², σ", np.hstack([c["ln1"]["mu"], c["ln1"]["var"], c["ln1"]["sigma"]]), POS, ["μ", "σ²", "σ"])])
    add("2.3", "LayerNorm1 の出力", [("U1 = (X0 - μ) / σ", c["U1"], POS, DIMS)])
    add("3.1", "Q, K, V", [("Q = U1 Wq", c["Q"], POS, DIMS), ("K = U1 Wk", c["K"], POS, DIMS), ("V = U1 Wv", c["V"], POS, DIMS)])
    add("3.2", "スコア（マスク付き）", [(f"S{h} = Q{h} K{h}ᵀ / √2", c["S"][h], POS, KEYS) for h in range(H)])
    add("3.3", "Attention重み", [(f"A{h} = softmax(S{h})", c["A"][h], POS, KEYS) for h in range(H)])
    add("3.4", "ヘッド出力と連結", [(f"O{h} = A{h} V{h}", c["O"][h], POS, DIMS[h * DK:(h + 1) * DK]) for h in range(H)]
        + [("Ocat = [O0 | O1]", c["Ocat"], POS, DIMS)])
    add("3.5", "出力射影", [("Attn = Ocat Wo", c["Attn"], POS, DIMS)])
    add("4.1", "残差接続1", [("X1 = X0 + Attn", c["X1"], POS, DIMS)])
    add("4.2", "LayerNorm2", [("μ, σ², σ", np.hstack([c["ln2"]["mu"], c["ln2"]["var"], c["ln2"]["sigma"]]), POS, ["μ", "σ²", "σ"]),
                              ("U2 = (X1 - μ) / σ", c["U2"], POS, DIMS)])
    add("4.3", "FFN 1層目 + ReLU", [("Z = U2 W1", c["Z"], POS, HID), ("R = ReLU(Z)", c["R"], POS, HID)])
    add("4.4", "FFN 2層目", [("F = R W2", c["F"], POS, DIMS)])
    add("4.5", "残差接続2", [("X2 = X1 + F", c["X2"], POS, DIMS)])
    add("5.1", "ロジット", [("logits = X2 Wout", c["logits"], POS, VOCAB)])
    add("5.2", "softmax", [("p = softmax(logits)", c["probs"], POS, VOCAB)])
    add("5.3", "損失", [("-log p[正解]", col(c["nll"]), POS, ["nll"]), ("L（平均）", np.array([[c["loss"]]]), ["L"], ["値"])])

    add("6.1", "出力層の逆伝播", [("dlogits = (p - y) / 3", d["logits"], POS, VOCAB), ("dWout = X2ᵀ dlogits", g["Wout"], DIMS, VOCAB),
                               ("dX2 = dlogits Woutᵀ", d["X2"], POS, DIMS)])
    add("6.2", "FFN の逆伝播", [("dF = dX2（残差で素通り）", d["F"], POS, DIMS), ("dW2 = Rᵀ dF", g["W2"], HID, DIMS),
                              ("dR = dF W2ᵀ", d["R"], POS, HID), ("dZ = dR ⊙ [Z>0]", d["Z"], POS, HID),
                              ("dW1 = U2ᵀ dZ", g["W1"], DIMS, HID), ("dU2 = dZ W1ᵀ", d["U2"], POS, DIMS)])
    add("6.3", "LayerNorm2 の逆伝播と残差の合流", [("LN2経由: (dU2 - mean(dU2) - U2·mean(dU2⊙U2)) / σ", d["X1_ln2"], POS, DIMS),
                                          ("dX1 = dX2 + LN2経由", d["X1"], POS, DIMS)])
    add("6.4", "出力射影の逆伝播", [("dAttn = dX1", d["Attn"], POS, DIMS), ("dWo = Ocatᵀ dAttn", g["Wo"], DIMS, DIMS),
                                ("dOcat = dAttn Woᵀ", d["Ocat"], POS, DIMS)])
    items = []
    for h in range(H):
        sl = head_slice(h)
        items += [(f"dA{h} = dO{h} V{h}ᵀ（マスク部は0）", d["A"][h], POS, KEYS),
                  (f"dS{h} = A{h} ⊙ (dA{h} - rowsum(dA{h} ⊙ A{h}))", d["S"][h], POS, KEYS)]
    add("6.5", "Attention重み・softmax の逆伝播", items)
    items = []
    for h in range(H):
        sl = head_slice(h)
        items += [(f"dQ{h} = dS{h} K{h} / √2", d["Q"][:, sl], POS, DIMS[sl]),
                  (f"dK{h} = dS{h}ᵀ Q{h} / √2", d["K"][:, sl], POS, DIMS[sl]),
                  (f"dV{h} = A{h}ᵀ dO{h}", d["V"][:, sl], POS, DIMS[sl])]
    add("6.6", "Q, K, V の勾配", items)
    add("6.7", "射影行列の勾配と dU1", [("dWq = U1ᵀ dQ", g["Wq"], DIMS, DIMS), ("dWk = U1ᵀ dK", g["Wk"], DIMS, DIMS),
                                   ("dWv = U1ᵀ dV", g["Wv"], DIMS, DIMS), ("dU1 = dQ Wqᵀ + dK Wkᵀ + dV Wvᵀ", d["U1"], POS, DIMS)])
    add("6.8", "LayerNorm1 の逆伝播と残差の合流", [("LN1経由", d["X0_ln1"], POS, DIMS), ("dX0 = dX1 + LN1経由", d["X0"], POS, DIMS)])
    add("6.9", "埋め込みの勾配", [("dP = dX0", g["P"], POS, DIMS), ("dE（使われた行にだけ加算）", g["E"], VOCAB, DIMS)])

    add("7.1", "SGD更新後のパラメータ", [(f"{k}' = {k} - {lr:g}·d{k}", p1[k], *_param_labels(k)) for k in PARAMS])
    add("7.2", "損失変化の1次近似（テイラー展開）", [("‖g‖², 予測 ΔL = -lr‖g‖²", np.array([[gsq, -lr * gsq]]), ["値"], ["‖g‖²", "予測ΔL"])])
    add("7.3", "再順伝播（主要な中間値）",
        [("X0'", c1["X0"], POS, DIMS),
         ("μ', σ²', σ'（LN1）", np.hstack([c1["ln1"]["mu"], c1["ln1"]["var"], c1["ln1"]["sigma"]]), POS, ["μ", "σ²", "σ"]),
         ("U1'", c1["U1"], POS, DIMS), ("Q'", c1["Q"], POS, DIMS), ("K'", c1["K"], POS, DIMS), ("V'", c1["V"], POS, DIMS)]
        + [(f"S{h}'", c1["S"][h], POS, KEYS) for h in range(H)]
        + [(f"A{h}'", c1["A"][h], POS, KEYS) for h in range(H)]
        + [("Ocat'", c1["Ocat"], POS, DIMS), ("Attn'", c1["Attn"], POS, DIMS), ("X1'", c1["X1"], POS, DIMS),
           ("μ', σ²', σ'（LN2）", np.hstack([c1["ln2"]["mu"], c1["ln2"]["var"], c1["ln2"]["sigma"]]), POS, ["μ", "σ²", "σ"]),
           ("U2'", c1["U2"], POS, DIMS), ("Z'", c1["Z"], POS, HID), ("F'", c1["F"], POS, DIMS),
           ("X2'", c1["X2"], POS, DIMS), ("logits'", c1["logits"], POS, VOCAB), ("p'", c1["probs"], POS, VOCAB),
           ("-log p'[正解]", col(c1["nll"]), POS, ["nll"])])
    add("7.4", "損失の比較", [("L → L'", np.array([[c["loss"], c1["loss"], c1["loss"] - c["loss"], -lr * gsq]]), ["値"],
                              ["L（更新前）", "L'（更新後）", "実際のΔL", "予測ΔL"]),
                             ("正解トークンの確率 前→後", np.vstack([c["probs"][np.arange(T), TGT], c1["probs"][np.arange(T), TGT]]).T,
                              POS, ["更新前", "更新後"])])
    return cp


def _param_labels(k):
    rows = {"E": VOCAB, "P": POS, "W2": HID}.get(k, DIMS)
    cols = {"W1": HID, "Wout": VOCAB}.get(k, DIMS)
    return rows, cols


def fmt(x, dec):
    """厳密に0（構造的なゼロ）は "0"、丸めると0になるだけの小さな値は "0.0000" と書き分ける。"""
    if np.isneginf(x):
        return "−∞"
    if abs(x) < 1e-12:
        return "0"
    s = f"{x:.{dec}f}"
    return s.lstrip("-") if float(s) == 0 else s.replace("-", "−")


def render_text(cp, dec, only=None, step=None):
    out = []
    for cid, title, items in cp:
        if (only and cid != only) or (step and not cid.startswith(f"{step}.")):
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


def md_table(label, M, rl, cl, cell, skip=None):
    M = np.atleast_2d(M)
    masked = np.broadcast_to(MASK, M.shape) if is_masked(label, M) else np.zeros(M.shape, dtype=bool)
    skip = np.zeros(M.shape, dtype=bool) if skip is None else skip
    out = [f"**{md_escape(label)}**", "", "| | " + " | ".join(map(md_escape, cl)) + " |", "|---|" + "---:|" * len(cl)]
    out += ["| " + md_escape(r) + " | " + " | ".join(cell(v, m, k) for v, m, k in zip(row, mrow, krow)) + " |"
            for r, row, mrow, krow in zip(rl, M, masked, skip)]
    return out + [""]


def is_masked(label, M):
    """因果マスクで値が決まる表（S, A, dA, dS とその更新後）かどうか。"""
    return M.shape == (T, T) and label.split()[0].rstrip("'").startswith(("S", "A", "dA", "dS"))


def skip_cells(cid, label, M):
    """手計算では求めなくてよいセル（位置0は自分しか見ないので A の0行目は常に [1, 0, 0]）。"""
    k = np.zeros(np.atleast_2d(M).shape, dtype=bool)
    name = label.split()[0]
    if cid == "3.1" and name == "Q":
        k[0] = True                      # Q の0行目は下流に一切影響しない
    if cid == "3.2" or (cid == "6.5" and name.startswith("dA")):
        k[0, 0] = True                   # S[0,0] と dA[0,0] は softmax(1要素) = 1 に吸収される
    return k


def render_markdown(cp, dec, blank=False):
    if blank:
        head = ["# 記入用シート（mini_gpt.py --blank が自動生成）", "",
                "空欄に手計算の結果を書き込み、チェックポイントごとに answers.md と照合してください。",
                "「−∞」「0*」は因果マスクで決まる値、「—」は計算しなくてよいセルです（位置0は自分自身しか見ないので、"
                "Q の0行目・S と dA の (t0, k0) は結果に影響しません）。", ""]
        cell = lambda v, m, k: ("−∞" if np.isneginf(v) else "0*") if m else ("—" if k else " ")
    else:
        head = ["# 解答集（mini_gpt.py --markdown が自動生成）", "",
                f"数値は小数点以下{dec}桁に丸めています。「0」は厳密に0（構造的に勾配が来ない等）、「0.0000」は丸めると0になるだけの小さな値です。",
                "手計算では有効数字4桁程度を保ち、差が |自分 − 解答| ≤ 0.002 + 0.002·|解答| 程度なら丸め誤差とみなしてOKです。",
                "これより大きくずれたら、そのチェックポイントで止まって原因を探し、以降はこの解答の値を使って先へ進んでください（誤差の伝播を防ぐ）。",
                "記入用シートで「—」（計算不要）としたセルにも、参考として値を載せています。", ""]
        cell = lambda v, m, k: fmt(v, dec)
    out = head
    for cid, title, items in cp:
        out += [f"## {cid} {title}", ""]
        for label, M, rl, cl in items:
            out += md_table(label, M, rl, cl, cell, skip_cells(cid, label, M))
    return "\n".join(out)


def render_params(p, lr):
    out = ["# 初期パラメータ（mini_gpt.py --params-md が自動生成）", "",
           f"学習率 lr = {lr:g}（SGD）。正本は params.json。", ""]
    names = {"E": "トークン埋め込み E（行 = トークン）", "P": "位置埋め込み P（行 = 位置）",
             "Wq": "Wq", "Wk": "Wk", "Wv": "Wv", "Wo": "Wo（行 d0,d1 = ヘッド0 の出力、d2,d3 = ヘッド1 の出力）",
             "W1": "W1（FFN 1層目）", "W2": "W2（FFN 2層目）", "Wout": "Wout（出力層、列 = トークン）"}
    for k in PARAMS:
        out += md_table(names[k], p[k], *_param_labels(k), lambda v, m, k: f"{v:g}".replace("-", "−"))
    return "\n".join(out)


def train(p, lr, steps):
    for s in range(steps + 1):
        c = forward(p)
        if s % max(1, steps // 20) == 0 or s == steps:
            pred = " ".join(VOCAB[i] for i in c["probs"].argmax(-1))
            print(f"step {s:4d}  loss {c['loss']:.4f}  予測: {pred}")
        if s < steps:
            g, _ = backward(p, c)
            p = sgd(p, g, lr)
    for h in range(H):
        print(f"A{h}:\n{np.round(c['A'][h], 3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(HERE / "params.json"))
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--step", type=int)
    ap.add_argument("--only")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--blank", action="store_true", help="記入用の空欄表（Markdown）を出力")
    ap.add_argument("--params-md", action="store_true", help="パラメータ表（Markdown）を出力")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--decompose", action="store_true", help="ロジットを直接経路/各ヘッド/FFNの寄与に分解")
    ap.add_argument("--train", type=int, metavar="STEPS")
    ap.add_argument("--lr", type=float, help="params.json の学習率を上書き")
    a = ap.parse_args()
    p, lr = load_params(a.params)
    lr = a.lr if a.lr is not None else lr
    if a.gradcheck:
        c = forward(p)
        g, _ = backward(p, c)
        for k, e in grad_check(p, g).items():
            print(f"{k:5s} 数値微分との最大誤差 {e:.2e}")
        return
    if a.decompose:
        c = forward(p)
        for k, v in logit_decomposition(p, c).items():
            print(f"--- {k} ---\n{np.round(v, a.decimals)}")
        print(f"--- 合計（= logits）---\n{np.round(c['logits'], a.decimals)}")
        return
    if a.train:
        train(p, lr, a.train)
        return
    if a.params_md:
        print(render_params(p, lr))
        return
    cp = checkpoints(p, lr)
    if a.markdown or a.blank:
        print(render_markdown(cp, a.decimals, blank=a.blank))
    else:
        print(render_text(cp, a.decimals, a.only, a.step))


if __name__ == "__main__":
    main()
