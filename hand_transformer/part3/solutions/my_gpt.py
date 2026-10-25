"""K3 の参考解答：Part 1 / Part 2 のモデルを行列レベルで書いたもの（k3_check.py の API どおり）。

まず part3/k3_matrix/my_gpt.py を自分で埋めてください。ここは、詰まったときと、書き終えたあとに見比べるためのものです。
mini_gpt.py とは独立に書いてあります（同じ答えが出ることを k3_check.py で確かめられます）。

    python3 part3/k3_check.py part3/solutions/my_gpt.py

方針:
- 入力は task = {"inp", "tgt", "posi", "mask"}。行 = 「どの文のどの位置か」。mask[i, j] = True は「行 i は行 j を見てはいけない」。
  Part 1 は3行・ふつうの因果マスク。Part 2 は2文を6行に積み、文をまたぐ部分も True にしたブロック対角のマスク。
  forward はマスクの形を気にしないので、**同じコードが両方で動く**（これが「バッチへの一般化」の一番簡単な形）。
- 逆伝播は順伝播の逆順に、derivations.md の問1〜8 の式をそのまま書く。
"""
import numpy as np

H, DK = 2, 2                     # ヘッド数、ヘッドあたりの次元（d_model = H·DK = 4）
PARAM_NAMES = ["E", "P", "Wq", "Wk", "Wv", "Wo", "W1", "W2", "Wout"]


# ---------------------------------------------------------------- 部品（順伝播と逆伝播を対で）

def ln_fwd(x):
    """γ=1, β=0, ε=0、母分散。統計量も返す（2.2, 4.2 で照合する）。"""
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    sigma = np.sqrt(var)
    return (x - mu) / sigma, {"mu": mu, "var": var, "sigma": sigma}


def ln_bwd(dy, y, sigma):
    """derivations.md 問8：dx = (dy − mean(dy) − y·mean(dy⊙y)) / σ（行ごとの平均）。"""
    return (dy - dy.mean(axis=1, keepdims=True) - y * (dy * y).mean(axis=1, keepdims=True)) / sigma


def softmax_rows(s):
    z = s - s.max(axis=1, keepdims=True)          # マスク（−∞）は exp で 0 になる。各行に1つは有限の値がある
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cols(h):
    return slice(DK * h, DK * (h + 1))


# ---------------------------------------------------------------- 順伝播

def forward(p, task):
    inp, tgt, posi = np.asarray(task["inp"]), np.asarray(task["tgt"]), np.asarray(task["posi"])
    mask = np.asarray(task["mask"], dtype=bool)
    n = len(inp)
    c = {"task": task}
    c["X0"] = p["E"][inp] + p["P"][posi]
    c["U1"], c["ln1"] = ln_fwd(c["X0"])
    Q, K, V = c["U1"] @ p["Wq"], c["U1"] @ p["Wk"], c["U1"] @ p["Wv"]
    c.update(Q=Q, K=K, V=V, S=[], A=[], O=[])
    for h in range(H):
        S = np.where(mask, -np.inf, Q[:, cols(h)] @ K[:, cols(h)].T / np.sqrt(DK))
        A = softmax_rows(S)
        c["S"].append(S)
        c["A"].append(A)
        c["O"].append(A @ V[:, cols(h)])
    c["Ocat"] = np.hstack(c["O"])
    c["Attn"] = c["Ocat"] @ p["Wo"]
    c["X1"] = c["X0"] + c["Attn"]
    c["U2"], c["ln2"] = ln_fwd(c["X1"])
    c["Z"] = c["U2"] @ p["W1"]
    c["R"] = np.where(c["Z"] > 0, c["Z"], 0.0)
    c["F"] = c["R"] @ p["W2"]
    c["X2"] = c["X1"] + c["F"]
    z = c["X2"] @ p["Wout"]
    c["logits"] = z
    c["probs"] = softmax_rows(z)
    zmax = z.max(axis=1)
    lse = zmax + np.log(np.exp(z - zmax[:, None]).sum(axis=1))
    c["nll"] = lse - z[np.arange(n), tgt]           # −log p[正解] = logsumexp − logit（README §7 ルール2）
    c["loss"] = float(c["nll"].mean())
    return c


# ---------------------------------------------------------------- 逆伝播

def backward(p, c):
    task = c["task"]
    inp, tgt, posi = np.asarray(task["inp"]), np.asarray(task["tgt"]), np.asarray(task["posi"])
    mask = np.asarray(task["mask"], dtype=bool)
    n = len(inp)
    g, d = {}, {}
    onehot = np.eye(p["Wout"].shape[1])[tgt]
    d["logits"] = (c["probs"] - onehot) / n                           # 問3（÷n は平均）
    g["Wout"] = c["X2"].T @ d["logits"]                               # 問1
    d["X2"] = d["logits"] @ p["Wout"].T
    d["F"] = d["X2"]                                                  # 問2：残差
    g["W2"] = c["R"].T @ d["F"]
    d["R"] = d["F"] @ p["W2"].T
    d["Z"] = d["R"] * (c["Z"] > 0)                                    # 問7-2
    g["W1"] = c["U2"].T @ d["Z"]
    d["U2"] = d["Z"] @ p["W1"].T
    d["X1_ln2"] = ln_bwd(d["U2"], c["U2"], c["ln2"]["sigma"])        # 問8
    d["X1"] = d["X2"] + d["X1_ln2"]                                   # 問2：2経路の和
    d["Attn"] = d["X1"]
    g["Wo"] = c["Ocat"].T @ d["Attn"]
    d["Ocat"] = d["Attn"] @ p["Wo"].T
    dQ, dK, dV = np.zeros_like(c["Q"]), np.zeros_like(c["K"]), np.zeros_like(c["V"])
    d["A"], d["S"] = [], []
    for h in range(H):
        A, dO = c["A"][h], d["Ocat"][:, cols(h)]                     # 問7-1：連結の逆は分割
        dA = np.where(mask, 0.0, dO @ c["V"][:, cols(h)].T)           # 問6
        dS = A * (dA - (A * dA).sum(axis=1, keepdims=True))           # 問4
        dQ[:, cols(h)] = dS @ c["K"][:, cols(h)] / np.sqrt(DK)        # 問5
        dK[:, cols(h)] = dS.T @ c["Q"][:, cols(h)] / np.sqrt(DK)
        dV[:, cols(h)] = A.T @ dO
        d["A"].append(dA)
        d["S"].append(dS)
    d.update(Q=dQ, K=dK, V=dV)
    g["Wq"], g["Wk"], g["Wv"] = c["U1"].T @ dQ, c["U1"].T @ dK, c["U1"].T @ dV
    d["U1"] = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T        # 問2：3本の枝の和
    d["X0_ln1"] = ln_bwd(d["U1"], c["U1"], c["ln1"]["sigma"])
    d["X0"] = d["X1"] + d["X0_ln1"]
    g["E"] = np.zeros_like(p["E"])
    g["P"] = np.zeros_like(p["P"])
    for row in range(n):                                              # 問7-3：使われた行に足し込む
        g["E"][inp[row]] += d["X0"][row]
        g["P"][posi[row]] += d["X0"][row]
    return g, d


def sgd(p, g, lr):
    return {k: p[k] - lr * g[k] for k in PARAM_NAMES}


# ---------------------------------------------------------------- Part 2 の分析（任意。k3_check.py が見つければ照合する）
# 行の並びは Part 2 と同じ：0..2 = 猫文 t0..t2、3..5 = 犬文 t0..t2。語彙 4 = 鳴く、5 = 吠える。

CAT_T2, DOG_T2, NAKU, HOERU = 2, 5, 4, 5


def _after_attention(p, X1):
    """X1 から先（LN2 → FFN → 残差 → Wout）だけを計算し直す。"""
    U2, _ = ln_fwd(X1)
    X2 = X1 + np.maximum(U2 @ p["W1"], 0.0) @ p["W2"]
    return X2 @ p["Wout"]


def ablate_t2(p, c):
    """5.4：t2 の Attention 出力を0にしたときの t2 の logits（2×V）。"""
    X1 = c["X1"].copy()
    X1[[CAT_T2, DOG_T2]] = c["X0"][[CAT_T2, DOG_T2]]
    return _after_attention(p, X1)[[CAT_T2, DOG_T2]]


def head_contributions(p, c):
    """5.5：ヘッドごとに [パターン項, 値項, 直接効果, パッチングの総効果]（H×4）。

    猫文 t2 と犬文 t2 のヘッド h の出力の差 ΔO = a猫·V猫 − a犬·V犬 = (a猫 − a犬)·V猫 + a犬·(V猫 − V犬)。
    それぞれを Wo_h → Wout の (鳴く − 吠える) 方向に射影したものがパターン項・値項、和が直接効果。
    パッチングの総効果：猫文 t2 の O_h を犬文 t2 のものに差し替えて最後まで計算し直したとき、(鳴く − 吠える) がどれだけ減るか。
    """
    r = p["Wout"][:, NAKU] - p["Wout"][:, HOERU]
    gap = lambda z: z[CAT_T2, NAKU] - z[CAT_T2, HOERU]
    out = []
    for h in range(H):
        a_cat, a_dog = c["A"][h][CAT_T2, 0:3], c["A"][h][DOG_T2, 3:6]
        V_cat, V_dog = c["V"][0:3, cols(h)], c["V"][3:6, cols(h)]
        to_gap = p["Wo"][cols(h), :] @ r
        pattern = (a_cat - a_dog) @ V_cat @ to_gap
        value = a_dog @ (V_cat - V_dog) @ to_gap
        direct = (c["O"][h][CAT_T2] - c["O"][h][DOG_T2]) @ to_gap
        O = [o.copy() for o in c["O"]]
        O[h][CAT_T2] = c["O"][h][DOG_T2]
        patched = _after_attention(p, c["X0"] + np.hstack(O) @ p["Wo"])
        out.append([pattern, value, direct, gap(c["logits"]) - gap(patched)])
    return np.array(out)


def qk_circuit(p, h):
    """8.1：Wq_h Wk_hᵀ（4×4）。u_q · M · u_k が「クエリ u_q がキー u_k をどれだけ見るか」（÷√2 の前）。"""
    return p["Wq"][:, cols(h)] @ p["Wk"][:, cols(h)].T


def ov_circuit(p, h):
    """8.1：Wv_h Wo_h Wout（4×V）。ソース u がヘッド h を通って各トークンの logit をいくら押すか（注意の重み 1 あたり）。"""
    return p["Wv"][:, cols(h)] @ p["Wo"][cols(h), :] @ p["Wout"]
