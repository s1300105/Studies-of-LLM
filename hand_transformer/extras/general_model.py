"""発展課題用：mini_gpt.py と同じ構造のモデルを、任意の語彙・複数文（バッチ）で動かす版。

mini_gpt.py は手計算と1対1で対応させるため入力を固定しているが、こちらは
  - 語彙サイズ・系列（複数可）を自由に指定できる
  - Attention / FFN / 残差 を個別に外せる（アブレーション）
  - SGD / Adam で多ステップ学習できる
"""
import numpy as np

DK = 2


def layernorm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    sigma = np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps)
    return xc / sigma, sigma


def layernorm_backward(dy, y, sigma):
    return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sigma


def softmax(s):
    e = np.exp(s - np.max(s, -1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def init_params(vocab_size, T, d=4, ff=8, scale=0.5, seed=0):
    rng = np.random.default_rng(seed)
    shapes = {"E": (vocab_size, d), "P": (T, d), "Wq": (d, d), "Wk": (d, d), "Wv": (d, d),
              "Wo": (d, d), "W1": (d, ff), "W2": (ff, d), "Wout": (d, vocab_size)}
    return {k: rng.normal(0, scale, s) for k, s in shapes.items()}


def forward(p, inp, tgt, use_attn=True, use_ffn=True):
    """inp, tgt: (B, T) の整数配列。"""
    inp, tgt = np.asarray(inp), np.asarray(tgt)
    B, T = inp.shape
    d = p["E"].shape[1]
    H = d // DK
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    c = {"inp": inp, "tgt": tgt, "mask": mask, "H": H}
    c["X0"] = p["E"][inp] + p["P"][None]
    c["U1"], c["s1"] = layernorm(c["X0"])
    c["Q"], c["K"], c["V"] = c["U1"] @ p["Wq"], c["U1"] @ p["Wk"], c["U1"] @ p["Wv"]
    c["A"], O = [], []
    for h in range(H):
        sl = slice(h * DK, (h + 1) * DK)
        S = c["Q"][..., sl] @ np.swapaxes(c["K"][..., sl], 1, 2) / np.sqrt(DK)
        A = softmax(np.where(mask, -np.inf, S))
        c["A"].append(A)
        O.append(A @ c["V"][..., sl])
    c["Ocat"] = np.concatenate(O, -1)
    c["Attn"] = c["Ocat"] @ p["Wo"] * use_attn
    c["X1"] = c["X0"] + c["Attn"]
    c["U2"], c["s2"] = layernorm(c["X1"])
    c["Z"] = c["U2"] @ p["W1"]
    c["R"] = np.maximum(c["Z"], 0)
    c["F"] = c["R"] @ p["W2"] * use_ffn
    c["X2"] = c["X1"] + c["F"]
    c["logits"] = c["X2"] @ p["Wout"]
    c["probs"] = softmax(c["logits"])
    bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
    c["loss"] = -np.log(c["probs"][bi, ti, tgt]).mean()
    c["use"] = (use_attn, use_ffn)
    return c


def backward(p, c):
    use_attn, use_ffn = c["use"]
    B, T = c["inp"].shape
    H, mask = c["H"], c["mask"]
    g = {}
    Y = np.zeros_like(c["probs"])
    bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
    Y[bi, ti, c["tgt"]] = 1
    dlog = (c["probs"] - Y) / (B * T)
    flat = lambda x: x.reshape(-1, x.shape[-1])
    g["Wout"] = flat(c["X2"]).T @ flat(dlog)
    dX2 = dlog @ p["Wout"].T
    dF = dX2 * use_ffn
    g["W2"] = flat(c["R"]).T @ flat(dF)
    dZ = (dF @ p["W2"].T) * (c["Z"] > 0)
    g["W1"] = flat(c["U2"]).T @ flat(dZ)
    dX1 = dX2 + layernorm_backward(dZ @ p["W1"].T, c["U2"], c["s2"])
    dAttn = dX1 * use_attn
    g["Wo"] = flat(c["Ocat"]).T @ flat(dAttn)
    dOcat = dAttn @ p["Wo"].T
    dQ, dK, dV = np.zeros_like(c["Q"]), np.zeros_like(c["K"]), np.zeros_like(c["V"])
    for h in range(H):
        sl = slice(h * DK, (h + 1) * DK)
        A = c["A"][h]
        dA = np.where(mask, 0, dOcat[..., sl] @ np.swapaxes(c["V"][..., sl], 1, 2))
        dV[..., sl] = np.swapaxes(A, 1, 2) @ dOcat[..., sl]
        dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(DK)
        dQ[..., sl] = dS @ c["K"][..., sl]
        dK[..., sl] = np.swapaxes(dS, 1, 2) @ c["Q"][..., sl]
    for name, dM in (("Wq", dQ), ("Wk", dK), ("Wv", dV)):
        g[name] = flat(c["U1"]).T @ flat(dM)
    dU1 = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
    dX0 = dX1 + layernorm_backward(dU1, c["U1"], c["s1"])
    g["P"] = dX0.sum(0)
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], c["inp"], dX0)
    return g


def train(p, inp, tgt, steps=2000, lr=0.05, optimizer="adam", use_attn=True, use_ffn=True, log_every=None):
    p = {k: v.copy() for k, v in p.items()}
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v2 = {k: np.zeros_like(v) for k, v in p.items()}
    hist = []
    for s in range(1, steps + 1):
        c = forward(p, inp, tgt, use_attn, use_ffn)
        hist.append(c["loss"])
        g = backward(p, c)
        for k in p:
            if optimizer == "adam":
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
                p[k] -= lr * (m[k] / (1 - 0.9 ** s)) / (np.sqrt(v2[k] / (1 - 0.999 ** s)) + 1e-8)
            else:
                p[k] -= lr * g[k]
        if log_every and s % log_every == 0:
            print(f"  step {s:5d}  loss {c['loss']:.4f}")
    return p, forward(p, inp, tgt, use_attn, use_ffn), hist


def grad_check(p, inp, tgt, eps=1e-6, **kw):
    c = forward(p, inp, tgt, **kw)
    g = backward(p, c)
    worst = 0.0
    for k in p:
        for idx in np.ndindex(p[k].shape):
            q = {kk: vv.copy() for kk, vv in p.items()}
            q[k][idx] += eps
            lp = forward(q, inp, tgt, **kw)["loss"]
            q[k][idx] -= 2 * eps
            lm = forward(q, inp, tgt, **kw)["loss"]
            worst = max(worst, abs((lp - lm) / (2 * eps) - g[k][idx]))
    return worst
