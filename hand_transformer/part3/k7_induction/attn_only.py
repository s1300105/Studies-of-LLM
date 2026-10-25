#!/usr/bin/env python3
"""K7/K8 共通：Attention だけの Transformer（LayerNorm なし・FFN なし・バイアスなし）の NumPy 実装。

順伝播・手書きの逆伝播・数値微分による検算・Adam を、Part 1 と同じ記号で書いてある。

モデル（行ベクトル規約、b = バッチ、t = 位置）:
    X0 = E[ids] + P[t]                                  残差ストリーム（B×T×D）
    層 l = 0, 1, ...、ヘッド h ごとに:
        Q = X_l Wq, K = X_l Wk, V = X_l Wv              （Wq, Wk: D×dk、Wv: D×dv）
        S = Q Kᵀ / √dk + 因果マスク, A = softmax(S), O = A V
        X_{l+1} = X_l + Σ_h O Wo                        （Wo: dv×D）
    logits = X_L Wout                                   （Wout: D×語彙）
    L = 損失マスク付きの交差エントロピーの平均

Part 1 との違い：LayerNorm と FFN が無い（層 = Attention だけ）、層が2つ、ヘッドごとの行列を別々に持つ。
同じ層のヘッドは同じ X_l を読み、出力を足し合わせて X_{l+1} に書く（Part 1 の [O0|O1]·Wo と同じこと）。

パラメータは平らな dict: "E", "P", "Wout", "L0H1.Wq" のように「層・ヘッド・行列」で名前を付ける。
ヘッドごとに dk, dv が違ってもよい（K8 の手作りの重みで使う）。

使い方:
    python3 attn_only.py --gradcheck      # 逆伝播を中心差分で検算（K7 の最初にやる）
"""
import argparse

import numpy as np

HEAD_MATS = ("Wq", "Wk", "Wv", "Wo")


def hname(l, h, m):
    return f"L{l}H{h}.{m}"


def init_params(vocab, T, D, n_layers, n_heads, dk, rng, std=0.02, std_emb=None):
    """乱数で初期化する（GPT-2 / nanoGPT と同じ正規分布、標準偏差 std）。"""
    std_emb = std if std_emb is None else std_emb
    p = {"E": rng.normal(0, std_emb, (vocab, D)),
         "P": rng.normal(0, std_emb, (T, D)),
         "Wout": rng.normal(0, std, (D, vocab))}
    for l in range(n_layers):
        for h in range(n_heads):
            p[hname(l, h, "Wq")] = rng.normal(0, std, (D, dk))
            p[hname(l, h, "Wk")] = rng.normal(0, std, (D, dk))
            p[hname(l, h, "Wv")] = rng.normal(0, std, (D, dk))
            p[hname(l, h, "Wo")] = rng.normal(0, std, (dk, D))
    return p


def structure(p):
    """パラメータ名から (層の数, 各層のヘッド数) を読み取る。"""
    heads = {}
    for k in p:
        if k.startswith("L"):
            l, h = k.split(".")[0][1:].split("H")
            heads.setdefault(int(l), set()).add(int(h))
    n_layers = len(heads)
    return n_layers, [len(heads[l]) for l in range(n_layers)]


def xt_dy(X, dY):
    """Σ_b Σ_t X[b,t]ᵀ dY[b,t]（= 行列積の重みの勾配 Xᵀ dY を、バッチの全系列について足したもの）。"""
    return X.reshape(-1, X.shape[-1]).T @ dY.reshape(-1, dY.shape[-1])


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def forward(p, ids, tgt=None, lmask=None, ablate=(), keep_cache=True):
    """順伝播。

    ids: (B, T) の整数、tgt: (B, T) の正解、lmask: (B, T) の損失に入れる位置（1/0）
    ablate: 出力を 0 にする (層, ヘッド) の集合（ゼロ・アブレーション）
    戻り値 c: 中間値の dict（c["A"][(l, h)] が注意の重み、c["loss"] が損失）
    """
    B, T = ids.shape
    n_layers, n_heads = structure(p)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)          # True = 未来（見てはいけない）
    c = {"ids": ids, "X": [], "Q": {}, "K": {}, "V": {}, "A": {}, "O": {}, "ablate": set(ablate)}
    X = p["E"][ids] + p["P"][None, :T]
    c["X"].append(X)
    for l in range(n_layers):
        Xn = X.copy()
        for h in range(n_heads[l]):
            Wq, Wk, Wv, Wo = (p[hname(l, h, m)] for m in HEAD_MATS)
            Q, K, V = X @ Wq, X @ Wk, X @ Wv
            S = Q @ K.transpose(0, 2, 1) / np.sqrt(Wq.shape[1])
            S = np.where(mask, -np.inf, S)
            A = softmax(S)
            O = A @ V
            if (l, h) not in c["ablate"]:
                Xn = Xn + O @ Wo
            if keep_cache:
                c["Q"][(l, h)], c["K"][(l, h)], c["V"][(l, h)] = Q, K, V
                c["O"][(l, h)] = O
            c["A"][(l, h)] = A
        X = Xn
        c["X"].append(X)
    c["logits"] = X @ p["Wout"]
    if tgt is not None:
        lg = c["logits"]
        m = lg.max(-1, keepdims=True)
        lse = (np.log(np.exp(lg - m).sum(-1, keepdims=True)) + m)[..., 0]
        nll = lse - np.take_along_axis(lg, tgt[..., None], -1)[..., 0]   # = −log p[正解]
        lmask = np.ones_like(nll) if lmask is None else lmask
        c["nll"], c["lmask"], c["tgt"] = nll, lmask, tgt
        c["loss"] = float((nll * lmask).sum() / lmask.sum())
    return c


def backward(p, c):
    """手書きの逆伝播。Part 1 の 6.1〜6.9 と同じ部品だけを使う（LN・ReLU が無いぶん短い）。"""
    ids, tgt, lmask = c["ids"], c["tgt"], c["lmask"]
    n_layers, n_heads = structure(p)
    g = {}
    probs = softmax(c["logits"])
    Y = np.zeros_like(probs)
    np.put_along_axis(Y, tgt[..., None], 1.0, -1)
    dlogits = (probs - Y) * (lmask / lmask.sum())[..., None]        # (p − y)/N（N = 損失に入れた位置の数）
    XL = c["X"][-1]
    g["Wout"] = xt_dy(XL, dlogits)
    dX = dlogits @ p["Wout"].T
    for l in reversed(range(n_layers)):
        X = c["X"][l]
        dXin = dX.copy()                                            # 残差：勾配はそのまま下へ
        for h in range(n_heads[l]):
            Wq, Wk, Wv, Wo = (p[hname(l, h, m)] for m in HEAD_MATS)
            Q, K, V, A, O = c["Q"][(l, h)], c["K"][(l, h)], c["V"][(l, h)], c["A"][(l, h)], c["O"][(l, h)]
            if (l, h) in c["ablate"]:
                for m, W in zip(HEAD_MATS, (Wq, Wk, Wv, Wo)):
                    g[hname(l, h, m)] = np.zeros_like(W)
                continue
            scale = 1.0 / np.sqrt(Wq.shape[1])
            g[hname(l, h, "Wo")] = xt_dy(O, dX)
            dO = dX @ Wo.T
            dA = dO @ V.transpose(0, 2, 1)
            dV = A.transpose(0, 2, 1) @ dO
            dS = A * (dA - (A * dA).sum(-1, keepdims=True))         # softmax の逆（マスクの所は A = 0 なので 0）
            dQ = dS @ K * scale
            dK = dS.transpose(0, 2, 1) @ Q * scale
            g[hname(l, h, "Wq")] = xt_dy(X, dQ)
            g[hname(l, h, "Wk")] = xt_dy(X, dK)
            g[hname(l, h, "Wv")] = xt_dy(X, dV)
            dXin += dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T               # 分岐：Q, K, V の3本の勾配を足す
        dX = dXin
    T = ids.shape[1]
    g["P"] = np.zeros_like(p["P"])
    g["P"][:T] = dX.sum(0)
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], ids, dX)                                      # 表引き：使われた行に足し込む
    return g


class Adam:
    """Adam（Kingma & Ba 2015）。重み減衰なし。"""

    def __init__(self, p, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}

    def step(self, p, g):
        self.t += 1
        for k in p:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g[k] ** 2
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def gradcheck(seed=0, h=1e-6, verbose=True):
    """小さなモデルで、全パラメータの勾配を中心差分 (L(θ+h) − L(θ−h))/2h と照合する。"""
    rng = np.random.default_rng(seed)
    vocab, T, D, B = 7, 6, 8, 3
    p = init_params(vocab, T, D, n_layers=2, n_heads=2, dk=4, rng=rng, std=0.5)
    ids = rng.integers(0, vocab, (B, T))
    tgt = rng.integers(0, vocab, (B, T))
    lmask = (rng.random((B, T)) < 0.7).astype(float)
    lmask[0, 0] = 1.0
    worst = 0.0
    for ablate in [(), ((0, 1),)]:
        c = forward(p, ids, tgt, lmask, ablate=ablate)
        g = backward(p, c)
        for k in p:
            num = np.zeros_like(p[k])
            for idx in np.ndindex(p[k].shape):
                old = p[k][idx]
                p[k][idx] = old + h
                lp = forward(p, ids, tgt, lmask, ablate=ablate, keep_cache=False)["loss"]
                p[k][idx] = old - h
                lm = forward(p, ids, tgt, lmask, ablate=ablate, keep_cache=False)["loss"]
                p[k][idx] = old
                num[idx] = (lp - lm) / (2 * h)
            err = np.abs(num - g[k]).max()
            scale = max(np.abs(num).max(), 1e-12)
            worst = max(worst, err / scale)
            if verbose:
                tag = f"ablate={list(ablate)}" if ablate else "通常"
                print(f"  {tag:18s} {k:10s} 形 {str(p[k].shape):8s} 最大|手書き−数値| = {err:.2e}"
                      f"（最大|勾配| {scale:.2e}）")
    ok = worst < 1e-6
    print(f"最大の相対誤差 {worst:.2e} → {'一致' if ok else '不一致'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gradcheck", action="store_true", help="逆伝播を数値微分で検算する")
    args = ap.parse_args()
    if args.gradcheck:
        raise SystemExit(0 if gradcheck() else 1)
    ap.print_help()
