"""発展課題 C1：Pre-LN と Post-LN で、深いモデルの勾配の届き方を比べる。

本編は Pre-LN（GPT-2 型）：  X ← X + Attn(LN(X)),  X ← X + FFN(LN(X)),  最後に LN → Wout
元論文（Vaswani et al. 2017）は Post-LN：  X ← LN(X + Attn(X)),  X ← LN(X + FFN(X))

Pre-LN の残差ストリームは LN を通らない一本道なので、出力側の勾配が足し算だけで埋め込みまで届く
（本編の 6.3, 6.8 で「dX2 がそのまま dX1, dX0 に含まれる」のを手で確かめたのと同じ）。
Post-LN では1サブ層ごとに LN の逆伝播（÷σ と2つの射影）を通るので、層を重ねると届き方が変わる。

本編と同じ部品（d_k=2 ではなく d_k=8、バイアスなし、ReLU、γ=1・β=0 の LN）で深いモデルを作り、
  [2] 初期化直後の勾配の大きさを、埋め込み E と各層について測る
  [3] 「1つ前のトークンを答える」タスク（Attention が必須）を Adam で学習させ、warmup の有無で比べる
逆伝播は手書き（数値微分で検算する）。

実行:  python3 extras/pre_post_ln.py      （約20秒）
"""
import numpy as np

from general_model import layernorm, layernorm_backward, softmax

V, T, B, D, H, FF = 16, 8, 16, 32, 4, 128
DK = D // H


def init(n_layers, std, seed):
    rng = np.random.default_rng(seed)
    r = lambda *s: rng.normal(0, std, s)
    layers = [{"Wq": r(D, D), "Wk": r(D, D), "Wv": r(D, D), "Wo": r(D, D), "W1": r(D, FF), "W2": r(FF, D)}
              for _ in range(n_layers)]
    return {"E": r(V, D), "P": r(T, D), "Wout": r(D, V), "layers": layers}


MASK = np.triu(np.ones((T, T), dtype=bool), 1)


def attn_fwd(U, w):
    q, k, v = (U @ w[n] for n in ("Wq", "Wk", "Wv"))
    split = lambda x: x.reshape(B, T, H, DK).transpose(0, 2, 1, 3)        # (B, H, T, DK)
    qh, kh, vh = split(q), split(k), split(v)
    A = softmax(np.where(MASK, -np.inf, qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(DK)))
    O = (A @ vh).transpose(0, 2, 1, 3).reshape(B, T, D)
    return O @ w["Wo"], (U, qh, kh, vh, A, O)


def attn_bwd(dout, w, cache, g):
    U, qh, kh, vh, A, O = cache
    flat = lambda x: x.reshape(-1, x.shape[-1])
    g["Wo"] = flat(O).T @ flat(dout)
    dO = (dout @ w["Wo"].T).reshape(B, T, H, DK).transpose(0, 2, 1, 3)
    dA = dO @ vh.transpose(0, 1, 3, 2)
    dv = A.transpose(0, 1, 3, 2) @ dO
    dS = A * (dA - (dA * A).sum(-1, keepdims=True)) / np.sqrt(DK)
    dq, dk = dS @ kh, dS.transpose(0, 1, 3, 2) @ qh
    merge = lambda x: x.transpose(0, 2, 1, 3).reshape(B, T, D)
    dU = 0
    for n, dm in (("Wq", dq), ("Wk", dk), ("Wv", dv)):
        g[n] = flat(U).T @ flat(merge(dm))
        dU = dU + merge(dm) @ w[n].T
    return dU


def ffn_fwd(U, w):
    Z = U @ w["W1"]
    R = np.maximum(Z, 0)
    return R @ w["W2"], (U, Z, R)


def ffn_bwd(dout, w, cache, g):
    U, Z, R = cache
    flat = lambda x: x.reshape(-1, x.shape[-1])
    g["W2"] = flat(R).T @ flat(dout)
    dZ = (dout @ w["W2"].T) * (Z > 0)
    g["W1"] = flat(U).T @ flat(dZ)
    return dZ @ w["W1"].T


def loss_and_grads(p, inp, tgt, post_ln, need_grad=True):
    X = p["E"][inp] + p["P"][None]
    tape = []
    for w in p["layers"]:
        for fwd, bwd in ((attn_fwd, attn_bwd), (ffn_fwd, ffn_bwd)):
            if post_ln:                      # X ← LN(X + f(X))
                out, cache = fwd(X, w)
                X, s = layernorm(X + out)
                tape.append((bwd, w, cache, X, s))
            else:                            # X ← X + f(LN(X))
                U, s = layernorm(X)
                out, cache = fwd(U, w)
                tape.append((bwd, w, cache, U, s))
                X = X + out
    if post_ln:
        Xf, sf = X, None                     # Post-LN の出力はすでに LN 済み
    else:
        Xf, sf = layernorm(X)                # Pre-LN は最後に LN を1つ足す（GPT-2 と同じ）
    logits = Xf @ p["Wout"]
    probs = softmax(logits)
    bi, ti = np.meshgrid(np.arange(B), np.arange(T), indexing="ij")
    loss = -np.log(probs[bi, ti, tgt]).mean()
    if not need_grad:
        return loss, None
    dlog = probs.copy()
    dlog[bi, ti, tgt] -= 1
    dlog /= B * T
    g = {"Wout": Xf.reshape(-1, D).T @ dlog.reshape(-1, V), "layers": [{} for _ in p["layers"]]}
    dX = dlog @ p["Wout"].T
    if not post_ln:
        dX = layernorm_backward(dX, Xf, sf)
    for i, (bwd, w, cache, Y, s) in reversed(list(enumerate(tape))):
        gl = g["layers"][i // 2]
        if post_ln:                          # Y = LN(X_in + f(X_in))
            dsum = layernorm_backward(dX, Y, s)
            dX = dsum + bwd(dsum, w, cache, gl)
        else:                                # X_out = X_in + f(LN(X_in))、Y = LN(X_in)
            dX = dX + layernorm_backward(bwd(dX, w, cache, gl), Y, s)
    g["P"] = dX.sum(0)
    g["E"] = np.zeros_like(p["E"])
    np.add.at(g["E"], inp, dX)
    return loss, g


def grad_check(post_ln):
    p = init(2, 0.3, seed=1)
    rng = np.random.default_rng(2)
    inp, tgt = rng.integers(0, V, (B, T)), rng.integers(0, V, (B, T))
    _, g = loss_and_grads(p, inp, tgt, post_ln)
    worst, h = 0.0, 1e-5
    for get in (lambda q: q["E"], lambda q: q["Wout"], lambda q: q["layers"][0]["Wq"], lambda q: q["layers"][1]["W1"]):
        for idx in [(0, 0), (1, 2), (3, 1)]:
            ana = get(g)[idx]
            get(p)[idx] += h
            lp = loss_and_grads(p, inp, tgt, post_ln, need_grad=False)[0]
            get(p)[idx] -= 2 * h
            lm = loss_and_grads(p, inp, tgt, post_ln, need_grad=False)[0]
            get(p)[idx] += h
            worst = max(worst, abs((lp - lm) / (2 * h) - ana))
    return worst


def norms(n_layers, std, post_ln, seeds=5):
    """初期化直後の ‖dE‖ と、各層のパラメータ勾配のノルム（seed 平均）。"""
    nE, per_layer = [], []
    for seed in range(seeds):
        p = init(n_layers, std, seed)
        rng = np.random.default_rng(100 + seed)
        inp, tgt = rng.integers(0, V, (B, T)), rng.integers(0, V, (B, T))
        _, g = loss_and_grads(p, inp, tgt, post_ln)
        nE.append(np.linalg.norm(g["E"]))
        per_layer.append([np.sqrt(sum((x ** 2).sum() for x in gl.values())) for gl in g["layers"]])
    return np.mean(nE), np.mean(per_layer, axis=0)


def leaves(tree):
    """パラメータ（または勾配）の木を、決まった順序の (辞書, キー) のリストにする。"""
    out = [(tree, k) for k in ("E", "P", "Wout")]
    return out + [(layer, k) for layer in tree["layers"] for k in ("Wq", "Wk", "Wv", "Wo", "W1", "W2")]


def train_prev_token(post_ln, lr, warmup, seed, n_layers=6, steps=400):
    """毎ステップ新しいランダム列で「位置 t の正解 = 位置 t−1 の入力」を Adam で学習。最後の20ステップの平均損失。"""
    p = init(n_layers, 1 / np.sqrt(D), seed)
    rng = np.random.default_rng(50 + seed)
    m = [np.zeros_like(d[k]) for d, k in leaves(p)]
    v = [np.zeros_like(d[k]) for d, k in leaves(p)]
    hist = []
    for s in range(1, steps + 1):
        inp = rng.integers(0, V, (B, T))
        tgt = np.concatenate([inp[:, :1], inp[:, :-1]], 1)
        loss, g = loss_and_grads(p, inp, tgt, post_ln)
        hist.append(loss)
        cur = lr * min(1.0, s / warmup) if warmup else lr     # 線形 warmup
        for j, ((d, k), (gd, gk)) in enumerate(zip(leaves(p), leaves(g))):
            m[j] = 0.9 * m[j] + 0.1 * gd[gk]
            v[j] = 0.999 * v[j] + 0.001 * gd[gk] ** 2
            d[k] -= cur * (m[j] / (1 - 0.9 ** s)) / (np.sqrt(v[j] / (1 - 0.999 ** s)) + 1e-8)
    return np.mean(hist[:1]), np.mean(hist[-20:])


def main():
    print(f"[1] 逆伝播の数値微分チェック（2層の小さな例）: Pre-LN {grad_check(False):.1e}, Post-LN {grad_check(True):.1e}")
    print(f"\n設定: 語彙{V}, 系列長{T}, バッチ{B}, d_model {D}, ヘッド{H}, FFN {FF}")
    print("\n[2] 初期化直後の勾配ノルム（正解はランダム、seed 5つの平均）")
    std = 1 / np.sqrt(D)
    print(f"  初期化 N(0, 1/d) = N(0, {std:.3f}²)、層数を変える")
    print("    層数 |  Pre-LN: ‖dE‖  1層目  最終層 |  Post-LN: ‖dE‖  1層目  最終層")
    for n_layers in (1, 2, 4, 8, 16, 32):
        row = f"    {n_layers:4d} |"
        for post_ln in (False, True):
            nE, pl = norms(n_layers, std, post_ln)
            row += f"          {nE:.3f}  {pl[0]:.3f}  {pl[-1]:.3f} |"
        print(row)
    for std, label in ((1 / np.sqrt(D), "N(0, 1/d)"), (0.02, "N(0, 0.02²)（GPT-2 の初期化の標準偏差）")):
        print(f"  6層の層ごとの勾配ノルム（入力側 → 出力側）、初期化 {label}")
        for post_ln in (False, True):
            _, pl = norms(6, std, post_ln)
            print(f"    {'Post-LN' if post_ln else 'Pre-LN '}  " + "  ".join(f"{x:.3f}" for x in pl))

    print(f"\n[3] 6層で「1つ前のトークンを答える」タスクを Adam（lr=0.01）で400ステップ。最後の20ステップの平均損失"
          f"（当てずっぽうなら log{V} = {np.log(V):.2f}）")
    for post_ln, warmup in ((False, 0), (True, 0), (True, 200)):
        res = [train_prev_token(post_ln, 0.01, warmup, seed) for seed in range(3)]
        print(f"  {'Post-LN' if post_ln else 'Pre-LN '}  warmup {warmup:3d} ステップ:  "
              + "  ".join(f"seed{s} {a:.3f} → {b:.3f}" for s, (a, b) in enumerate(res)))


if __name__ == "__main__":
    main()
