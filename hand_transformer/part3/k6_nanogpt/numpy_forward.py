#!/usr/bin/env python3
"""K6：学習した nanoGPT のチェックポイントを、Part 1 の記号（E, P, Wq, …, Wout）に読み替えて NumPy で順伝播し、
PyTorch の logits と一致することを確かめる。

    cd ~/nanoGPT
    python3 /path/to/part3/k6_nanogpt/numpy_forward.py --out_dir out-base --start 吾輩は猫である

PyTorch が必要（チェックポイントを読むためと、比べる相手の logits を出すため）。nanoGPT の model.py を import するので、
nanoGPT のフォルダの中で実行するか --nanogpt でフォルダを指定する。patch_nanogpt.py のスイッチ（use_wpe 等）にも対応。

読み替え（nn.Linear の weight は (出力, 入力) の形で y = x·Wᵀ + b なので、行ベクトル規約の W は weight の転置）:
    E    = transformer.wte.weight                    (語彙 × C)
    P    = transformer.wpe.weight[:T]                (T × C)
    Wq | Wk | Wv = h.{l}.attn.c_attn.weight.T を列で3等分   (C × C が3つ)
    Wo   = h.{l}.attn.c_proj.weight.T                (C × C)
    W1   = h.{l}.mlp.c_fc.weight.T                   (C × 4C)
    W2   = h.{l}.mlp.c_proj.weight.T                 (4C × C)
    LN   = (weight, bias) が γ, β。ε = 1e-5
    Wout = lm_head.weight.T = Eᵀ                     （重み共有：同じテンソル）
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np


def layernorm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)   # 母分散（÷C）。PyTorch の LayerNorm と同じ
    y = (x - mu) / np.sqrt(var + eps)
    return y * g + (0.0 if b is None else b)


_erf = np.vectorize(math.erf)


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))   # nn.GELU() の既定（tanh 近似ではない厳密版）


def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nanogpt", default=".", help="nanoGPT のフォルダ（model.py がある所）")
    ap.add_argument("--out_dir", default="out-base")
    ap.add_argument("--dataset", default=None, help="meta.pkl を読む data/<名前>（既定はチェックポイントの設定）")
    ap.add_argument("--start", default="吾輩は猫である")
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.nanogpt))
    import torch
    from model import GPT, GPTConfig

    ckpt = torch.load(os.path.join(args.nanogpt, args.out_dir, "ckpt.pt"), map_location="cpu")
    model_args = ckpt["model_args"]
    model = GPT(GPTConfig(**model_args))
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.eval()
    print("model_args:", model_args)
    print("lm_head.weight と wte.weight は同じテンソルか:",
          model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())

    dataset = args.dataset or ckpt["config"]["dataset"]
    with open(os.path.join(args.nanogpt, "data", dataset, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    ids = [meta["stoi"][c] for c in args.start]
    T = len(ids)

    W = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    C, nh = model_args["n_embd"], model_args["n_head"]
    hs = C // nh
    use_wpe = model_args.get("use_wpe", True)
    causal = model_args.get("causal", True)
    use_ln_f = model_args.get("use_ln_f", True)
    get = lambda k: W.get(k)                                   # bias=False なら None

    print(f"\n部品の対応（C = {C}、ヘッド {nh} 本 × {hs} 次元、層 {model_args['n_layer']}、語彙 {W['transformer.wte.weight'].shape[0]}）")
    for k, v in W.items():
        if k.startswith("transformer.h.") and not k.startswith("transformer.h.0."):
            continue
        print(f"  {k:36s} {str(tuple(v.shape)):14s}")
    print("  （transformer.h.1. 以降も h.0 と同じ形）")

    # ---- 順伝播（Part 1 の 2.1〜5.2 と同じ順）
    E, Pm = W["transformer.wte.weight"], W["transformer.wpe.weight"]
    X = E[ids] + (Pm[:T] if use_wpe else 0.0)                                   # 2.1
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    for l in range(model_args["n_layer"]):
        pre = f"transformer.h.{l}."
        U1 = layernorm(X, W[pre + "ln_1.weight"], get(pre + "ln_1.bias"))       # 2.2–2.3
        Wqkv = W[pre + "attn.c_attn.weight"].T                                   # C × 3C
        bqkv = get(pre + "attn.c_attn.bias")
        QKV = U1 @ Wqkv + (0.0 if bqkv is None else bqkv)
        Q, K, V = QKV[:, :C], QKV[:, C:2 * C], QKV[:, 2 * C:]                    # 3.1
        O = []
        for h in range(nh):
            sl = slice(h * hs, (h + 1) * hs)
            S = Q[:, sl] @ K[:, sl].T / math.sqrt(hs)                            # 3.2
            if causal:
                S = np.where(mask, -np.inf, S)
            O.append(softmax(S) @ V[:, sl])                                      # 3.3–3.4
        Ocat = np.concatenate(O, axis=1)
        bo = get(pre + "attn.c_proj.bias")
        X = X + Ocat @ W[pre + "attn.c_proj.weight"].T + (0.0 if bo is None else bo)   # 3.5, 4.1
        U2 = layernorm(X, W[pre + "ln_2.weight"], get(pre + "ln_2.bias"))       # 4.2
        b1, b2 = get(pre + "mlp.c_fc.bias"), get(pre + "mlp.c_proj.bias")
        Z = U2 @ W[pre + "mlp.c_fc.weight"].T + (0.0 if b1 is None else b1)      # 4.3（ReLU の代わりに GELU）
        F = gelu(Z) @ W[pre + "mlp.c_proj.weight"].T + (0.0 if b2 is None else b2)   # 4.4
        X = X + F                                                                # 4.5
    if use_ln_f:
        X = layernorm(X, W["transformer.ln_f.weight"], get("transformer.ln_f.bias"))   # Part 1 には無い最終 LN
    logits_np = X @ W["lm_head.weight"].T                                        # 5.1（Wout = Eᵀ）

    with torch.no_grad():
        idx = torch.tensor([ids])
        logits_pt, _ = model(idx, targets=idx)                                   # targets を渡すと全位置の logits が出る
    logits_pt = logits_pt[0].double().numpy()
    diff = np.abs(logits_np - logits_pt).max()
    print(f"\n入力 {args.start!r}（{T} 文字）で、NumPy と PyTorch の logits の最大差 = {diff:.2e}（float32 の丸め程度なら一致）")
    p = softmax(logits_np)
    print("各位置の次の文字の予測（上位3つと確率）:")
    for t in range(T):
        top = np.argsort(-p[t])[:3]
        print(f"  {args.start[:t + 1]!r:>16} → " + "  ".join(f"{meta['itos'][int(i)]!r} {p[t, i]:.3f}" for i in top))


if __name__ == "__main__":
    main()
