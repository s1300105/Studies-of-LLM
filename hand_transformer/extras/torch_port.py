"""発展課題 S6：同じモデルを PyTorch で書き、autograd の勾配を手計算（mini_gpt.py）と突き合わせる。

手で導いた逆伝播（6.1〜6.9）は、PyTorch では loss.backward() の1行になる。
「自分の手計算 = mini_gpt.py = PyTorch の autograd」が一致すれば、逆伝播を本当に理解できたことの確認になる。
ついでに、実際の LLM 学習で使う AdamW が、素の SGD とどう違うか（1ステップ目の更新量）を見る。

対応関係:
    LN（γ=1, β=0, ε=0, 母分散）     → F.layer_norm(x, (4,), eps=0.0)
    マスク付き Attention（÷√d_k）    → F.scaled_dot_product_attention(q, k, v, is_causal=True)
    3位置の交差エントロピーの平均    → F.cross_entropy(logits, targets)
    θ ← θ − lr·dθ                    → torch.optim.SGD(params, lr=1.0)

実行:  pip install torch   のあと   python3 extras/torch_port.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("PyTorch が入っていません。pip install torch してから実行してください。")

torch.set_default_dtype(torch.float64)   # 手計算・NumPy と桁まで比べるため倍精度にする
INP = torch.tensor(mg.INP)
TGT = torch.tensor(mg.TGT)


def forward(P):
    x0 = P["E"][INP] + P["P"]                                          # 2.1
    u1 = F.layer_norm(x0, (mg.D,), eps=0.0)                             # 2.3
    q, k, v = (
        (u1 @ P[w]).view(mg.T, mg.H, mg.DK).transpose(0, 1)             # (ヘッド, 位置, d_k)
        for w in ("Wq", "Wk", "Wv")
    )
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)         # 3.2〜3.4（scale は既定で 1/√d_k）
    x1 = x0 + o.transpose(0, 1).reshape(mg.T, mg.D) @ P["Wo"]           # 3.5, 4.1
    z = F.layer_norm(x1, (mg.D,), eps=0.0) @ P["W1"]                    # 4.2, 4.3
    x2 = x1 + F.relu(z) @ P["W2"]                                       # 4.4, 4.5
    logits = x2 @ P["Wout"]                                             # 5.1
    return F.cross_entropy(logits, TGT), logits                         # 5.2, 5.3


def load():
    p, lr = mg.load_params()
    return {k: torch.tensor(v, requires_grad=True) for k, v in p.items()}, p, lr


def main():
    P, p_np, lr = load()
    loss, _ = forward(P)
    loss.backward()                                                     # 6.1〜6.9 がこの1行
    g_np, _ = mg.backward(p_np, mg.forward(p_np))
    print(f"L（PyTorch）= {loss.item():.4f}   L（mini_gpt.py）= {mg.forward(p_np)['loss']:.4f}")
    print("autograd の勾配と mini_gpt.py の勾配の最大差:")
    print("  " + "  ".join(f"{k} {np.abs(P[k].grad.numpy() - g_np[k]).max():.1e}" for k in mg.PARAMS))

    opt = torch.optim.SGD(P.values(), lr=lr)
    opt.step()
    with torch.no_grad():
        loss1, logits1 = forward(P)
    print(f"\ntorch.optim.SGD(lr={lr:g}) で1ステップ: L' = {loss1.item():.4f}  予測: "
          + " ".join(mg.VOCAB[i] for i in logits1.argmax(-1).tolist()))

    print("\nAdamW（weight_decay=0.01 は PyTorch の既定値）の1ステップ目の更新量 |Δθ|:")
    for alr in (0.01, 0.1):
        P, _, _ = load()
        before = {k: v.detach().clone() for k, v in P.items()}
        opt = torch.optim.AdamW(P.values(), lr=alr)
        forward(P)[0].backward()
        grads = torch.cat([P[k].grad.flatten() for k in mg.PARAMS]).abs()
        opt.step()
        delta = torch.cat([(P[k].detach() - before[k]).flatten() for k in mg.PARAMS]).abs()
        nz = grads > 1e-10                     # 数学的に0の勾配も float では 1e-17 程度の誤差が残るので閾値で分ける
        print(f"  lr={alr:<5g} |勾配|>1e-10 の {int(nz.sum())} 個: |Δθ|/lr = {(delta[nz] / alr).min():.3f}〜{(delta[nz] / alr).max():.3f}"
              f"（|勾配| は {grads[nz].min():.1e}〜{grads[nz].max():.2f}、{grads[nz].max() / grads[nz].min():.0f} 倍の開きがあるのに）"
              f"  / 勾配≈0 の {int((~nz).sum())} 個: |Δθ|/lr ≤ {(delta[~nz] / alr).max():.3f}（weight decay の分だけ）")

    print("\n損失の推移（各20ステップ）")
    runs = [("SGD   lr=1", torch.optim.SGD, 1.0), ("AdamW lr=0.01", torch.optim.AdamW, 0.01),
            ("AdamW lr=0.1", torch.optim.AdamW, 0.1)]
    for name, cls, alr in runs:
        P, _, _ = load()
        opt = cls(P.values(), lr=alr)
        hist = []
        for _ in range(21):
            opt.zero_grad()
            loss, _ = forward(P)
            hist.append(loss.item())
            loss.backward()
            opt.step()
        print(f"  {name:<14}" + " ".join(f"{hist[s]:.4f}" for s in (0, 1, 2, 5, 10, 20)) + "   （step 0, 1, 2, 5, 10, 20）")


if __name__ == "__main__":
    main()
