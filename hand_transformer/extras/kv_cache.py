"""発展課題 S2：自己回帰生成と KV キャッシュ。

学習（teacher forcing）では3語を一度に入れたが、生成では <s> から1語ずつ予測して入力に足していく。
素朴にやると、1語足すたびに全位置の Q, K, V と Attention を最初から計算し直すことになる。

因果マスクのおかげで「位置 t の出力は位置 0..t の入力だけで決まり、後ろに語を足しても変わらない」。
だから過去の位置の K, V は一度計算したら使い回せる（KV キャッシュ）。新しい位置では
  q, k, v を1行だけ計算 → k, v をキャッシュに追加 → q とキャッシュ内の全 K でスコア → softmax → V の加重和
とすればよく、マスクは不要（キャッシュには過去しか入っていないので）。

このスクリプトでは
  1. 手計算で求めた1ステップ更新後の重み（7.1）で <s> から貪欲生成する
  2. KV キャッシュ版の logits が、毎回全部計算し直した logits と一致することを確かめる
  3. 4語目を生成しようとすると、位置埋め込み P が3行しか無いので止まることを確かめる
  4. 計算量（Q/K/V の行数とスコアの個数）を数える

実行:  python3 extras/kv_cache.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mini_gpt as mg  # noqa: E402
from general_model import forward  # noqa: E402

DK = mg.DK


def full_recompute(p, seq):
    """毎回すべての位置を計算し直す（general_model.forward、LN の ε=0 で mini_gpt と同じ式）。"""
    return forward(p, [seq], None, ln_eps=0)["logits"][0, -1]


def ln(x):
    xc = x - x.mean()
    return xc / np.sqrt((xc ** 2).mean())


def cached_step(p, token, pos, cache):
    """新しい1トークン（位置 pos）だけを計算する。cache = {"K": [...], "V": [...]} に1行ずつ追加。"""
    x0 = p["E"][token] + p["P"][pos]          # P に pos 行目が無ければここで IndexError
    u1 = ln(x0)
    q, k, v = u1 @ p["Wq"], u1 @ p["Wk"], u1 @ p["Wv"]
    cache["K"].append(k)
    cache["V"].append(v)
    K, V = np.array(cache["K"]), np.array(cache["V"])   # (pos+1, 4)：過去＋今の行だけ
    o = []
    for h in range(mg.H):
        sl = slice(h * DK, (h + 1) * DK)
        a = mg.softmax(q[sl] @ K[:, sl].T / np.sqrt(DK))   # マスク不要：未来はまだキャッシュに無い
        o.append(a @ V[:, sl])
    x1 = x0 + np.concatenate(o) @ p["Wo"]
    x2 = x1 + np.maximum(ln(x1) @ p["W1"], 0) @ p["W2"]
    return x2 @ p["Wout"]


def generate(p, label, max_new=4):
    print(f"\n[{label}]")
    seq, cache = [0], {"K": [], "V": []}
    for pos in range(max_new):
        try:
            z_cache = cached_step(p, seq[-1], pos, cache)
        except IndexError:
            print(f"  位置{pos}: P は {p['P'].shape[0]} 行しか無いので位置{pos}の埋め込みが存在せず、ここで停止"
                  f"（= このモデルの文脈長は {p['P'].shape[0]}）")
            break
        z_full = full_recompute(p, seq)
        nxt = int(z_cache.argmax())
        probs = mg.softmax(z_cache)
        print(f"  入力 {' '.join(mg.VOCAB[i] for i in seq):<10} → 次は「{mg.VOCAB[nxt]}」(p={probs[nxt]:.4f})  "
              f"キャッシュ版と全再計算の logits の最大差 {np.abs(z_cache - z_full).max():.1e}")
        seq.append(nxt)
    print(f"  生成結果: {' '.join(mg.VOCAB[i] for i in seq)}")


def cost(n):
    """n 語を生成するときに計算する Q/K/V の行数と、スコア q·k の個数（1ヘッドあたり）。"""
    full_rows = sum(t for t in range(1, n + 1))                 # 毎回 t 行を計算し直す
    full_scores = sum(t * (t + 1) // 2 for t in range(1, n + 1))  # マスクで下三角だけ
    return n, full_rows, n * (n + 1) // 2, full_scores


def main():
    p0, lr = mg.load_params()
    c0 = mg.forward(p0)
    g, _ = mg.backward(p0, c0)
    p1 = mg.sgd(p0, g, lr)
    print("学習時（3語まとめて入力）の予測: 更新前 "
          + " ".join(mg.VOCAB[i] for i in c0["probs"].argmax(-1))
          + " / 更新後 " + " ".join(mg.VOCAB[i] for i in mg.forward(p1)["probs"].argmax(-1)))
    generate(p0, "初期の重み（params.json）で <s> から生成")
    generate(p1, "手計算の1ステップ更新後の重み（7.1）で <s> から生成")

    print("\n[計算量] n 語生成するとき（1ヘッドあたり）")
    print("     n   Q/K/V の行数（キャッシュ / 全再計算）   スコアの個数（キャッシュ / 全再計算）")
    for n in (3, 100, 1000):
        rows_c, rows_f, sc_c, sc_f = cost(n)
        print(f"  {n:4d}   {rows_c:>10,} / {rows_f:<14,}            {sc_c:>10,} / {sc_f:,}")


if __name__ == "__main__":
    main()
