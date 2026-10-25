"""K2 Part 1 をスカラー自動微分で再検証する。

Part 1 の順伝播（2.1〜5.3）を、K1 の Value（1つ1つの数がノード）だけで組み立て、
loss.backward() で得た全180個の勾配を mini_gpt.py（＝手計算の正解）と照合する。
さらに SGD（lr = 1）で1ステップ更新し、もう一度 Value で順伝播して L' = 0.1056 を再現する。

    python3 part3/k2_verify_part1.py            # 参考解答 part3/solutions/value.py を使う
    python3 part3/k2_verify_part1.py --mine     # 自分の part3/k1_autograd/value.py を使う
    python3 part3/k2_verify_part1.py --part 2   # Part 2（2文・6行・ブロック対角マスク、勾配 188 個）で同じことをする

マイルストーン：勾配の最大誤差 ≤ 1e-10、L' が mini_gpt.py と一致（差 ≤ 1e-10）。

ここで使う演算は + * pow exp log だけ（ReLU も）。行列は「Value のリストのリスト」、行列積は積和の鎖。
mini_gpt.py の行列の式と、手計算の1マス1マスのちょうど中間の粒度です。
最後に「グラフのノード数」と「手計算で掛けた回数」（README §5）を並べて表示します。
"""
import argparse
import importlib.util
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part2"))
import mini_gpt as m  # noqa: E402


def load_value_module(path):
    spec = importlib.util.spec_from_file_location("value_for_k2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- Value だけで書いた Part 1 のモデル

def vsum(xs):
    """Value の和。sum() と同じだが、0 + x のノードを作らない。"""
    out = xs[0]
    for x in xs[1:]:
        out = out + x
    return out


def matmul(X, W):
    """X（n×k、Value の2重リスト）@ W（k×m）。成分ごとの積和を、そのままグラフにする。"""
    k, mm = len(W), len(W[0])
    return [[vsum([row[i] * W[i][j] for i in range(k)]) for j in range(mm)] for row in X]


def layernorm(X):
    """γ=1, β=0, ε=0、母分散（÷d）。行ごと。"""
    out = []
    for x in X:
        d = len(x)
        mu = vsum(x) * (1.0 / d)
        xc = [xi - mu for xi in x]
        var = vsum([v ** 2 for v in xc]) * (1.0 / d)
        inv_sigma = (var ** 0.5) ** -1
        out.append([v * inv_sigma for v in xc])
    return out


def attention_row(q, keys, values, allowed, scale):
    """1つのクエリ位置の1ヘッド分。allowed = 見てよいキー位置のリスト（マスクされた位置はノードを作らない）。"""
    s = [vsum([qi * ki for qi, ki in zip(q, keys[j])]) * scale for j in allowed]
    mx = max(v.data for v in s)                         # 最大値を引く（定数として。softmax はずらしても変わらない）
    e = [(v - mx).exp() for v in s]
    inv = vsum(e) ** -1
    a = [v * inv for v in e]
    return [vsum([a[n] * values[j][c] for n, j in enumerate(allowed)]) for c in range(len(values[0]))]


def forward(P, task):
    """P: {名前: Value の2重リスト}。task: mini_gpt.py と同じ辞書（inp, tgt, posi, mask）。損失（Value）を返す。"""
    inp, tgt, posi, mask = task["inp"], task["tgt"], task["posi"], task["mask"]
    n, H, DK = len(inp), m.H, m.DK
    X0 = [[e + p for e, p in zip(P["E"][i], P["P"][t])] for i, t in zip(inp, posi)]          # 2.1
    U1 = layernorm(X0)                                                                     # 2.2–2.3
    Q, K, V = matmul(U1, P["Wq"]), matmul(U1, P["Wk"]), matmul(U1, P["Wv"])                # 3.1
    Ocat = [[] for _ in range(n)]
    for h in range(H):                                                                      # 3.2–3.4
        c0, c1 = h * DK, (h + 1) * DK
        Kh, Vh = [k[c0:c1] for k in K], [v[c0:c1] for v in V]
        for i in range(n):
            allowed = [j for j in range(n) if not mask[i][j]]
            Ocat[i] += attention_row(Q[i][c0:c1], Kh, Vh, allowed, 1.0 / np.sqrt(DK))
    X1 = [[x + a for x, a in zip(xr, ar)] for xr, ar in zip(X0, matmul(Ocat, P["Wo"]))]    # 3.5–4.1
    R = [[z.relu() for z in row] for row in matmul(layernorm(X1), P["W1"])]               # 4.2–4.3
    X2 = [[x + f for x, f in zip(xr, fr)] for xr, fr in zip(X1, matmul(R, P["W2"]))]       # 4.4–4.5
    logits = matmul(X2, P["Wout"])                                                         # 5.1
    nll = []
    for z, y in zip(logits, tgt):                                                          # 5.2–5.3
        mx = max(v.data for v in z)
        lse = vsum([(v - mx).exp() for v in z]).log() + mx                                 # logsumexp − logit
        nll.append(lse - z[y])
    return vsum(nll) * (1.0 / n)


# ---------------------------------------------------------------- 数え上げ

def graph_stats(loss, topological_order):
    topo = topological_order(loss)
    ops = Counter(v._op or "葉（パラメータ・定数）" for v in topo)
    muls = [v for v in topo if v._op == "*"]
    zero = sum(1 for v in muls if any(c.data == 0 for c in v._prev))
    sign = sum(1 for v in muls if not any(c.data == 0 for c in v._prev) and any(abs(c.data) == 1 for c in v._prev))
    depth = {}
    for v in topo:                                       # 入力側から順に「最長の経路の長さ」を伝える
        depth[id(v)] = 1 + max((depth[id(c)] for c in v._prev), default=-1)
    return len(topo), ops, len(muls), zero, sign, depth[id(loss)]


def topo_fallback(root):
    """学習者の value.py に topological_order が無いときの代わり（solutions/value.py と同じもの）。"""
    topo, seen, stack = [], set(), [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            topo.append(v)
        elif id(v) not in seen:
            seen.add(id(v))
            stack.append((v, True))
            stack.extend((c, False) for c in v._prev if id(c) not in seen)
    return topo


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", action="store_true", help="自分の part3/k1_autograd/value.py を使う")
    ap.add_argument("--value", help="使う value.py のパス（--mine より優先）")
    ap.add_argument("--part", type=int, choices=(1, 2), default=1, help="1 = Part 1（既定）、2 = Part 2 の2文タスク")
    a = ap.parse_args()
    path = Path(a.value) if a.value else HERE / ("k1_autograd" if a.mine else "solutions") / "value.py"
    vm = load_value_module(path)
    Value = vm.Value
    topological_order = getattr(vm, "topological_order", topo_fallback)

    if a.part == 1:
        ref, task = m, m.TASK1
    else:
        import mini_gpt2 as ref
        task = ref.TASK2
    p, lr = ref.load_params()
    names = list(p)
    print(f"Value の実装: {path}")
    print(f"タスク: Part {a.part}（{len(task['inp'])} 行、パラメータ {sum(v.size for v in p.values())} 個、lr = {lr:g}）")

    # 1) 順伝播と逆伝播（Value）
    t0 = time.perf_counter()
    try:
        P = {k: [[Value(x) for x in row] for row in p[k]] for k in names}
        loss = forward(P, task)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
    except NotImplementedError as e:
        sys.exit(f"未実装の部分があります: {e}\n先に K1 を完成させてください（cd part3/k1_autograd && python3 test_value.py）")
    gv = {k: np.array([[v.grad for v in row] for row in P[k]]) for k in names}

    # 2) 正解（mini_gpt.py の行列版）と照合
    c = m.forward(p, task)
    g, _ = m.backward(p, c)
    print(f"\nL（Value）= {loss.data:.4f}   L（mini_gpt）= {c['loss']:.4f}   差 {abs(loss.data - c['loss']):.1e}")
    print("勾配の最大誤差（パラメータ群ごと）:")
    errs = {k: float(np.abs(gv[k] - g[k]).max()) for k in names}
    print("  " + "  ".join(f"{k} {e:.1e}" for k, e in errs.items()))
    worst = max(errs.values())
    n_grad = sum(v.size for v in gv.values())
    print(f"  全 {n_grad} 個の最大誤差 {worst:.1e}  → {'OK（≤ 1e-10）' if worst <= 1e-10 else 'NG（> 1e-10）'}")

    # 3) SGD で1ステップ → もう一度 Value で順伝播
    p1 = {k: p[k] - lr * gv[k] for k in names}
    loss1 = forward({k: [[Value(x) for x in row] for row in p1[k]] for k in names}, task)
    ref_L1 = m.forward(m.sgd(p, g, lr), task)["loss"]
    ok_L1 = abs(loss1.data - ref_L1) <= 1e-10
    print(f"\nSGD（lr = {lr:g}）で1ステップ後: L' = {loss1.data:.4f}（mini_gpt: {ref_L1:.4f}、差 {abs(loss1.data - ref_L1):.1e}）"
          f"  → {'OK' if ok_L1 else 'NG'}")

    # 4) 数え上げ
    n_nodes, ops, n_mul, n_zero, n_sign, depth = graph_stats(loss, topological_order)
    print(f"\n計算グラフ（1回目の順伝播）: ノード {n_nodes} 個、深さ {depth}（葉から損失までの最長の経路）")
    print("  内訳: " + "、".join(f"{op} {cnt}" for op, cnt in ops.most_common()))
    print(f"  積ノード {n_mul} 個のうち、0 との積 {n_zero} 個、±1 との積 {n_sign} 個、それ以外 {n_mul - n_zero - n_sign} 個")
    if a.part == 1:
        print("  手計算（README §5、cost.md）: 1回目の順伝播の積は工夫なしで 540 回、実際に掛けるのは 282 回。"
              "逆伝播は工夫なしで 1,116 回、実際は 625 回")
    print(f"  backward() は {n_nodes} 個のノードの _backward を1回ずつ呼んだ")
    print(f"  所要時間: グラフ作成（順伝播）{1000 * (t1 - t0):.0f} ミリ秒、backward {1000 * (t2 - t1):.0f} ミリ秒")
    ok = worst <= 1e-10 and ok_L1
    print("\nマイルストーン:", "達成" if ok else "未達")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
