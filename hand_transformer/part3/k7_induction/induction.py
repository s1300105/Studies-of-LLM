#!/usr/bin/env python3
"""K7：2層 Attention-only Transformer に「ランダム列を2回くり返す」系列を学習させ、induction head が生まれる瞬間を見る。

データ（1系列）:
    <s>  a b c … (N 個)  a b c … (同じ N 個)  <pad> …
    - a b c … は内容トークン 0..V−1 から「重複なし」でランダムに選ぶ
    - N は系列ごとに nmin..nmax からランダム（N を固定すると、理屈の上では1層でも「N−1 個前を見る」位置だけのヘッドで
      解ける。../k8_handbuilt/solution_1layer_fixed.py で確かめる。--fixed-n で学習させることもできる）
    - 1回目の N 個は予測できない（「まだ出ていないトークンのどれか」までしか絞れない）。2回目は「前に同じトークンが出た位置の、次のトークン」を
      コピーすれば当たる（2回目の先頭トークン自体は、1回目の終わりを知らないと予測できないので評価に入れない）
    - 位置 i の正解は i+1 のトークン。<pad> が正解になる位置は損失に入れない（<pad> は因果マスクのおかげで前の位置に影響しない）

表示する量:
    loss_1st / loss_2nd   1回目 / 2回目（予測できる部分）の損失。2回目の損失が「急に」下がるのが phase change
    acc_2nd               2回目の正解率（argmax）
    prev                  層0 の各ヘッドが「1つ前の位置」に払う注意の平均（previous-token head なら 1 に近い）
    ind                   層1 の各ヘッドが「前に同じトークンが出た位置の次」に払う注意の平均（induction head なら 1 に近い）

使い方（hand_transformer/part3/k7_induction/ で）:
    python3 attn_only.py --gradcheck        # 先に逆伝播を検算
    python3 induction.py                    # 2層を学習 → 注意パターン → アブレーション → 重みから回路を読む
    python3 induction.py --layers 1         # 1層では解けないことを確かめる（--steps 10000 で長く学習させても）
    python3 induction.py --layers 1 --fixed-n 8   # N を固定した1層（位置の近道を学習で見つけられるか）
    python3 induction.py --seeds 0,1,2,3,4,5,6,7  # seed を変えて、phase change のステップと回路の型を比べる
"""
import argparse
import time

import numpy as np

import attn_only as ao


# ---------------------------------------------------------------- データ

def make_batch(rng, B, V, nmin, nmax, fixed_n=None):
    """B 本の系列を作る。戻り値は dict（ids, tgt, N, 損失マスク）。"""
    T = 1 + 2 * nmax
    BOS, PAD = V, V + 1
    N = np.full(B, fixed_n) if fixed_n else rng.integers(nmin, nmax + 1, B)
    perm = np.argsort(rng.random((B, V)), axis=1)              # 行ごとに 0..V−1 の並べ替え（重複なし）
    pos = np.arange(T)[None, :]
    k1 = np.broadcast_to(pos - 1, (B, T))                      # 1回目の何番目か
    k2 = pos - 1 - N[:, None]                                  # 2回目の何番目か
    in1 = (k1 >= 0) & (k1 < N[:, None])
    in2 = (k2 >= 0) & (k2 < N[:, None])
    ids = np.full((B, T), PAD)
    ids[:, 0] = BOS
    rows = np.arange(B)[:, None].repeat(T, 1)
    ids[in1] = perm[rows[in1], k1[in1]]
    ids[in2] = perm[rows[in2], k2[in2]]
    tgt = np.full((B, T), PAD)
    tgt[:, :-1] = ids[:, 1:]
    valid = tgt != PAD
    m1st = valid & (pos <= N[:, None])                         # 正解が 1回目の N 個 + 2回目の先頭（予測不能）
    m2nd = valid & (pos >= N[:, None] + 1)                     # 正解が 2回目の 2個目以降（コピーで予測可能）
    return {"ids": ids, "tgt": tgt, "N": N, "valid": valid.astype(float),
            "m1st": m1st.astype(float), "m2nd": m2nd.astype(float), "V": V}


# ---------------------------------------------------------------- 評価

def evaluate(p, data, ablate=()):
    c = ao.forward(p, data["ids"], data["tgt"], data["valid"], ablate=ablate, keep_cache=False)
    nll = c["nll"]
    pred = c["logits"].argmax(-1)
    m1, m2 = data["m1st"], data["m2nd"]
    r = {"loss": c["loss"],
         "loss_1st": float((nll * m1).sum() / m1.sum()),
         "loss_2nd": float((nll * m2).sum() / m2.sum()),
         "acc_2nd": float(((pred == data["tgt"]) * m2).sum() / m2.sum()),
         "acc_1st": float(((pred == data["tgt"]) * m1).sum() / m1.sum())}
    r["prev"], r["ind"], r["ind1"], r["dup"], r["bos"] = attention_scores(c["A"], data)
    return r, c


def attention_scores(A, data):
    """各ヘッドの注意が、特定の相対位置にどれだけ乗っているか（平均）。"""
    ids, N = data["ids"], data["N"]
    B, T = ids.shape
    b = np.arange(B)[:, None]
    i = np.arange(T)[None, :].repeat(B, 0)
    valid_q = (data["valid"] > 0) & (i >= 1)                   # クエリ位置（<pad> 以外、先頭以外）
    m2 = data["m2nd"] > 0
    prev, ind, ind1, dup, bos = {}, {}, {}, {}, {}
    for key, a in A.items():
        prev[key] = float(a[b, i, np.maximum(i - 1, 0)][valid_q].mean())
        j_ind = np.clip(i - N[:, None] + 1, 0, T - 1)          # 前に同じトークンが出た位置の「次」（コピー元）
        ind[key] = float(a[b, i, j_ind][m2].mean())
        j_ind1 = np.clip(i - N[:, None] + 2, 0, T - 1)         # コピー元のさらに次（「1つずれた」回路が見る位置）
        ind1[key] = float(a[b, i, j_ind1][m2].mean())
        j_dup = np.clip(i - N[:, None], 0, T - 1)              # 前に同じトークンが出た位置そのもの
        dup[key] = float(a[b, i, j_dup][m2].mean())
        bos[key] = float(a[:, :, 0][valid_q].mean())
    return prev, ind, ind1, dup, bos


# ---------------------------------------------------------------- 学習

def train(args, seed, log=True):
    rng = np.random.default_rng(seed)
    V, T = args.vocab, 1 + 2 * args.nmax
    p = ao.init_params(V + 2, T, args.d_model, args.layers, args.heads, args.d_head, rng, std=args.init_std)
    opt = ao.Adam(p, lr=args.lr)
    eval_data = make_batch(np.random.default_rng(10_000 + seed), args.eval_batch, V, args.nmin, args.nmax, args.fixed_n)
    hist = []
    t0 = time.time()
    if log:
        print(f"{'step':>6} {'loss':>7} {'loss_1st':>8} {'loss_2nd':>8} {'acc_2nd':>7}  "
              f"{'prev(層0)':<11} {'ind(最後の層)':<11}  loss_2nd のグラフ（1目盛り = 0.1）")
    for step in range(args.steps + 1):
        if step % args.log_every == 0 or step == args.steps:
            r, _ = evaluate(p, eval_data)
            hist.append((step, r))
            if log:
                L = args.layers - 1
                pv = " ".join(f"{r['prev'][(0, h)]:.2f}" for h in range(args.heads))
                iv = " ".join(f"{r['ind'][(L, h)]:.2f}" for h in range(args.heads))
                bar = "#" * int(round(min(r["loss_2nd"], 4.0) * 10))
                print(f"{step:6d} {r['loss']:7.4f} {r['loss_1st']:8.4f} {r['loss_2nd']:8.4f} {r['acc_2nd']:7.3f}  "
                      f"{pv:<11} {iv:<11}  {bar}")
        if step == args.steps:
            break
        batch = make_batch(rng, args.batch, V, args.nmin, args.nmax, args.fixed_n)
        c = ao.forward(p, batch["ids"], batch["tgt"], batch["valid"])
        g = ao.backward(p, c)
        opt.step(p, g)
    if log:
        print(f"（学習 {args.steps} ステップ、{time.time() - t0:.1f} 秒）")
    return p, hist, eval_data


def phase_change_step(hist, thr=1.0):
    """2回目の損失が初めて thr（既定 1.0）を下回ったステップ（下回らなければ None）。

    1.0 は「既に出たトークンのどれか」という段階の損失（≈ 1.85）より十分低く、コピーが効き始めないと届かない値。"""
    for step, r in hist:
        if r["loss_2nd"] < thr:
            return step
    return None


# ---------------------------------------------------------------- 解析の表示

def show_patterns(p, data, args, which=0):
    """1本の系列について、各ヘッドの各クエリ位置が一番見ているキー位置を表示する。"""
    one = {k: (v[which:which + 1] if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    _, c = evaluate(p, one)
    N = int(one["N"][0])
    n = 2 * N + 1
    names = ["<s>" if t == args.vocab else str(t) for t in one["ids"][0, :n]]
    print(f"\n系列の例（N = {N}）: " + " ".join(names))
    print("各ヘッドで、クエリ位置 i が一番大きな重みを置くキー位置 j（括弧内はその重み）")
    print("  位置 i    : " + " ".join(f"{i:>9d}" for i in range(n)))
    print("  トークン  : " + " ".join(f"{s:>9s}" for s in names))
    for (l, h), A in sorted(c["A"].items()):
        a = A[0]
        cells = []
        for i in range(n):
            j = int(a[i, :i + 1].argmax())
            cells.append(f"{j:>3d}({a[i, j]:.2f})")
        print(f"  層{l} ヘッド{h}: " + " ".join(cells))
    pred = c["logits"][0].argmax(-1)
    print("  予測      : " + " ".join(f"{('<s>' if t == args.vocab else '<pad>' if t == args.vocab + 1 else str(t)):>9s}"
                                    for t in pred[:n]))


def show_scores(r, args):
    print("\n各ヘッドの注意の内訳（評価用の系列全体の平均）")
    print("  prev = i→i−1、ind = i→コピー元（前に同じトークンが出た位置の次）、ind+1 = i→コピー元の次、"
          "dup = i→前に同じトークンが出た位置、bos = i→<s>")
    print("  ヘッド           prev      ind    ind+1      dup      bos")
    for key in sorted(r["prev"]):
        l, h = key
        print(f"  層{l} ヘッド{h}  {r['prev'][key]:8.3f} {r['ind'][key]:8.3f} {r['ind1'][key]:8.3f} "
              f"{r['dup'][key]:8.3f} {r['bos'][key]:8.3f}")


def show_prev_by_position(p, data, args, r):
    """層0 で prev が最大のヘッドについて、クエリ位置 i ごとの A[i, i−1] の平均。"""
    hp = max(range(args.heads), key=lambda h: r["prev"][(0, h)])
    _, c = evaluate(p, data)
    A = c["A"][(0, hp)]
    valid = data["valid"] > 0
    T = A.shape[1]
    cells = [f"{i}:{A[valid[:, i], i, i - 1].mean():.2f}" for i in range(1, T) if valid[:, i].any()]
    print(f"\n層0 ヘッド{hp} の A[i, i−1]（クエリ位置 i ごとの平均）")
    print("  " + " ".join(cells))
    print(f"  （induction head が「1つ前のトークン」を読むキー位置は 2..N、つまり最大 {args.nmax}）")


def show_acc_by_k(p, data, args):
    """2回目の何番目のクエリか（k）ごとの正解率。k 番目のクエリ S[k] の正解は S[k+1]。"""
    _, c = evaluate(p, data)
    corr = c["logits"].argmax(-1) == data["tgt"]
    B, T = data["ids"].shape
    k = np.arange(T)[None, :] - data["N"][:, None] - 1
    m2 = data["m2nd"] > 0
    cells = []
    for kk in range(args.nmax - 1):
        m = m2 & (k == kk)
        if m.any():
            cells.append(f"k={kk}:{corr[m].mean():.3f}")
    print("\n2回目の k 番目のクエリ（S[k] を見て S[k+1] を当てる）ごとの正解率")
    print("  " + " ".join(cells))
    print("  （k が大きいほど「まだ2回目に出ていないトークン」は少ない：残りは N−1−k 個）")


def show_ablation(p, data, args, base):
    print("\nアブレーション（そのヘッドの出力を 0 にして、評価用の系列で測り直す）")
    print("  消したヘッド      loss_2nd   acc_2nd   loss_1st   最後の層の ind（各ヘッド）")
    L = args.layers - 1

    def row(label, r):
        iv = " ".join(f"{r['ind'][(L, h)]:.3f}" for h in range(args.heads))
        print(f"  {label:<14s} {r['loss_2nd']:9.4f} {r['acc_2nd']:9.3f} {r['loss_1st']:10.4f}   {iv}")

    row("なし", base)
    for l in range(args.layers):
        for h in range(args.heads):
            r, _ = evaluate(p, data, ablate={(l, h)})
            row(f"層{l} ヘッド{h}", r)
    if args.layers >= 2:
        r, _ = evaluate(p, data, ablate={(0, h) for h in range(args.heads)})
        row("層0 を全部", r)
        r, _ = evaluate(p, data, ablate={(1, h) for h in range(args.heads)})
        row("層1 を全部", r)


def show_ov_and_copies(p, data, args):
    """各ヘッドの OV 回路 E·Wv·Wo·Wout の対角（+ ならコピー、− なら抑制）と、2回目のクエリが注意を払う場所の内訳。"""
    V = args.vocab
    _, c = evaluate(p, data)
    N = data["N"]
    T = data["ids"].shape[1]
    m2 = data["m2nd"] > 0
    j = np.arange(T)[None, None, :]
    first = (j >= 1) & (j <= N[:, None, None])
    second = j >= N[:, None, None] + 1
    print("\n各ヘッドの OV 回路 E·Wv·Wo·Wout の対角（+ = 見たトークンを上げる、− = 下げる）と、2回目のクエリの注意の内訳")
    print("  ヘッド         対角の平均   対角以外   <s> へ   1回目へ   2回目へ")
    for (l, h), A in sorted(c["A"].items()):
        C = p["E"][:V] @ p[ao.hname(l, h, "Wv")] @ p[ao.hname(l, h, "Wo")] @ p["Wout"][:, :V]
        print(f"  層{l} ヘッド{h}  {np.diag(C).mean():10.2f} {C[~np.eye(V, dtype=bool)].mean():10.2f} "
              f"{A[:, :, 0][m2].mean():8.2f} {(A * first).sum(-1)[m2].mean():9.2f} {(A * second).sum(-1)[m2].mean():9.2f}")
    print("  （OV はトークン埋め込み E を読んだときの値。層0 のヘッドは層1 を通らずに直接 logits へ届く分、"
          "層1 のヘッドは層0 が書き込んだ分を含まない）")


def show_circuits(p, args, r):
    """重みだけから回路を読む（Elhage et al. 2021 の QK 回路・OV 回路）。データは使わない。"""
    V, L = args.vocab, args.layers - 1
    hp = max(range(args.heads), key=lambda h: r["prev"][(0, h)])
    hi = max(range(args.heads), key=lambda h: r["ind"][(L, h)])
    W = lambda l, h, m: p[ao.hname(l, h, m)]
    T = 1 + 2 * args.nmax
    print(f"\n重みから回路を読む（P = 層0 ヘッド{hp}、I = 層{L} ヘッド{hi}）")
    # (1) P の位置どうしの QK：(P Wq)(P Wk)ᵀ/√dk。各行（クエリ位置 i）で一番大きいキー位置 j ≤ i
    Pq, Pk = p["P"][:T] @ W(0, hp, "Wq"), p["P"][:T] @ W(0, hp, "Wk")
    Sp = Pq @ Pk.T / np.sqrt(Pq.shape[1])
    am = [int(np.argmax(Sp[i, :i + 1])) for i in range(1, T)]
    hit = sum(a == i - 1 for a, i in zip(am, range(1, T)))
    print(f"  (1) P の位置 QK 行列 P·Wq·Wkᵀ·Pᵀ/√dk：クエリ位置 i = 1..{T - 1} のうち、最大のキーが i−1 なのは {hit}/{T - 1} 行")
    print("      各行の最大のキー位置: " + " ".join(f"{i}→{a}" for i, a in zip(range(1, T), am)))
    # (2) K-composition：クエリのトークン a と、キー位置の「1つ前のトークン」b（P が運んできたもの）の相性
    Eq = p["E"][:V] @ W(L, hi, "Wq")
    Ek = (p["E"][:V] @ W(0, hp, "Wv") @ W(0, hp, "Wo")) @ W(L, hi, "Wk")
    M = Eq @ Ek.T / np.sqrt(Eq.shape[1])
    diag = int((M.argmax(1) == np.arange(V)).sum())
    print(f"  (2) K-composition 行列 (E·Wq_I)(E·Wv_P·Wo_P·Wk_I)ᵀ/√dk（{V}×{V}、行 = 今のトークン a、列 = キー位置の1つ前のトークン b）：")
    print(f"      各行の最大が対角（a = b）なのは {diag}/{V} 行。対角の平均 {np.diag(M).mean():.2f}、対角以外の平均 {M[~np.eye(V, dtype=bool)].mean():.2f}")
    # (3) OV 回路：I が見たトークン c を出力のどの logit に変えるか
    C = p["E"][:V] @ W(L, hi, "Wv") @ W(L, hi, "Wo") @ p["Wout"][:, :V]
    diag = int((C.argmax(1) == np.arange(V)).sum())
    print(f"  (3) I の OV 回路 E·Wv_I·Wo_I·Wout（{V}×{V}、行 = 見たトークン c、列 = 上がる logit）：")
    print(f"      各行の最大が対角（c をそのまま出す = コピー）なのは {diag}/{V} 行。対角の平均 {np.diag(C).mean():.2f}、対角以外の平均 {C[~np.eye(V, dtype=bool)].mean():.2f}")


def baselines(args):
    V = args.vocab
    print("\n参考になる損失の値")
    print(f"  ln V = ln {V} = {np.log(V):.4f}（一様に当て推量）")
    Ns = np.arange(args.nmin, args.nmax + 1)
    # 1回目の位置 k（0始まり）の正解は、まだ出ていない V−k 個のどれか。重複なしを知っていれば ln(V−k)。
    # 2回目の先頭の正解は「既に出た N 個のどれか」なので ln N。
    vals = []
    for N in Ns:
        vals += [np.log(V - k) for k in range(N)] + [np.log(N)]
    print(f"  1回目で「まだ出ていないトークンのどれか」を一様に当てたときの loss_1st = {np.mean(vals):.4f}（目安）")
    vals2 = []
    for N in Ns:
        vals2 += [np.log(N - 1)] * (N - 1)
    print(f"  2回目で「既に出た N 個のうち、今のトークン以外のどれか」を一様に当てたときの loss_2nd = {np.mean(vals2):.4f}"
          "（目安。下限ではない）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--d-head", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=16, help="内容トークンの種類 V（このほかに <s> と <pad>）")
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--fixed-n", type=int, default=None, help="N を固定する（位置だけの近道が使える設定）")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--eval-batch", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7, help="既定の 7 は教科書型の回路になる seed（--seeds の表を参照）")
    ap.add_argument("--seeds", type=str, default=None, help="カンマ区切りの seed で学習だけ回し、phase change のステップを比べる")
    ap.add_argument("--save", type=str, default=None, help="学習後の重みを .npz に保存する（K8 で読み比べる用）")
    args = ap.parse_args()
    if args.fixed_n:
        args.nmin = args.nmax = args.fixed_n

    print(f"設定: 層 {args.layers}、ヘッド {args.heads}/層、d_model {args.d_model}、d_head {args.d_head}、"
          f"V {args.vocab}（+<s>, <pad>）、N {args.nmin}..{args.nmax}、T {1 + 2 * args.nmax}、"
          f"Adam lr {args.lr}、バッチ {args.batch}、初期値 N(0, {args.init_std}²)")

    if args.seeds:
        L = args.layers - 1
        print("\n回路の判定：層0 で prev が最大のヘッドを P、最後の層で ind（または ind+1）が最大のヘッドを I とする")
        print("  型：教科書型 = I の ind > 0.8、1つずれ型 = I の ind+1 > 0.5、どちらでもなければ その他")
        print(f"{'seed':>4} {'loss_2nd':>9} {'acc_2nd':>8} {'phase change':>12}  {'型':<8} "
              f"{'P':>5} {'prev(P)':>8} {'I':>5} {'ind(I)':>7} {'ind+1(I)':>9} {'P を消す':>9} {'I を消す':>9} {'P消→I の注意':>13}")
        for s in [int(x) for x in args.seeds.split(",")]:
            p, hist, data = train(args, s, log=False)
            a, z = hist[0][1], hist[-1][1]
            st = phase_change_step(hist)
            hp = max(range(args.heads), key=lambda h: z["prev"][(0, h)])
            hi = max(range(args.heads), key=lambda h: max(z["ind"][(L, h)], z["ind1"][(L, h)]))
            kind = ("教科書型" if z["ind"][(L, hi)] > 0.8 else "1つずれ型" if z["ind1"][(L, hi)] > 0.5 else "その他")
            key = "ind" if kind != "1つずれ型" else "ind1"
            rp, _ = evaluate(p, data, ablate={(0, hp)})
            ri, _ = evaluate(p, data, ablate={(L, hi)})
            print(f"{s:4d} {z['loss_2nd']:9.4f} {z['acc_2nd']:8.3f} {str(st):>12}  {kind:<8} "
                  f"{'L0H%d' % hp:>5} {z['prev'][(0, hp)]:8.3f} {'L%dH%d' % (L, hi):>5} {z['ind'][(L, hi)]:7.3f} "
                  f"{z['ind1'][(L, hi)]:9.3f} {rp['loss_2nd']:9.4f} {ri['loss_2nd']:9.4f} {rp[key][(L, hi)]:13.3f}")
        return

    p, hist, data = train(args, args.seed)
    z = hist[-1][1]
    st = phase_change_step(hist)
    print(f"\nphase change：loss_2nd が初めて 1.0 を下回ったのは " + (f"ステップ {st}" if st is not None else "（下回らなかった）"))
    baselines(args)
    show_scores(z, args)
    show_acc_by_k(p, data, args)
    show_prev_by_position(p, data, args, z)
    show_patterns(p, data, args)
    show_ablation(p, data, args, z)
    show_ov_and_copies(p, data, args)
    if args.layers >= 2:
        show_circuits(p, args, z)
    if args.save:
        np.savez(args.save, **p)
        print(f"\n重みを {args.save} に保存しました")


if __name__ == "__main__":
    main()
