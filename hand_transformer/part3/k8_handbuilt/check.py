#!/usr/bin/env python3
"""K8 の採点スクリプト：手で決めた重みの Attention-only Transformer が、K7 と同じデータで何点取れるかを測る。

使い方（hand_transformer/part3/k8_handbuilt/ で）:
    python3 check.py my_weights.py                 # 自分の重み（build() を書いたファイル）
    python3 check.py solution.py                   # 参考解（2層の induction head）
    python3 check.py solution.py --sink            # 参考解の改良版（一致が無いときは <s> を見る）
    python3 check.py solution_1layer_fixed.py --fixed-n 8   # 1層・位置だけの解（N = 8 固定のデータ）
    python3 check.py solution_1layer_fixed.py --fixed-n 8 --eval-varied   # 同じ重みを N = 4..10 のデータで測る

重みのファイルには build(V, T, **opts) を書き、パラメータの dict を返す（形は ../k7_induction/attn_only.py と同じ）:
    "E": (V+2)×D, "P": T×D, "Wout": D×(V+2),
    "L{層}H{ヘッド}.Wq": D×dk, ".Wk": D×dk, ".Wv": D×dv, ".Wo": dv×D   （dk, dv はヘッドごとに自由）
トークン番号は 0..V−1 が内容、V が <s>、V+1 が <pad>。順伝播では S = Q·Kᵀ/√dk と √dk で割ることに注意。

合格の目安：2回目（コピーで予測できる部分）の正解率 acc_2nd ≥ 0.99。
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "k7_induction"))
import attn_only as ao  # noqa: E402
import induction as ind  # noqa: E402


def load_build(path):
    spec = importlib.util.spec_from_file_location("weights_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build


def check_shapes(p, V, T):
    ok = True
    D = p["E"].shape[1]
    want = {"E": (V + 2, D), "P": (T, D), "Wout": (D, V + 2)}
    for k, s in want.items():
        if p[k].shape != s:
            print(f"  形が違う: {k} は {p[k].shape}、期待は {s}")
            ok = False
    n_layers, n_heads = ao.structure(p)
    for l in range(n_layers):
        for h in range(n_heads[l]):
            Wq, Wk, Wv, Wo = (p[ao.hname(l, h, m)] for m in ao.HEAD_MATS)
            if not (Wq.shape[0] == Wk.shape[0] == Wv.shape[0] == D and Wo.shape[1] == D
                    and Wq.shape[1] == Wk.shape[1] and Wv.shape[1] == Wo.shape[0]):
                print(f"  形が合わない: 層{l} ヘッド{h} Wq {Wq.shape} Wk {Wk.shape} Wv {Wv.shape} Wo {Wo.shape}")
                ok = False
    return ok, D, n_layers, n_heads


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", help="build(V, T, **opts) を定義した .py ファイル")
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--nmin", type=int, default=4)
    ap.add_argument("--nmax", type=int, default=10)
    ap.add_argument("--fixed-n", type=int, default=None, help="N を固定したデータで測る（T = 1 + 2N）")
    ap.add_argument("--eval-varied", action="store_true",
                    help="--fixed-n で作った重みを、N = nmin..nmax が混ざったデータで測る（T は 1 + 2·max(N, nmax)）")
    ap.add_argument("--sink", action="store_true", help="build(..., sink=True) を呼ぶ（参考解の改良版）")
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    V = args.vocab
    opts = {"sink": True} if args.sink else {}
    if args.fixed_n:
        opts["n"] = args.fixed_n
    if args.fixed_n and not args.eval_varied:
        nmin = nmax = args.fixed_n
    else:
        nmin, nmax = args.nmin, max(args.nmax, args.fixed_n or 0)
    T = 1 + 2 * nmax
    build = load_build(args.weights)
    p = {k: np.asarray(v, dtype=float) for k, v in build(V, T, **opts).items()}
    ok, D, n_layers, n_heads = check_shapes(p, V, T)
    if not ok:
        raise SystemExit(1)
    n_par = sum(v.size for v in p.values())
    nz = sum(int((v != 0).sum()) for v in p.values())
    heads = ", ".join(f"層{l}: " + " ".join(f"H{h}(dk {p[ao.hname(l, h, 'Wq')].shape[1]}, dv {p[ao.hname(l, h, 'Wv')].shape[1]})"
                                            for h in range(n_heads[l])) for l in range(n_layers))
    print(f"重み: D = {D}、層 {n_layers}（{heads}）、パラメータ {n_par} 個（0 でないもの {nz} 個）")
    print(f"データ: V = {V}（+<s>, <pad>）、N = {nmin}..{nmax}、T = {T}、{args.batch} 系列（seed {args.seed}）")

    data = ind.make_batch(np.random.default_rng(args.seed), args.batch, V, nmin, nmax, None)
    with np.errstate(all="ignore"):
        r, c = ind.evaluate(p, data)
    if not np.isfinite(r["loss"]):
        print("  損失が有限でない（logits が大きすぎる／NaN）。重みの大きさを見直す")
    print(f"\n  loss_1st = {r['loss_1st']:.4f}（参考: ln V = {np.log(V):.4f}、ln(V+2) = {np.log(V + 2):.4f}）")
    print(f"  loss_2nd = {r['loss_2nd']:.4f}")
    print(f"  acc_2nd  = {r['acc_2nd']:.4f}")
    pred = c["logits"].argmax(-1)
    corr = pred == data["tgt"]
    m2 = data["m2nd"] > 0
    byN = " ".join(f"N={n}:{corr[m2 & (data['N'][:, None] == n)].mean():.3f}"
                   for n in range(nmin, nmax + 1) if (m2 & (data["N"][:, None] == n)).any())
    print(f"  N ごとの acc_2nd: {byN}")

    print("\n各ヘッドの注意（prev = i→i−1、ind = i→コピー元、bos = i→<s>）")
    for key in sorted(r["prev"]):
        print(f"  層{key[0]} ヘッド{key[1]}: prev {r['prev'][key]:.3f}  ind {r['ind'][key]:.3f}  bos {r['bos'][key]:.3f}")
    print("\nアブレーション（そのヘッドの出力を 0 にする）")
    for l in range(n_layers):
        for h in range(n_heads[l]):
            with np.errstate(all="ignore"):
                ra, _ = ind.evaluate(p, data, ablate={(l, h)})
            print(f"  層{l} ヘッド{h} を消す: loss_2nd {ra['loss_2nd']:.4f}  acc_2nd {ra['acc_2nd']:.4f}")

    passed = r["acc_2nd"] >= 0.99
    print(f"\n判定: acc_2nd = {r['acc_2nd']:.4f} → {'合格（≥ 0.99）' if passed else '未達（< 0.99）'}")
    if passed and r["loss_2nd"] > 0.1:
        print(f"  ただし loss_2nd = {r['loss_2nd']:.4f} > 0.1：argmax は当たっているが確率は低い（点差が小さく、自信が無い）")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
