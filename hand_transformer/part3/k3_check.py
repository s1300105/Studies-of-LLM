"""K3 照合スクリプト：自分で書いた行列版の実装を、Part 1・Part 2 の全チェックポイントで正解と突き合わせる。

    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py            # Part 1 と Part 2 の両方
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --part 1   # Part 1 だけ（まずここから）
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py -v         # 量ごとに誤差を全部表示
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --gradcheck  # 自分の forward だけで数値微分（正解を使わない）
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --autograd   # K1 のスカラー自動微分（K2 のグラフ）の勾配と比べる
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --autograd --value part3/k1_autograd/value.py   # 自分の Value で
    python3 part3/k3_check.py part3/solutions/my_gpt.py            # 参考解答（照合スクリプト自体の確認）

照合する相手は mini_gpt.py（Part 1）と part2/mini_gpt2.py（Part 2）。API は part3/k3_matrix/my_gpt.py の冒頭を参照。
必須は forward の "loss" と "logits"、backward の g（9個）だけで、そのほかの量は返せば照合し、無ければ「未提供」と表示する。

判定：|自分 − 正解| ≤ tol·(1 + |正解|)（既定 tol = 1e-8。コード同士なので手計算の許容差よりずっと厳しい）。
S はマスクされていないマスだけ、dA もマスクされていないマスだけを比べる（マスク部を −1e9 や「計算しない」で済ませてもよい）。
最初にずれたチェックポイントが分かったら、Part 1 なら python3 extras/bug_signatures.py の表と見比べる。
"""
import argparse
import copy
import importlib.util
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part2"))
import mini_gpt as m  # noqa: E402
import mini_gpt2 as m2  # noqa: E402

H = m.H


class Missing(Exception):
    """学習者が（任意の）量を返していない。"""


class ShapeError(Exception):
    pass


def load_module(path):
    spec = importlib.util.spec_from_file_location("learner_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- タスク

def parts():
    p1, lr1 = m.load_params()
    p2, lr2 = m2.load_params()
    return {
        1: dict(name="Part 1（<s> 猫 が → 猫 が 鳴く、3行）", p=p1, lr=lr1, task=m.TASK1, B=1, T=3, rows=m.POS, vocab=m.VOCAB),
        2: dict(name="Part 2（猫文・犬文、6行、ブロック対角マスク）", p=p2, lr=lr2, task=m2.TASK2, B=2, T=3, rows=m2.ROWS, vocab=m2.VOCAB),
    }


def learner_task(P):
    """学習者に渡す task（毎回新しいコピー）。バッチ形式で書きたい人向けに B×T の形も添える。"""
    t = P["task"]
    B, T = P["B"], P["T"]
    return {"inp": list(t["inp"]), "tgt": list(t["tgt"]), "posi": list(t["posi"]), "mask": np.array(t["mask"], dtype=bool),
            "batch": {"inp": np.array(t["inp"]).reshape(B, T), "tgt": np.array(t["tgt"]).reshape(B, T),
                      "mask": np.triu(np.ones((T, T), dtype=bool), 1)}}


# ---------------------------------------------------------------- 照合する量の一覧

def get(obj, path):
    try:
        for key in path:
            obj = obj[key]
    except (KeyError, IndexError, TypeError):
        raise Missing from None
    if obj is None:
        raise Missing
    return obj


ATTN = {"S": -np.inf, "A": 0.0, "dA": 0.0, "dS": 0.0}   # 注意の形の量と、ブロック外（文をまたぐ部分）の値


def fwd_items(cid_map, tag=""):
    """順伝播の量：(チェックポイント, 名前, c の中の場所, 注意の種類)。cid_map で 2.1→7.3 のように付け替える。"""
    k = lambda cid: cid_map(cid)
    items = [(k("2.1"), "X0", ("X0",), None)]
    items += [(k("2.2"), f"{s}（LN1）", ("ln1", key), None) for s, key in (("μ", "mu"), ("σ²", "var"), ("σ", "sigma"))]
    items += [(k("2.3"), "U1", ("U1",), None)] + [(k("3.1"), x, (x,), None) for x in "QKV"]
    items += [(k("3.2"), f"S{h}", ("S", h), "S") for h in range(H)] + [(k("3.3"), f"A{h}", ("A", h), "A") for h in range(H)]
    items += [(k("3.4"), f"O{h}", ("O", h), None) for h in range(H)] + [(k("3.4"), "Ocat", ("Ocat",), None)]
    items += [(k("3.5"), "Attn", ("Attn",), None), (k("4.1"), "X1", ("X1",), None)]
    items += [(k("4.2"), f"{s}（LN2）", ("ln2", key), None) for s, key in (("μ", "mu"), ("σ²", "var"), ("σ", "sigma"))]
    items += [(k("4.2"), "U2", ("U2",), None), (k("4.3"), "Z", ("Z",), None), (k("4.3"), "R", ("R",), None),
              (k("4.4"), "F", ("F",), None), (k("4.5"), "X2", ("X2",), None), (k("5.1"), "logits", ("logits",), None),
              (k("5.2"), "probs", ("probs",), None), (k("5.3"), "nll", ("nll",), None), (k("5.3"), "L", ("loss",), None)]
    return [(cid, name + tag, path, kind) for cid, name, path, kind in items]


BWD = [("6.1", "dlogits", "d", ("logits",), None), ("6.1", "dWout", "g", ("Wout",), None), ("6.1", "dX2", "d", ("X2",), None),
       ("6.2", "dF", "d", ("F",), None), ("6.2", "dW2", "g", ("W2",), None), ("6.2", "dR", "d", ("R",), None),
       ("6.2", "dZ", "d", ("Z",), None), ("6.2", "dW1", "g", ("W1",), None), ("6.2", "dU2", "d", ("U2",), None),
       ("6.3", "LN2経由", "d", ("X1_ln2",), None), ("6.3", "dX1", "d", ("X1",), None),
       ("6.4", "dAttn", "d", ("Attn",), None), ("6.4", "dWo", "g", ("Wo",), None), ("6.4", "dOcat", "d", ("Ocat",), None)]
BWD += [("6.5", f"{n}{h}", "d", (n[1:], h), n) for h in range(H) for n in ("dA", "dS")]
BWD += [("6.6", f"d{x}", "d", (x,), None) for x in "QKV"]
BWD += [("6.7", f"dW{x}", "g", (f"W{x}",), None) for x in "qkv"] + [("6.7", "dU1", "d", ("U1",), None)]
BWD += [("6.8", "LN1経由", "d", ("X0_ln1",), None), ("6.8", "dX0", "d", ("X0",), None),
        ("6.9", "dP", "g", ("P",), None), ("6.9", "dE", "g", ("E",), None)]

GROUPS = {"Wq, Wk（QK）": ["Wq", "Wk"], "Wv, Wo（OV）": ["Wv", "Wo"], "Attention 全部": ["Wq", "Wk", "Wv", "Wo"],
          "Wout だけ": ["Wout"], "W1, W2（FFN）": ["W1", "W2"], "E, P（埋め込み）": ["E", "P"], "全部": m.PARAMS}


# ---------------------------------------------------------------- 実行と比較

def run_model(f_forward, f_backward, f_sgd, P):
    """forward → backward → sgd → forward。学習者のコードが p や task を書き換えても影響しないようコピーを渡す。"""
    p, lr = P["p"], P["lr"]
    c = f_forward(copy.deepcopy(p), learner_task(P))
    out = f_backward(copy.deepcopy(p), c)
    g, d = out if isinstance(out, tuple) else (out, {})
    g = {k: np.asarray(g[k], dtype=float) for k in m.PARAMS}
    p1 = f_sgd(copy.deepcopy(p), g, lr) if f_sgd else {k: p[k] - lr * g[k] for k in m.PARAMS}
    p1 = {k: np.asarray(p1[k], dtype=float) for k in m.PARAMS}
    c1 = f_forward(copy.deepcopy(p1), learner_task(P))
    return {"c": c, "g": g, "d": d or {}, "p1": p1, "c1": c1}


def normalize(x, ref, kind, P):
    """(B, T, …) → (N, …)、(B, T, T) → ブロック対角の N×N、(N,) ↔ (N, 1) をそろえる。"""
    x = np.asarray(x, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if x.shape == ref.shape:
        return x
    B, T = P["B"], P["T"]
    if kind and x.shape == (B, T, T) and ref.shape == (B * T, B * T):
        out = np.full(ref.shape, ATTN[kind])
        for b in range(B):
            out[b * T:(b + 1) * T, b * T:(b + 1) * T] = x[b]
        return out
    if x.size == ref.size:
        return x.reshape(ref.shape)
    raise ShapeError(f"形が {x.shape}（正解は {ref.shape}）")


def compare(x, ref, kind, P, tol):
    """(最大誤差, ずれ最大のマスの位置, 合格か) を返す。"""
    x, ref = normalize(x, ref, kind, P), np.asarray(ref, dtype=float)
    use = np.ones(ref.shape, dtype=bool)
    if kind in ("S", "dA"):
        use = ~np.asarray(P["task"]["mask"], dtype=bool)        # マスクされていないマスだけ
    both_inf = np.isinf(x) & np.isinf(ref) & (np.sign(x) == np.sign(ref))
    with np.errstate(invalid="ignore"):                         # −∞ − (−∞) = nan は both_inf で 0 にする
        err = np.where(both_inf | ~use, 0.0, np.abs(np.nan_to_num(x - ref, nan=np.inf, posinf=np.inf, neginf=np.inf)))
    lim = tol * (1 + np.abs(np.where(np.isfinite(ref), ref, 0.0)))
    worst = np.unravel_index(np.argmax(err - lim), err.shape) if err.size else ()
    return float(err.max()) if err.size else 0.0, worst, bool((err <= lim).all()), x


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def build_checks(P, L, R, mod):
    """(cid, 名前, 正解の値, 学習者の値を返す関数, 注意の種類) のリスト。学習者の量は、無ければ Missing を投げる関数で渡す。"""
    part2 = P["task"] is m2.TASK2
    lr, task = P["lr"], P["task"]
    n, tgt = len(task["inp"]), list(task["tgt"])
    ck = []
    add = lambda cid, name, ref, fn, kind=None, rl=None: ck.append((cid, name, ref, fn, kind, rl))
    prow = lambda k: {"E": P["vocab"], "P": ["t0", "t1", "t2"], "W2": m.HID}.get(k, m.DIMS)   # パラメータの行ラベル
    lg = lambda C: {"logits": np.asarray(get(C, ("logits",)), dtype=float).reshape(n, -1)}   # (B, T, V) でも N×V に

    def opt(name):
        f = getattr(mod, name, None)
        if f is None:
            raise Missing
        return f

    for cid, name, path, kind in fwd_items(lambda c: c):
        add(cid, name, get(R["c"], path), lambda path=path: get(L["c"], path), kind, P["rows"])
    if part2:
        add("5.4", "C", m2.contrast(R["c"]), lambda: m2.contrast(lg(L["c"])))
        add("5.4", "ablate_t2", m2.ablate_t2(P["p"], R["c"]), lambda: opt("ablate_t2")(copy.deepcopy(P["p"]), L["c"]))
        add("5.5", "head_contributions", m2.decomposition_items(P["p"], R["c"])[0][1],
            lambda: opt("head_contributions")(copy.deepcopy(P["p"]), L["c"]))
    for cid, name, src, path, kind in BWD:
        add(cid, name, get(R[src], path), lambda src=src, path=path: get(L[src], path), kind,
            P["rows"] if src == "d" else prow(path[0]))
    for k in m.PARAMS:
        add("7.1", f"{k}'", R["p1"][k], lambda k=k: L["p1"][k], None, prow(k))
    gsq = lambda G: sum(float((G[k] ** 2).sum()) for k in m.PARAMS)
    add("7.2", "‖g‖², 予測ΔL", [gsq(R["g"]), -lr * gsq(R["g"])], lambda: [gsq(L["g"]), -lr * gsq(L["g"])])
    for cid, name, path, kind in fwd_items(lambda c: "7.3", "'"):
        add(cid, name, get(R["c1"], path), lambda path=path: get(L["c1"], path), kind, P["rows"])
    p_tgt = lambda C: softmax(lg(C)["logits"])[np.arange(n), tgt]
    add("7.4", "L, L', ΔL", [R["c"]["loss"], R["c1"]["loss"], R["c1"]["loss"] - R["c"]["loss"]],
        lambda: [float(L["c"]["loss"]), float(L["c1"]["loss"]), float(L["c1"]["loss"]) - float(L["c"]["loss"])])
    add("7.4", "p[正解]（前・後）", np.vstack([p_tgt(R["c"]), p_tgt(R["c1"])]).T,
        lambda: np.vstack([p_tgt(L["c"]), p_tgt(L["c1"])]).T)
    if part2:
        add("7.4", "C'", m2.contrast(R["c1"]), lambda: m2.contrast(lg(L["c1"])))
        add("7.4", "ablate_t2'", m2.ablate_t2(R["p1"], R["c1"]), lambda: opt("ablate_t2")(copy.deepcopy(L["p1"]), L["c1"]))
        add("7.4", "head_contributions'", m2.decomposition_items(R["p1"], R["c1"])[0][1],
            lambda: opt("head_contributions")(copy.deepcopy(L["p1"]), L["c1"]))

        def groups(fwd, G):
            rows = []
            for ks in GROUPS.values():
                q = {k: (P["p"][k] - lr * G[k] if k in ks else P["p"][k]) for k in m.PARAMS}
                cq = fwd(q)
                rows.append([m2.contrast(lg(cq)), float(cq["loss"])])
            return np.array(rows)
        ref_groups = groups(lambda q: m.forward(q, task), R["g"])
        add("7.5", "群ごとの更新（C, L）", ref_groups,
            lambda: groups(lambda q: mod.forward(copy.deepcopy(q), learner_task(P)), L["g"]))
        src = list(m2.SRC.values())
        for cid, PP, CR, CL, PL in (("8.1", P["p"], R["c"], L["c"], lambda: P["p"]), ("8.2", R["p1"], R["c1"], L["c1"], lambda: L["p1"])):
            tag = "" if cid == "8.1" else "'"
            for h in range(H):
                Mh = m2.qk_table(PP, h)
                add(cid, f"QK回路{h}{tag}", Mh, lambda h=h, PL=PL: opt("qk_circuit")(copy.deepcopy(PL()), h))
                add(cid, f"t2スコア{h}{tag}", CR["U1"][m2.A2] @ Mh @ CR["U1"][src].T / np.sqrt(m.DK),
                    lambda h=h, PL=PL, CL=CL: np.asarray(get(CL, ("U1",))).reshape(-1, m.D)[m2.A2]
                    @ opt("qk_circuit")(copy.deepcopy(PL()), h) @ np.asarray(get(CL, ("U1",))).reshape(-1, m.D)[src].T / np.sqrt(m.DK))
                add(cid, f"OV回路{h}{tag}", m2.ov_table(PP, CR, h)[0], lambda h=h, PL=PL: opt("ov_circuit")(copy.deepcopy(PL()), h))
                add(cid, f"OV行{h}{tag}", m2.ov_table(PP, CR, h)[1],
                    lambda h=h, PL=PL, CL=CL: np.asarray(get(CL, ("U1",))).reshape(-1, m.D)[src] @ opt("ov_circuit")(copy.deepcopy(PL()), h))
    return ck


def check_part(P, mod, tol, verbose):
    print(f"\n==== {P['name']} ====")
    R = run_model(m.forward, m.backward, m.sgd, P)
    try:
        L = run_model(mod.forward, mod.backward, getattr(mod, "sgd", None), P)
    except NotImplementedError as e:
        print(f"  未実装: {e}")
        return {"ok": 0, "ng": 1, "skip": 0, "first": "（実行できず）"}
    except Exception:  # noqa: BLE001  学習者のコードの例外はそのまま見せる
        print("  実行中に例外が起きました（Part 1 で動くなら、N・語彙数・マスクの扱いを固定していないか確認）:")
        print("  " + traceback.format_exc().replace("\n", "\n  "))
        return {"ok": 0, "ng": 1, "skip": 0, "first": "（実行できず）"}
    try:
        float(L["c"]["loss"]), np.asarray(L["c"]["logits"])
    except (KeyError, TypeError):
        print('  forward の戻り値に "loss" と "logits" が必要です')
        return {"ok": 0, "ng": 1, "skip": 0, "first": "（必須の量が無い）"}

    stats = {"ok": 0, "ng": 0, "skip": 0, "first": None}
    by_cid = {}
    for cid, name, ref, fn, kind, rows in build_checks(P, L, R, mod):
        try:
            err, where, ok, x = compare(fn(), ref, kind, P, tol)
            status = "OK" if ok else "NG"
            detail = f"最大誤差 {err:.1e}"
            if not ok:
                ref_a = np.asarray(ref, dtype=float)
                loc = ""
                if rows is not None and len(where) == 2 and ref_a.shape[0] == len(rows):
                    loc = f"行 {rows[where[0]]}、列 {where[1]}"
                elif len(where):
                    loc = "位置 " + ", ".join(map(str, where))
                detail += f"（{loc + '：' if loc else ''}自分 {x[where]:.6g}、正解 {ref_a[where]:.6g}）"
        except Missing:
            status, detail = "--", "未提供"
        except ShapeError as e:
            status, detail = "NG", str(e)
        except Exception as e:  # noqa: BLE001
            status, detail = "NG", f"{type(e).__name__}: {e}"
        stats[{"OK": "ok", "NG": "ng", "--": "skip"}[status]] += 1
        if status == "NG" and stats["first"] is None:
            stats["first"] = f"{cid} {name}"
        by_cid.setdefault(cid, []).append((status, name, detail))

    for cid, entries in by_cid.items():
        if verbose:
            for status, name, detail in entries:
                print(f"  [{status}] {cid:<4} {name:<36} {detail}")
            continue
        sts = {s for s, _, _ in entries}
        head = "NG" if "NG" in sts else ("OK" if "OK" in sts else "--")
        names = " ".join(n for s, n, _ in entries if s == "OK")
        line = f"  [{head}] {cid:<4} {names}"
        missing = [n for s, n, _ in entries if s == "--"]
        if missing:
            line += f"  （未提供: {' '.join(missing)}）"
        print(line)
        for status, name, detail in entries:
            if status == "NG":
                print(f"         ✗ {name}: {detail}")
    print(f"  合格 {stats['ok']}、不一致 {stats['ng']}、未提供 {stats['skip']}"
          + (f"。最初の不一致: {stats['first']}" if stats["first"] else ""))
    return stats


def gradcheck(P, mod, eps=1e-6):
    """学習者の forward だけを使った中心差分と、学習者の backward の g を比べる（mini_gpt は使わない）。"""
    p = copy.deepcopy(P["p"])
    out = mod.backward(copy.deepcopy(p), mod.forward(copy.deepcopy(p), learner_task(P)))
    g = out[0] if isinstance(out, tuple) else out
    print(f"\n==== 数値微分チェック：{P['name']}（自分の forward の中心差分 vs 自分の backward、h = {eps:g}）====")
    worst = 0.0
    for k in m.PARAMS:
        e = 0.0
        for idx in np.ndindex(p[k].shape):
            q = copy.deepcopy(p)
            q[k][idx] += eps
            lp = float(mod.forward(copy.deepcopy(q), learner_task(P))["loss"])
            q[k][idx] -= 2 * eps
            lm = float(mod.forward(copy.deepcopy(q), learner_task(P))["loss"])
            e = max(e, abs((lp - lm) / (2 * eps) - float(np.asarray(g[k])[idx])))
        worst = max(worst, e)
        print(f"  {k:5s} 最大誤差 {e:.1e}")
    print(f"  全体の最大誤差 {worst:.1e}（正しければ 1e-9 程度以下。mini_gpt.py --gradcheck は 1e-10 程度）")


def autograd_check(P, mod, value_path):
    """K1 の Value で組んだグラフ（k2_verify_part1.py の forward）の勾配と、学習者の backward の g を比べる。"""
    k2 = load_module(HERE / "k2_verify_part1.py")
    vm = k2.load_value_module(value_path)
    p = P["p"]
    Pv = {k: [[vm.Value(x) for x in row] for row in p[k]] for k in m.PARAMS}
    try:
        loss = k2.forward(Pv, P["task"])
        loss.backward()
    except NotImplementedError as e:
        print(f"\n  {value_path} に未実装の部分があります: {e}")
        return
    out = mod.backward(copy.deepcopy(p), mod.forward(copy.deepcopy(p), learner_task(P)))
    g = out[0] if isinstance(out, tuple) else out
    print(f"\n==== スカラー自動微分との照合：{P['name']}（Value: {value_path}）====")
    worst = 0.0
    for k in m.PARAMS:
        e = float(np.abs(np.array([[v.grad for v in row] for row in Pv[k]]) - np.asarray(g[k], dtype=float)).max())
        worst = max(worst, e)
        print(f"  {k:5s} 最大差 {e:.1e}")
    print(f"  全体の最大差 {worst:.1e}（正しければ 1e-15 以下。数値微分と違って打ち切り誤差が無いので、ずっと小さくなる）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("module", help="自分の実装（.py）のパス")
    ap.add_argument("--part", choices=("1", "2", "all"), default="all")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("-v", "--verbose", action="store_true", help="量ごとに誤差を表示")
    ap.add_argument("--gradcheck", action="store_true", help="自分の forward だけで数値微分して backward と比べる")
    ap.add_argument("--autograd", action="store_true", help="K1 のスカラー自動微分の勾配と backward を比べる")
    ap.add_argument("--value", default=str(HERE / "solutions" / "value.py"), help="--autograd で使う value.py")
    a = ap.parse_args()
    mod = load_module(Path(a.module))
    print(f"照合する実装: {a.module}（判定 |自分 − 正解| ≤ {a.tol:g}·(1 + |正解|)）")
    P = parts()
    which = [1, 2] if a.part == "all" else [int(a.part)]
    if a.gradcheck or a.autograd:
        for k in which:
            if a.gradcheck:
                gradcheck(P[k], mod)
            if a.autograd:
                autograd_check(P[k], mod, Path(a.value))
        return
    results = {k: check_part(P[k], mod, a.tol, a.verbose) for k in which}
    print("\n==== まとめ ====")
    for k, s in results.items():
        verdict = "全チェックポイント一致" if s["ng"] == 0 else f"不一致あり（最初: {s['first']}）"
        print(f"  Part {k}: {verdict}。合格 {s['ok']}、未提供 {s['skip']}")
    if any(s["ng"] for s in results.values()):
        print("  ヒント: 最初にずれたチェックポイントの1つ前までは正しい。Part 1 のずれ方は python3 extras/bug_signatures.py の表と見比べる。")
        print("         --gradcheck で自分の forward と backward が整合しているか（＝ずれが順伝播側か逆伝播側か）も分かる。")
    sys.exit(1 if any(s["ng"] for s in results.values()) else 0)


if __name__ == "__main__":
    main()
