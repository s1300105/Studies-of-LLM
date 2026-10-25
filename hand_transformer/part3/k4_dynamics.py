"""K4 2文タスクの学習ダイナミクス（参考実験。自分の K3 実装で同じ表を再現する）。

Part 2 では1ステップだけ手で回した。ここでは同じモデル・同じデータで数百ステップ回し、
「文脈の差 C がどう育つか」「どのヘッドが何を運ぶようになるか」「readout・OV・QK のどれが先に動くか」を見る。

    python3 part3/k4_dynamics.py                      # 全部（数十秒）
    python3 part3/k4_dynamics.py --section main       # part2/params.json から SGD と Adam で 300 ステップ
    python3 part3/k4_dynamics.py --section seeds      # 乱数初期化 seed 0–9（既定 σ = 0.02、3000 ステップ）
    python3 part3/k4_dynamics.py --section seeds --scale 0.5
    python3 part3/k4_dynamics.py --section robust     # 初期値を 1e-12 だけずらしたら結果は変わるか
    python3 part3/k4_dynamics.py --section trap       # 罠（Wout の2列が同じ／Wv = Wo = 0）
    python3 part3/k4_dynamics.py --impl part3/k3_matrix/my_gpt.py   # 自分の K3 実装で回す
    python3 part3/k4_dynamics.py --section main --csv k4_main.csv   # 毎ステップの記録を CSV に（グラフ用）
    python3 part3/k4_dynamics.py --section main --symmetrize        # 毎ステップ θ を (θ + g(θ))/2 に戻す（ミラー対称性を丸め誤差ごと保つ）

ログの記号（Part 2 と同じ）:
    L      6位置の平均損失。下限は 2·ln2/6 = 0.2310（t0 は 猫 か 犬 か原理的に分からない）。
           文脈を使わない最善は 4·ln2/6 = 0.4621（t0 と t2 がどちらも 50:50）
    C      (鳴く − 吠える)@猫文 t2 − (同)@犬文 t2。文脈の使い分け。初期値では厳密に 0
    a0, a1 ヘッド0・1 の 猫文 t2 → t1（猫）への注意の重み（犬文の t2 → t1 と同じになる。表の最後に差の最大値を出す）
    値項    5.5 の分解の「値項」（ヘッドごと）。パターン項はミラー対称性で 0（丸め誤差の大きさを出す）
    直接・総効果  5.5 の「直接効果」（Wo → Wout を通した (鳴く − 吠える) への寄与）と「パッチングの総効果」
    S(t2→t0/t1/t2)  猫文 t2 のクエリの各キーへのスコア（÷√2 済み、8.1 の「QK スコア表」と同じもの）
    移植    7.5 の「群ごとの更新」を学習の途中にも広げたもの：step t の群 X だけを使い、残りは初期値のまま。
           readout = Wout、OV = Wv, Wo、QK = Wq, Wk。
           「X だけ移植したモデルの C（または L）」が、群 X が学んだことの大きさの目安になる
    ミラー  Part 2 A5 のミラー g（d0 ↔ d1、猫 ↔ 犬・鳴く ↔ 吠える、値の空間はヘッドの中で入れ替え）。
           数学的には SGD は g で変わらない θ を g で変わらないまま保つが、丸め誤差でずれることがある。
           main では、t0 の p(猫) − p(犬)（対称なら厳密に 0）と、θ と g(θ) の差の最大を出す

Adam は β1 = 0.9, β2 = 0.999, ε = 1e-8、バイアス補正あり、weight decay なし。
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "part2"))
sys.path.insert(0, str(HERE))
import mini_gpt as m  # noqa: E402
import mini_gpt2 as m2  # noqa: E402
import k3_check as k3  # noqa: E402  --impl のとき、K3 と同じ task を作り、c を N 行の形にそろえるのに使う

TASK = m2.TASK2
FLOOR = 2 * np.log(2) / 6           # 0.2310
NOCTX = 4 * np.log(2) / 6           # 0.4621（文脈を使わない最善）
CAT_T2, DOG_T2, CAT_T1, DOG_T1 = m2.A2, m2.B2, m2.A1, m2.B1
GROUPS = {"readout": ["Wout"], "OV": ["Wv", "Wo"], "QK": ["Wq", "Wk"]}
LOG_STEPS = (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300)

FWD, BWD = m.forward, m.backward    # --impl で差し替える
SYMMETRIZE = False                  # --symmetrize のとき True
IMPL = False                        # --impl のとき True
PB = {"task": TASK, "B": 2, "T": 3}  # k3_check.learner_task・normalize に渡す、2文 × 3位置 の情報
NEED = ("loss", "logits", "X0", "X2", "V", "A", "S", "O")   # --impl の forward の c に要るもの


# ---------------------------------------------------------------- ミラー（Part 2 A5）

MIR_D = [1, 0, 2, 3]                # 残差ストリーム：d0 ↔ d1
MIR_VOC = [0, 2, 1, 3, 5, 4]        # 語彙：猫 ↔ 犬、鳴く ↔ 吠える（<s> と が はそのまま）
MIR_VAL = [1, 0, 3, 2]              # 値の空間（Wv の列・Wo の行）：各ヘッドの中で入れ替え。Q・K の空間と FFN の中間層はそのまま
NEKO, INU = m2.VOCAB.index("猫"), m2.VOCAB.index("犬")


def mirror(p):
    """ミラー g をパラメータに当てる（188 個の並べ替え）。part2/params.json は g で変わらない。"""
    return {"E": p["E"][MIR_VOC][:, MIR_D], "P": p["P"][:, MIR_D], "Wq": p["Wq"][MIR_D], "Wk": p["Wk"][MIR_D],
            "Wv": p["Wv"][MIR_D][:, MIR_VAL], "Wo": p["Wo"][MIR_VAL][:, MIR_D], "W1": p["W1"][MIR_D],
            "W2": p["W2"][:, MIR_D], "Wout": p["Wout"][MIR_D][:, MIR_VOC]}


def mirror_gap(p):
    """(Wout の 猫・犬 列での |θ − g(θ)| の最大, それ以外の全パラメータでの最大)。"""
    q = mirror(p)
    d = {k: np.abs(p[k] - q[k]) for k in p}
    nekoinu = float(d["Wout"][:, [NEKO, INU]].max())
    other = max(float(v.max()) for k, v in d.items() if k != "Wout")
    other = max(other, float(np.delete(d["Wout"], [NEKO, INU], axis=1).max()))
    return nekoinu, other


# ---------------------------------------------------------------- 最適化

def sgd(lr):
    def step(p, g, st):
        return {k: p[k] - lr * g[k] for k in p}, st
    return step


def adam(lr, b1=0.9, b2=0.999, eps=1e-8):
    def step(p, g, st):
        t = st.get("t", 0) + 1
        mm = {k: b1 * st.get("m", {}).get(k, 0.0) + (1 - b1) * g[k] for k in p}
        vv = {k: b2 * st.get("v", {}).get(k, 0.0) + (1 - b2) * g[k] ** 2 for k in p}
        q = {k: p[k] - lr * (mm[k] / (1 - b1 ** t)) / (np.sqrt(vv[k] / (1 - b2 ** t)) + eps) for k in p}
        return q, {"t": t, "m": mm, "v": vv}
    return step


def forward(p):
    """学習（勾配）用の c。--impl のときは K3 と同じ task（"batch" 付き）と、パラメータのコピーを渡す。"""
    if not IMPL:
        return FWD(p, TASK)
    return FWD({k: v.copy() for k, v in p.items()}, k3.learner_task(PB))


def grads(p, c):
    if not IMPL:
        out = BWD(p, c)
        return out[0] if isinstance(out, tuple) else out
    out = BWD({k: v.copy() for k, v in p.items()}, c)
    g = out[0] if isinstance(out, tuple) else out
    return {k: np.asarray(g[k], dtype=float) for k in p}


def rows(c):
    """読み取り用の c。--impl のときは (B, T, …) を N = 6 行に、(B, T, T) の注意をブロック対角の 6×6 にそろえる（K3 と同じ規則）。"""
    if not IMPL:
        return c
    n = len(TASK["inp"])
    try:
        out = {"loss": float(c["loss"]), "logits": k3.normalize(c["logits"], np.zeros((n, len(m2.VOCAB))), None, PB)}
        for key in ("X0", "X2", "V"):
            out[key] = k3.normalize(c[key], np.zeros((n, m.D)), None, PB)
        for key in ("A", "S"):
            out[key] = [k3.normalize(c[key][h], np.zeros((n, n)), key, PB) for h in range(m.H)]
        out["O"] = [k3.normalize(c["O"][h], np.zeros((n, m.DK)), None, PB) for h in range(m.H)]
    except (KeyError, IndexError, TypeError, k3.ShapeError) as e:
        sys.exit(f"--impl の forward の c には {', '.join(NEED)} が必要です（N 行の形でも (B, T, …) のバッチ形式でもよい。"
                 f"A, S, O はヘッドごとのリスト）。{type(e).__name__}: {e}")
    return out


def contrast(c):
    return float(m2.contrast({"logits": np.asarray(c["logits"]).reshape(len(TASK["inp"]), -1)}))


def symmetrize(p):
    q = mirror(p)
    return {k: (p[k] + q[k]) / 2 for k in p}


def train(p, step, steps):
    """steps 回更新し、各ステップのパラメータを返す（発散したらそこで止める）。--symmetrize なら毎回 (θ + g(θ))/2 に戻す。"""
    if SYMMETRIZE:
        p = symmetrize(p)
    traj, st = [p], {}
    with np.errstate(all="ignore"):
        for _ in range(steps):
            c = forward(p)
            if not np.isfinite(c["loss"]):
                break
            p, st = step(p, grads(p, c), st)
            if SYMMETRIZE:
                p = symmetrize(p)
            traj.append(p)
    return traj


# ---------------------------------------------------------------- 1つのモデルの読み取り

def read(p, p0):
    c = rows(forward(p))
    dec = m2.decomposition_items(p, c)[0][1]                    # ヘッドごとの [パターン項, 値項, 直接効果, 総効果]
    A = [np.asarray(c["A"][h]) for h in range(m.H)]
    z0 = np.asarray(c["logits"])[0]                            # 猫文 t0（犬文 t0 と同じ入力）。対称なら p(猫) = p(犬)
    pr0 = np.exp(z0 - z0.max()) / np.exp(z0 - z0.max()).sum()
    row = {"L": float(c["loss"]), "C": contrast(c), "dp_t0": float(pr0[NEKO] - pr0[INU]), "dz_t0": float(z0[NEKO] - z0[INU]),
           "a0": A[0][CAT_T2, CAT_T1], "a1": A[1][CAT_T2, CAT_T1],
           "asym": max(abs(A[h][CAT_T2, CAT_T1] - A[h][DOG_T2, DOG_T1]) for h in range(m.H)),
           "pat": float(np.abs(dec[:, 0]).max())}
    for h in range(m.H):
        row[f"pat{h}"], row[f"val{h}"], row[f"direct{h}"], row[f"total{h}"] = (float(x) for x in dec[h])
        S = np.asarray(c["S"][h])
        for j, key in enumerate(("t0", "t1", "t2")):
            row[f"S{h}_{key}"] = float(S[CAT_T2, j])
    for name, ks in GROUPS.items():                            # 移植：群 X だけ step t のもの、残りは初期値
        cq = rows(forward({k: (p[k] if k in ks else p0[k]) for k in p0}))
        row[f"C_{name}"], row[f"L_{name}"] = contrast(cq), float(cq["loss"])
    return row


def half_time(values):
    """値が「その実験での最大値」の半分に初めて届いたステップ。"""
    v = np.asarray(values)
    top = np.nanmax(v)
    return int(np.argmax(v >= 0.5 * top)) if top > 0 else None


# ---------------------------------------------------------------- セクション

def section_main(p0, steps, csv=None):
    print("=" * 100)
    print(f"[main] part2/params.json から {steps} ステップ。表は猫文 t2 の値（犬文はミラー規則で同じ）")
    configs = [("SGD lr = 1（Part 2 と同じ）", sgd(1.0)), ("SGD lr = 0.3", sgd(0.3)), ("Adam lr = 0.01", adam(0.01))]
    records = []
    for name, step in configs:
        traj = train(p0, step, steps)
        rows = [read(p, p0) for p in traj]
        records += [dict(optimizer=name.split("（")[0], step=t, **r) for t, r in enumerate(rows)]
        print(f"\n--- {name} ---")
        print("  (1) 何が起きているか")
        print(f"  {'step':>4} {'L':>7} {'C':>8} {'a0':>6} {'a1':>6} | {'値項0':>7} {'直接0':>7} {'総効果0':>7} | {'値項1':>7} {'直接1':>7} {'総効果1':>7}")
        for t in LOG_STEPS:
            if t >= len(rows):
                print(f"  {t:>4}  発散（nan）")
                break
            r = rows[t]
            print(f"  {t:>4} {r['L']:7.4f} {r['C']:8.4f} {r['a0']:6.4f} {r['a1']:6.4f} | {r['val0']:7.3f} {r['direct0']:7.3f} {r['total0']:7.3f}"
                  f" | {r['val1']:7.3f} {r['direct1']:7.3f} {r['total1']:7.3f}")
        print("  (2) どこを見るか（QK スコア）と、どの群が学んだか（移植）")
        print(f"  {'step':>4} | {'S0(t2→t0':>9} {'t1':>6} {'t2)':>6} | {'S1(t2→t0':>9} {'t1':>6} {'t2)':>6} | "
              f"{'移植 QK: C':>10} {'OV: C':>7} {'readout: C':>10} {'readout: L':>10}")
        for t in LOG_STEPS:
            if t >= len(rows):
                break
            r = rows[t]
            print(f"  {t:>4} | {r['S0_t0']:9.4f} {r['S0_t1']:6.3f} {r['S0_t2']:6.3f} | {r['S1_t0']:9.4f} {r['S1_t1']:6.3f} {r['S1_t2']:6.3f} | "
                  f"{r['C_QK']:10.4f} {r['C_OV']:7.4f} {r['C_readout']:10.4f} {r['L_readout']:10.4f}")
        L = np.array([r["L"] for r in rows])
        late = L[20:] if len(L) > 20 else L
        t_spike = 20 + int(np.argmax(late)) if len(L) > 20 else 0
        print(f"  step 20 以降の L の最大値 {late.max():.4f}（step {t_spike}）。パターン項の最大 {max(r['pat'] for r in rows):.1e}、"
              f"猫文と犬文の注意の差の最大 {max(r['asym'] for r in rows):.1e}")
        drop_readout = [rows[0]["L_readout"] - r["L_readout"] for r in rows]
        th = {"readout（移植で L が下がる）": half_time(drop_readout),
              "QK（移植で C が育つ）": half_time([r["C_QK"] for r in rows]),
              "OV（移植で C が育つ）": half_time([r["C_OV"] for r in rows])}
        order = " → ".join(k.split("（")[0] for k, _ in sorted(th.items(), key=lambda kv: (kv[1] is None, kv[1])))
        print("  半分まで動いたステップ: " + "、".join(f"{k} {v}" for k, v in th.items()) + f"  ⇒ 動く順: {order}")
        print(f"  readout だけの移植で C は最後まで {max(abs(r['C_readout']) for r in rows):.1e}"
              "（初期値の t2 の残差ストリームは2文で同じなので、出力層だけでは文脈を使えない）")
        ov_win = sum(r["C_OV"] > r["C_QK"] for r in rows[1:])
        print(f"  移植の C の絶対値：step 1 で QK {rows[1]['C_QK']:.4f}、OV {rows[1]['C_OV']:.4f}。OV > QK だったのは {ov_win}/{len(rows) - 1} ステップ"
              + ("（OV ≤ QK は step " + ", ".join(str(t) for t in range(1, len(rows)) if rows[t]["C_OV"] <= rows[t]["C_QK"]) + "）"
                 if ov_win < len(rows) - 1 else ""))
        gaps = [mirror_gap(p) for p in traj]
        dp = np.array([r["dp_t0"] for r in rows])
        dz = np.array([r["dz_t0"] for r in rows])
        first = next((t for t in range(len(dp)) if abs(dp[t]) > 1e-9), None)
        ups = [t for t in range(21, len(L)) if L[t] > L[t - 1]]
        jump = next((t for t in range(1, len(L)) if L[t] > L[t - 1] + 1e-3), None)
        grow = next((t for t in range(len(dz)) if abs(dz[t]) >= 1e-2), len(dz))   # 最初に |差| ≥ 1e-2 になるまでが「育つ」区間
        ratio = [dz[t] / dz[t - 1] for t in range(1, grow + 1) if t < len(dz) and abs(dz[t - 1]) > 1e-12]
        print(f"  ミラー対称性：θ と g(θ) の差の最大は Wout の 猫・犬 列で {max(a for a, _ in gaps):.1e}、それ以外のパラメータで {max(b for _, b in gaps):.1e}。"
              + (f"t0 の |p(猫) − p(犬)| の最大 {np.abs(dp).max():.3f}（step {int(np.abs(dp).argmax())}）" if np.abs(dp).max() > 1e-9
                 else f"t0 の |p(猫) − p(犬)| の最大 {np.abs(dp).max():.1e}"))
        print("  " + (f"t0 の |p(猫) − p(犬)| が 1e-9 を超えた最初の step {first}。" if first is not None else "t0 の |p(猫) − p(犬)| は最後まで 1e-9 以下。")
              + (f"t0 の logit 差（猫 − 犬）の1ステップの倍率 {min(ratio):+.2f}〜{max(ratio):+.2f}（|差| が 1e-12 を超えてから 1e-2 に届くまで）。" if ratio else ""))
        print(f"  step 20 以降に L が増えたステップ数 {len(ups)}"
              + (f"。L が 1e-3 以上増えた最初の step {jump}（{L[jump - 1]:.4f} → {L[jump]:.4f}）" if jump else ""))
    if csv:
        keys = list(records[0])
        with open(csv, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in records:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n毎ステップの記録を {csv} に書きました（{len(records)} 行、列: {', '.join(keys)}）")


def init_random(seed, scale):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(0.0, scale, s) for k, s in m2.SHAPES.items()}


def section_seeds(scale, steps):
    print("=" * 100)
    print(f"[seeds] 全パラメータを N(0, {scale:g}²) で初期化、seed 0–9、{steps} ステップ")
    print(f"  床 = L ≤ {FLOOR:.4f} + 0.01 に届いた、プラトー = L ≤ {NOCTX:.4f} + 0.01（文脈なしの最善）に届いた。a0, a1 = 最後の 猫文 t2 → t1 の注意")
    for name, step in (("SGD lr = 1", sgd(1.0)), ("Adam lr = 0.01", adam(0.01))):
        print(f"\n--- {name} ---")
        print(f"  {'seed':>4} {'L(0)':>7} {'プラトー到達':>12} {'床に到達':>8} {'最終 L':>8} {'最終 C':>10} {'a0':>6} {'a1':>6}  判定")
        tally = {}
        for seed in range(10):
            p, st, Ls, Cs = init_random(seed, scale), {}, [], []
            with np.errstate(all="ignore"):
                for t in range(steps + 1):
                    c = forward(p)
                    Ls.append(float(c["loss"]))
                    if not np.isfinite(Ls[-1]):
                        break
                    if t < steps:
                        p, st = step(p, grads(p, c), st)
            Ls = np.array(Ls)
            ok = np.isfinite(Ls[-1])
            c = rows(c) if ok else c
            C_final = contrast(c) if ok else float("nan")
            a_fin = [np.asarray(c["A"][h])[CAT_T2, CAT_T1] if ok else float("nan") for h in range(m.H)]
            hit = lambda thr: str(int(np.argmax(Ls <= thr))) if (Ls <= thr).any() else "—"
            if not np.isfinite(Ls[-1]):
                tag = "発散"
            elif Ls[-1] <= FLOOR + 0.01:
                tag = "床"
            elif abs(Ls[-1] - NOCTX) <= 0.01:
                tag = "プラトーで停滞"
            else:
                tag = "その他"
            tally[tag] = tally.get(tag, 0) + 1
            fin = f"{Ls[-1]:8.4f}" if np.isfinite(Ls[-1]) else f"{'nan':>8}"
            print(f"  {seed:>4} {Ls[0]:7.4f} {hit(NOCTX + 0.01):>12} {hit(FLOOR + 0.01):>8} {fin} {C_final:10.3g} {a_fin[0]:6.3f} {a_fin[1]:6.3f}  {tag}")
        print("  集計: " + "、".join(f"{k} {v}" for k, v in tally.items()))


def section_robust(p0, steps, n=20):
    print("=" * 100)
    print(f"[robust] 初期値の全成分に 1e-12·N(0,1) を足して {n} 回（{steps} ステップ）。丸め誤差ほどの違いで結果は変わるか")
    dead = [j for j in range(m.D) if not p0["Wq"][:, j].any() and not p0["Wk"][:, j].any()]   # 初期値で Q・K のその列が 0
    big = lambda q: max(float(np.abs(q[w][:, dead]).max()) for w in ("Wq", "Wk")) if dead else 0.0
    for name, step in (("SGD lr = 1", sgd(1.0)), ("SGD lr = 0.3", sgd(0.3)), ("Adam lr = 0.01", adam(0.01))):
        finals, diverged, dead_end = [], 0, []
        for k in range(1, n + 1):
            rng = np.random.default_rng(1000 + k)
            traj = train({kk: v + 1e-12 * rng.normal(size=v.shape) for kk, v in p0.items()}, step, steps)
            if len(traj) < steps + 1 or not np.isfinite(forward(traj[-1])["loss"]):
                diverged += 1
                continue
            finals.append(read(traj[-1], p0))
            dead_end.append(big(traj[-1]))
        traj0 = train(p0, step, steps)
        exact = read(traj0[-1], p0)
        rng_ = lambda key: f"{min(r[key] for r in finals):.4f}〜{max(r[key] for r in finals):.4f}" if finals else "—"
        print(f"\n--- {name} ---")
        print(f"  ずらさない : L {exact['L']:.4f}  C {exact['C']:.4f}  a0 {exact['a0']:.4f}  a1 {exact['a1']:.4f}  パターン項 {exact['pat']:.1e}")
        print(f"  ずらした {n} 回: 発散 {diverged} 回。残りの範囲 L {rng_('L')}  C {rng_('C')}  a0 {rng_('a0')}  a1 {rng_('a1')}"
              + (f"  パターン項 最大 {max(r['pat'] for r in finals):.1e}" if finals else ""))
        if dead and finals:
            print(f"  Wq・Wk の {'・'.join(f'd{j}' for j in dead)} 列（初期値で 0、勾配も厳密に 0）の |値| の最大：ずらさない {big(traj0[-1]):.1e}、"
                  f"ずらした {len(dead_end)} 回 {min(dead_end):.1e}〜{max(dead_end):.1e}")


def section_trap(p0):
    print("=" * 100)
    print("[trap] C は「読み出しの向き r = Wout[:,鳴く] − Wout[:,吠える]」と「t2 の残差ストリームの2文の差 Δ = X2[猫文 t2] − X2[犬文 t2]」の")
    print("       両方が無いと作れない（C = r·Δ）。両方 0 から始めたらどうなるか")
    nk, hk = m2.I_NAKU, m2.I_HOERU

    def show(p, step, steps, marks):
        traj = train(p, step, steps)
        print(f"  {'step':>5} {'L':>7} {'C':>10} {'|r|':>9} {'|Δ|':>9}")
        for t in marks:
            if t >= len(traj):
                print(f"  {t:>5}  発散")
                break
            q = traj[t]
            c = rows(forward(q))
            X2 = np.asarray(c["X2"]).reshape(len(TASK["inp"]), -1)
            r = np.linalg.norm(q["Wout"][:, nk] - q["Wout"][:, hk])
            print(f"  {t:>5} {float(c['loss']):7.4f} {contrast(c):10.3e} {r:9.2e} {np.linalg.norm(X2[CAT_T2] - X2[DOG_T2]):9.2e}")

    q = {k: v.copy() for k, v in p0.items()}
    avg = (q["Wout"][:, nk] + q["Wout"][:, hk]) / 2
    q["Wout"][:, nk], q["Wout"][:, hk] = avg, avg.copy()
    print(f"\n--- (a) Wout の 鳴く列と 吠える列を同じにする（両方を平均 {np.round(avg, 2).tolist()} に）、SGD lr = 0.3 ---")
    show(q, sgd(0.3), 300, (0, 1, 2, 3, 5, 10, 20, 50, 100, 300))
    q = {k: v.copy() for k, v in p0.items()}
    q["Wv"][:], q["Wo"][:] = 0.0, 0.0
    print("\n--- (b) Wv = Wo = 0 にする（Attention の出力も、その勾配も 0）、Adam lr = 0.01 ---")
    show(q, adam(0.01), 1000, (0, 1, 10, 50, 100, 300, 1000))


def main():
    global FWD, BWD, IMPL, SYMMETRIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("main", "seeds", "robust", "trap", "all"), default="all")
    ap.add_argument("--steps", type=int, help="main/robust の既定 300、seeds の既定 3000")
    ap.add_argument("--scale", type=float, default=0.02, help="seeds の初期化の標準偏差")
    ap.add_argument("--csv", help="main の毎ステップの記録を書き出す CSV のパス")
    ap.add_argument("--symmetrize", action="store_true", help="毎ステップ θ を (θ + g(θ))/2 に戻す（ミラー対称性を丸め誤差ごと保つ）")
    ap.add_argument("--impl", help="自分の K3 実装（forward(p, task), backward(p, c)）で回す。"
                    "forward の c に loss, logits, X0, X2, V, A, S, O が必要（N 行の形でも (B, T, …) でもよい）")
    a = ap.parse_args()
    SYMMETRIZE = a.symmetrize
    if SYMMETRIZE:
        print("--symmetrize：毎ステップ θ を (θ + g(θ))/2 に戻します")
    if a.impl:
        spec = importlib.util.spec_from_file_location("learner_impl", a.impl)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        FWD, BWD, IMPL = mod.forward, mod.backward, True
        print(f"実装: {a.impl}")
    p0, _ = m2.load_params()
    if IMPL:
        try:
            grads(p0, forward(p0))
            rows(forward(p0))
        except NotImplementedError as e:
            sys.exit(f"未実装: {e}（--impl には K3 の forward と backward の両方が要ります）")
    t0 = time.perf_counter()
    if a.section in ("main", "all"):
        section_main(p0, a.steps or 300, a.csv)
    if a.section in ("seeds", "all"):
        section_seeds(a.scale, a.steps or 3000)
    if a.section in ("robust", "all"):
        section_robust(p0, a.steps or 300)
    if a.section in ("trap", "all"):
        section_trap(p0)
    print(f"\n（所要 {time.perf_counter() - t0:.0f} 秒）")


if __name__ == "__main__":
    main()
