"""Part 2 セクション G「重みを手で設計する」の判定スクリプト。

固定するもの: E, P（part2/params.json）、W1 = W2 = 0（FFN なし）、LN（γ=1, β=0, ε=0）。
学習者が決めるもの: Wq, Wk, Wv, Wo（4×4）, Wout（4×6、列 = <s>, 猫, 犬, が, 鳴く, 吠える）。

使い方:
  python3 part2/check_design.py my_design.json        # 自分の設計を判定
  python3 part2/check_design.py                       # 引数なし = part2/params.json の重み（FAIL するはず）
  python3 part2/check_design.py --template > my.json  # params.json の Wq..Wout を書き出す（出発点に）

JSON には Wq, Wk, Wv, Wo, Wout を行のリストで書きます（例: "Wo": [[1,-1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]）。
書かなかった行列は params.json の値を使います。E, P, W1, W2, lr などのキーは無視します（"_" で始まるキーはメモ用）。

判定:
  [必須]  猫文・犬文の両方で t2 の nll < 0.1（p(正解) > 0.905）→ これを満たせば PASS
  [発展A] 両文の t1 で nll < 0.1（t1 → が）
  [発展B] L − 下限(2·ln2/6) < 0.05（t0 は 猫/犬 を当てられないので L は 0.2310 より下がらない）
順伝播は ../mini_gpt.py の forward を part2/mini_gpt2.py の TASK2（6行・文ごとの因果マスク）で呼びます。
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mini_gpt2 as m2  # noqa: E402

m = m2.m
VOCAB, ROWS, TGT, N = m2.VOCAB, m2.ROWS, m2.TGT, m2.N
NAKU, HOERU = m2.I_NAKU, m2.I_HOERU
A1, A2, B1, B2 = m2.A1, m2.A2, m2.B1, m2.B2     # 猫文 t1, 猫文 t2, 犬文 t1, 犬文 t2
DESIGN = ["Wq", "Wk", "Wv", "Wo", "Wout"]
THR = 0.1                                      # nll の合格ライン
FLOOR = 2 * np.log(2) / 6                      # 損失の下限（t0 の2行が ln2 ずつ）
EPS = 1e-9
# ミラー：d0↔d1 と 猫↔犬・鳴く↔吠える を同時に入れ替えると猫文が犬文に移る
MIRROR_ROW = [3, 4, 5, 0, 1, 2]
MIRROR_TOK = [0, 2, 1, 3, 5, 4]


def fmt(x, dec=4):
    return m.fmt(float(x), dec)


def vec(v, dec=4):
    return "(" + ", ".join(fmt(x, dec) for x in v) + ")"


def pad(s, w, right=False):
    """全角文字を幅2として揃える。"""
    n = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    return (" " * max(0, w - n) + s) if right else (s + " " * max(0, w - n))


def load_design(path):
    base, _ = m2.load_params()
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        sys.exit("エラー: JSON の一番外側は {\"Wq\": [...], ...} の形にしてください。")
    p = {k: base[k].copy() for k in m2.PARAMS}
    p["W1"][:] = 0.0
    p["W2"][:] = 0.0
    notes = []
    for k in DESIGN:
        if k not in raw:
            notes.append(f"{k} は未指定なので params.json の値を使います")
            continue
        try:
            a = np.array(raw[k], dtype=float)
        except (TypeError, ValueError):
            sys.exit(f"エラー: {k} に数値以外が入っています（分数は 0.5 のように小数で書いてください）。")
        if a.shape != m2.SHAPES[k]:
            sys.exit(f"エラー: {k} の形が {a.shape} です。{m2.SHAPES[k]}（行 × 列）にしてください。")
        if not np.all(np.isfinite(a)):
            sys.exit(f"エラー: {k} に inf か nan が入っています。")
        p[k] = a
    ignored = [k for k in raw if k not in DESIGN and not k.startswith("_")]   # "_" で始まるキーはコメント扱い
    if ignored:
        notes.append("無視したキー: " + ", ".join(ignored) + "（E, P は params.json、W1 = W2 = 0 に固定）")
    return p, notes


def run(p):
    """順伝播。W1 = W2 = 0 なので F = 0、logits = X1 Wout（LN2 の σ が0でも結果は同じ）。"""
    with np.errstate(all="ignore"):
        c = m.forward(p, m2.TASK2)
    if not np.all(np.isfinite(c["logits"])):     # X1 の行が定数で LN2 が 0/0 になったときだけ
        c["F"] = np.zeros_like(c["X1"])
        c["X2"] = c["X1"].copy()
        c["logits"] = c["X2"] @ p["Wout"]
        c["probs"] = m.softmax(c["logits"])
        mx = c["logits"].max(-1)
        lse = np.log(np.exp(c["logits"] - mx[:, None]).sum(-1)) + mx
        c["nll"] = lse - c["logits"][np.arange(N), TGT]
        c["loss"] = c["nll"].mean()
    return c


def head_terms(p, c, h):
    """ヘッド h が t2 で運ぶ (鳴く − 吠える) の差（猫文 − 犬文）の分解。FFN なしなので直接効果 = 総効果。"""
    sl = m.head_slice(h)
    r = p["Wout"][:, NAKU] - p["Wout"][:, HOERU]
    aA, aB = c["A"][h][A2, 0:3], c["A"][h][B2, 3:6]
    VA, VB = c["V"][0:3, sl], c["V"][3:6, sl]
    proj = lambda o: float(o @ p["Wo"][sl, :] @ r)
    return proj((aA - aB) @ VA), proj(aB @ (VA - VB))


def gap(c, row):
    lg = c["logits"]
    return lg[row, NAKU] - lg[row, HOERU]


def rival(c, row):
    """正解以外で logit が最大の語と、正解との logit 差。"""
    lg = c["logits"][row]
    t = TGT[row]
    j = max((j for j in range(len(VOCAB)) if j != t), key=lambda j: lg[j])
    return j, lg[t] - lg[j]


def close_rivals(c, row, margin=3.9):
    """正解との logit 差が margin 未満の語（差の小さい順）。"""
    lg = c["logits"][row]
    t = TGT[row]
    js = sorted((j for j in range(len(VOCAB)) if j != t and lg[t] - lg[j] < margin), key=lambda j: lg[t] - lg[j])
    return [(j, lg[t] - lg[j]) for j in js]


def report(p, c, notes, src):
    out = []
    say = out.append
    say(f"=== Part 2 セクション G 設計チェック: {src} ===")
    say("固定: E, P = part2/params.json / W1 = W2 = 0（FFN なし）/ LN（γ=1, β=0）。U1 = X0（σ1 = 1）")
    for n in notes:
        say("  * " + n)

    say("\n[1] 行ごとの結果")
    say("  " + pad("行", 14) + pad("正解", 8) + pad("予測", 8) + pad("p(正解)", 9, True) + pad("nll", 9, True))
    for i in range(N):
        pr = c["probs"][i]
        top = [VOCAB[j] for j in range(len(VOCAB)) if pr[j] >= pr.max() - 1e-12]   # 同点は「猫=犬」のように書く
        say("  " + pad(ROWS[i], 14) + pad(VOCAB[TGT[i]], 8) + pad("=".join(top), 8)
            + pad(fmt(pr[TGT[i]]), 9, True) + pad(fmt(c["nll"][i]), 9, True))
    say(f"  L = {fmt(c['loss'])}（下限 2·ln2/6 = {fmt(FLOOR)}、差 {fmt(c['loss'] - FLOOR)}）")

    say("\n[2] Attention 重み（t2 と t1 のクエリ行。k0, k1, k2 = 同じ文の t0, t1, t2）")
    for h in range(m.H):
        for nm, rows in (("猫文", (1, 2)), ("犬文", (4, 5))):
            base = 0 if nm == "猫文" else 3
            a1 = c["A"][h][rows[0], base:base + 2]
            a2 = c["A"][h][rows[1], base:base + 3]
            say(f"  ヘッド{h} {nm}: t2 → {vec(a2)}   t1 → {vec(a1)}")

    r = p["Wout"][:, NAKU] - p["Wout"][:, HOERU]
    C = gap(c, A2) - gap(c, B2)
    say("\n[3] 文脈の使い分け（t2 の 鳴く − 吠える）")
    say(f"  r = Wout[:,鳴く] − Wout[:,吠える] = {vec(r)}")
    say(f"  猫文 t2: {fmt(gap(c, A2))}   犬文 t2: {fmt(gap(c, B2))}   C = 猫文 − 犬文 = {fmt(C)}（目安: C > 4.5 が必要）")
    terms = [head_terms(p, c, h) for h in range(m.H)]
    for h, (pat, val) in enumerate(terms):
        say(f"  ヘッド{h} の寄与: パターン項 {fmt(pat)} + 値項 {fmt(val)} = {fmt(pat + val)}")
    say("  （X0 の項は t2 の X0 が2文で同じなので0。FFN が無いので C = 各ヘッドの寄与の和）")

    mir = np.allclose(c["probs"][MIRROR_ROW][:, MIRROR_TOK], c["probs"], atol=1e-9)
    say(f"  ミラー対称（犬文の確率 = 猫文の確率を 猫↔犬・鳴く↔吠える で入れ替えたもの）: {'はい' if mir else 'いいえ'}")
    nz = ", ".join(f"{k} {int(np.count_nonzero(np.abs(p[k]) > EPS))}" for k in DESIGN)
    say(f"  0でない成分の数: {nz}")

    req = [c["nll"][A2] < THR, c["nll"][B2] < THR]
    advA = [c["nll"][A1] < THR, c["nll"][B1] < THR]
    advB = c["loss"] - FLOOR < 0.05
    mark = lambda ok: "OK  " if ok else "NG  "
    say("\n[4] 判定")
    say(f"  [必須]  {mark(all(req))}猫文 t2 nll {fmt(c['nll'][A2])}、犬文 t2 nll {fmt(c['nll'][B2])}（どちらも < {THR}）")
    say(f"  [発展A] {mark(all(advA))}猫文 t1 nll {fmt(c['nll'][A1])}、犬文 t1 nll {fmt(c['nll'][B1])}（どちらも < {THR}）")
    say(f"  [発展B] {mark(advB)}L − 下限 = {fmt(c['loss'] - FLOOR)}（< 0.05）")
    ok = all(req)
    say("  → " + ("PASS（必須条件を満たしました）" if ok else "FAIL（必須条件を満たしていません）"))

    hints = diagnose(p, c, terms, C, r, req, advA, advB, mir)
    if hints:
        say("\n[5] 診断とヒント")
        if mir:
            say("  （ミラー対称なので猫文だけ診断します。犬文は 猫↔犬・鳴く↔吠える を入れ替えた同じ状況です）")
        for hnt in hints:
            say("      " + hnt.strip() if hnt.startswith("  →") else "  - " + hnt)
    return "\n".join(out), ok


def diagnose(p, c, terms, C, r, req, advA, advB, mir=False):
    hs = []
    t2rows = ((A2, "猫文"),) if mir else ((A2, "猫文"), (B2, "犬文"))
    t1rows = ((A1, "猫文"),) if mir else ((A1, "猫文"), (B1, "犬文"))
    U1 = c["U1"]
    dnoun = U1[A1] - U1[B1]                                   # U1[猫] − U1[犬] = (2, −2, 0, 0)
    contrib = [pat + val for pat, val in terms]
    big = max(abs(x) for x in contrib)
    if not all(req):
        if np.abs(r).max() < EPS:
            hs.append("r = 0：Wout の 鳴く 列と 吠える 列が同じです。t2 にどんな残差が届いても 鳴く と 吠える の差は出ません。")
        carry = [dnoun @ p["Wv"][:, m.head_slice(h)] for h in range(m.H)]   # = 2(Wv_h[d0] − Wv_h[d1])
        write = [carry[h] @ p["Wo"][m.head_slice(h), :] for h in range(m.H)]
        if np.abs(r).max() >= EPS and all(np.abs(v).max() < EPS for v in carry):
            hs.append("OV が 猫/犬 の差を運んでいません：(U1[猫] − U1[犬]) Wv = 2·(Wv[d0行] − Wv[d1行]) が両ヘッドで0です。"
                      "名詞の違いは d0 − d1 方向にしかないので、Wv の d0 行と d1 行を違う値にしてください。")
        elif np.abs(r).max() >= EPS and all(np.abs(w).max() < EPS for w in write):
            hs.append("Wv は 猫/犬 の差を読んでいますが、Wo がそれを残差に書き戻していません（使っているヘッドの Wo の行が0）。")
        elif np.abs(r).max() >= EPS and big < EPS:
            hs.append("OV は 猫/犬 の差を残差に書いていますが、その方向を r が読んでいません（Wo で書く向き · r = 0）。"
                      "Wo で書き込む向きと r を揃えてください。")
        pos = [x for x in contrib if x > 0.05 * big]
        neg = [x for x in contrib if x < -0.05 * big]
        cancel = bool(pos and neg and abs(C) < 0.5 * big)
        if cancel:
            hs.append("ヘッドが打ち消し合っています（寄与 " + " と ".join(fmt(x) for x in contrib) +
                      f"、和 C = {fmt(C)}）。片方のヘッドを0にするか、符号を揃えてください。")
        if C < -EPS and not cancel:
            hs.append(f"C = {fmt(C)} < 0：符号が逆です（猫文で 吠える 側に押している）。Wo か r の符号を反転してください。")
        weak = []
        for h in range(m.H):
            if abs(contrib[h]) > EPS or np.abs(carry[h]).max() > EPS:
                a = c["A"][h][A2, 1]
                if a < 0.9:
                    weak.append(f"ヘッド{h} a = {fmt(a)}、スコア t2 → (k0, k1, k2) = {vec(c['S'][h][A2, 0:3])}")
        if weak:
            hs.append("t2 → t1 の注意が弱い（" + "；".join(weak) + "）。ヘッドの寄与は a(t2→t1) 倍に縮みます。"
                      "k0 と k2 のスコアを0にそろえたとき、k1 のスコアが 2.89 で a = 0.9、3.64 で 0.95、5.29 で 0.99。"
                      "Q も K も位置の次元（d2, d3）から作りましょう。")
        for h in range(m.H):
            if abs(terms[h][0]) > 1e-6:
                hs.append(f"ヘッド{h} のパターン項が0ではありません（{fmt(terms[h][0])}）：QK が単語の次元を読み、"
                          "猫文と犬文で注意の向きが変わっています。悪いことではありませんが、意図したものか確かめましょう。")
        for row, nm in t2rows:
            if c["nll"][row] >= THR:
                cl = close_rivals(c, row) or [rival(c, row)]
                hs.append(f"{nm} t2：正解との logit 差が 3.9 未満の語 = " + "、".join(f"{VOCAB[j]} {fmt(g)}" for j, g in cl)
                          + "（5語とも差 3.9 以上あれば確実に nll < 0.1）。")
                pair = [(j, g) for j, g in cl if j in (NAKU, HOERU)]
                other = [VOCAB[j] for j, g in cl if j not in (NAKU, HOERU)]
                if pair and C >= 4.5 and pair[0][1] < C / 2 - EPS:
                    center = (gap(c, A2) + gap(c, B2)) / 2
                    hs.append(f"  → {nm} t2 では 正解 と {VOCAB[pair[0][0]]} の logit 差が {fmt(pair[0][1])} しかありません。C = {fmt(C)} は足りていますが、"
                              f"2文の (鳴く − 吠える) の中心 (猫文 + 犬文)/2 = {fmt(center)} が 0 からずれています。"
                              f"r が2文に共通の t2 の残差を読んでいるためです（X0[t2]·r = {fmt(c['X0'][A2] @ r)}）。"
                              "r を X0[t2] = (−1, −1, 1, 1) と直交させてください。")
                elif pair and C > EPS and not cancel:
                    hs.append(f"  → {nm} t2 では 正解 と {VOCAB[pair[0][0]]} の logit 差が {fmt(pair[0][1])} しかありません（文脈の差 C 不足）。"
                              "他の4語を十分押さえたとしても C > 4.5（片文あたり差 > 2.25）が必要です。Wv・Wo・r のどれかを大きくしてください。")
                elif pair:
                    hs.append(f"  → {nm} t2 では 正解 と {VOCAB[pair[0][0]]} の logit 差が {fmt(pair[0][1])} です。"
                              "Wv・Wo・r を大きくしても直りません。先に上の符号・向き・打ち消し合いの問題を直してください。")
                if other:
                    hs.append(f"  → {nm} t2 で {'・'.join(other)} が近すぎます（鳴く・吠える の共通の底上げ不足）。Wout の 鳴く・吠える 列に"
                              f" t2 を読む成分（d2 + d3 方向：t2 で +2、t0 で −2）を足すか、{'・'.join(other)} の列を t2 で下げましょう。")
    if not all(advA):
        for row, nm in t1rows:
            if c["nll"][row] >= THR:
                j, g = rival(c, row)
                rank = "1番手（予測）" if g < 0 else "2番手"
                msg = f"[発展A] {nm} t1：{rank}は {VOCAB[j]}（が との logit 差 {fmt(g)}、目安 ≥ 3.9）。"
                if j in (NAKU, HOERU):
                    msg += (f"t1 の残差にも名詞があるので r が t1 で 鳴く/吠える を押しています（X0[t1]·r = {fmt(c['X0'][row] @ r)}）。"
                            "が の列に t1 を読む成分（d2 − d3 方向）を大きめに足すか、r を小さくして Wo を大きくしましょう。")
                else:
                    msg += "が の列に t1 を読む成分（d2 − d3 方向：t1 で +2、t0 と t2 で 0）を足しましょう。"
                hs.append(msg)
        for h in range(m.H):
            self_a = c["A"][h][A1, 1]
            if self_a > 0.1 and abs(contrib[h]) > EPS:
                hs.append(f"[発展A] ヘッド{h} は t1 で自分自身（名詞）を {fmt(self_a)} 見ています。t1 にも 鳴く/吠える の押しが入ります。"
                          "t1 のクエリが k1 を嫌うように（例：Q を d3 から作ると t1 のクエリは t2 と逆符号）すると防げます。")
    if not advB:
        ex = c["nll"] - np.where(np.array(m2.POSI) == 0, np.log(2), 0.0)       # 各行の「下限からの超過」
        part = {f"t{t}": ex[np.array(m2.POSI) == t].sum() / N for t in range(3)}
        hs.append("[発展B] L − 下限 の内訳: " + "、".join(f"{k} の2行 {fmt(v)}" for k, v in part.items()) + "（和 = L − 下限）。")
        if part["t0"] >= max(part.values()) - EPS:
            t0 = c["probs"][0]
            hs.append(f"  → t0 で 猫・犬 以外に確率が {fmt(1 - t0[1] - t0[2])} 漏れています（t0 の nll {fmt(c['nll'][0])}、理想は ln2 = 0.6931）。"
                      "猫・犬 の列に t0 を読む成分（−(d2 + d3) 方向：t0 で +2、t2 で −2）を足しましょう。")
        else:
            hs.append("  → いちばん大きいのは t0 以外の行です。上の [必須]・[発展A] の診断から先に直しましょう。")
    return hs


def template():
    base, _ = m2.load_params()
    lines = ["{"]
    for i, k in enumerate(DESIGN):
        rows = ",\n    ".join("[" + ", ".join(f"{v:g}" for v in row) + "]" for row in base[k])
        lines.append(f'  "{k}": [\n    {rows}\n  ]' + ("," if i < len(DESIGN) - 1 else ""))
    lines.append("}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Part 2 セクション G の設計チェック")
    ap.add_argument("design", nargs="?", help="設計 JSON（省略すると part2/params.json）")
    ap.add_argument("--template", action="store_true", help="params.json の Wq..Wout を JSON で書き出す")
    ap.add_argument("--export", metavar="OUT", help="E, P, W1 = W2 = 0, lr = 1 を補った完全なパラメータを書き出す"
                    "（python3 part2/mini_gpt2.py --params OUT --only 5.5 などで自分の設計を回路として読める）")
    a = ap.parse_args()
    if a.template:
        print(template())
        return
    path = Path(a.design) if a.design else HERE / "params.json"
    p, notes = load_design(path)
    if a.export:
        full = {k: p[k].tolist() for k in m2.PARAMS}
        full["lr"] = 1.0
        Path(a.export).write_text(json.dumps(full, ensure_ascii=False), encoding="utf-8")
        notes.append(f"完全なパラメータを {a.export} に書き出しました")
    if not a.design:
        notes.insert(0, "引数なし：part2/params.json の Wq, Wk, Wv, Wo, Wout を判定します（FFN を外すとどうなるか）")
    c = run(p)
    text, ok = report(p, c, notes, path.name)
    print(text)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
