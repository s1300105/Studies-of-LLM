"""Part 2（2文タスク）の NumPy 正解実装。

モデルは Part 1（../mini_gpt.py）とまったく同じ。違うのはデータだけ：
    文A: <s> 猫 が → 猫 が 鳴く
    文B: <s> 犬 が → 犬 が 吠える
2文を6行に積み（A の t0, t1, t2, B の t0, t1, t2）、「同じ文の、自分以前の位置」だけを見られるマスクを使う。
位置埋め込み P は2文で共有する（各行が P の t0, t1, t2 行を使う）。損失は6位置の平均。

t2 の入力はどちらの文も「が」なので、Attention を通らない限り A と B の t2 は同じ計算になる。
「鳴く」か「吠える」かを決める情報は、t1（猫 / 犬）から Attention で運ばれてくるしかない。
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mini_gpt as m  # noqa: E402

VOCAB = ["<s>", "猫", "犬", "が", "鳴く", "吠える"]
INP = [0, 1, 3, 0, 2, 3]       # A: <s> 猫 が / B: <s> 犬 が
TGT = [1, 3, 4, 2, 3, 5]       # A: 猫 が 鳴く / B: 犬 が 吠える
POSI = [0, 1, 2, 0, 1, 2]
SENT = [0, 0, 0, 1, 1, 1]
N = len(INP)
MASK = np.array([[SENT[i] != SENT[j] or j > i for j in range(N)] for i in range(N)])
TASK2 = {"inp": INP, "tgt": TGT, "posi": POSI, "mask": MASK}
SHAPES = dict(m.SHAPES, E=(len(VOCAB), m.D), Wout=(m.D, len(VOCAB)))
PARAMS = list(SHAPES)
ROWS = [f"{'AB'[s]}{t} {VOCAB[i]}" for s, t, i in zip(SENT, POSI, INP)]
I_NAKU, I_HOERU = VOCAB.index("鳴く"), VOCAB.index("吠える")
A2, B2, A1, B1 = 2, 5, 1, 4    # 行番号：A の t2, B の t2, A の t1, B の t1


def load_params(path=HERE / "params.json"):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    p = {k: np.array(raw[k], dtype=float) for k in PARAMS}
    for k, s in SHAPES.items():
        assert p[k].shape == s, f"{k}: {p[k].shape} != {s}"
    return p, float(raw["lr"])


def forward(p):
    return m.forward(p, TASK2)


def backward(p, c):
    return m.backward(p, c)


def sgd(p, g, lr):
    return m.sgd(p, g, lr)


def contrast(c):
    """文脈の使い分け：(鳴く − 吠える) の logit 差が、文A の t2 と文B の t2 でどれだけ違うか。"""
    lg = c["logits"]
    return (lg[A2, I_NAKU] - lg[A2, I_HOERU]) - (lg[B2, I_NAKU] - lg[B2, I_HOERU])


# ---------------------------------------------------------------- 表示（チェックポイント）

D, H, DK, FF = m.D, m.H, m.DK, m.FF
DIMS, HID, KEYS = m.DIMS, m.HID, m.KEYS
POS3 = ["t0", "t1", "t2"]
SRC = {"<s>": 0, "A1 猫": A1, "B1 犬": B1, "が": A2}   # U1 の4種類の行（t0 と t2 は2文で共通）
BLOCK = {"A": slice(0, 3), "B": slice(3, 6)}


def blocks(M, label):
    """6×6 の注意系の行列を、文ごとの 3×3 ブロック2枚に分ける（文をまたぐ部分は常にマスク）。"""
    return [(f"{label}（文{s}）", M[b, b], [ROWS[i] for i in range(N)][b], KEYS) for s, b in BLOCK.items()]


def ablate_t2(p, c):
    """t2 の Attention 出力を0にしたときの t2 の logits（A と B は完全に同じになる）。"""
    X1 = c["X1"].copy()
    X1[[A2, B2]] = c["X0"][[A2, B2]]
    U2, _ = m.layernorm(X1)
    X2 = X1 + np.maximum(U2 @ p["W1"], 0) @ p["W2"]
    return (X2 @ p["Wout"])[[A2, B2]]


def qk_table(p, h):
    """ヘッド h の QK 回路：t2（が）のクエリから、4種類のキー行へのスコア（÷√2 済み）。"""
    sl = m.head_slice(h)
    Mh = p["Wq"][:, sl] @ p["Wk"][:, sl].T                  # 4×4：「どこを見るか」を決める双線形形式
    return Mh


def ov_table(p, c, h):
    """ヘッド h の OV 回路：各ソース行の値ベクトルが、Wo → Wout を通って各トークンの logit をいくら押すか。"""
    sl = m.head_slice(h)
    OVh = p["Wv"][:, sl] @ p["Wo"][sl, :] @ p["Wout"]       # 4×6：「何を運ぶか」
    U = np.vstack([c["U1"][i] for i in SRC.values()])
    return OVh, U @ OVh


def circuit_items(p, c, tag=""):
    items = []
    U = np.vstack([c["U1"][i] for i in SRC.values()])
    for h in range(H):
        Mh = qk_table(p, h)
        items.append((f"Wq{h} Wk{h}ᵀ{tag}（ヘッド{h} の QK 回路）", Mh, DIMS, DIMS))
        items.append((f"t2 クエリ → 各キー行のスコア（ヘッド{h}）{tag}", (c["U1"][A2] @ Mh @ U.T / np.sqrt(DK)).reshape(1, -1),
                      ["が(t2)"], list(SRC)))
    for h in range(H):
        _, rows = ov_table(p, c, h)
        items.append((f"U1[ソース] Wv{h} Wo{h} Wout{tag}（ヘッド{h} の OV 回路）", rows, list(SRC), VOCAB))
    return items


def contrast_items(p, c, tag=""):
    lg = c["logits"]
    ab = ablate_t2(p, c)
    return [(f"t2 の (鳴く − 吠える){tag}", np.array([[lg[A2, I_NAKU] - lg[A2, I_HOERU], lg[B2, I_NAKU] - lg[B2, I_HOERU], contrast(c)]]),
             ["値"], ["文A", "文B", "差 C = A − B"]),
            (f"t2 の Attention 出力を0にした logits{tag}（アブレーション）", ab, ["A2 が", "B2 が"], VOCAB)]


def checkpoints(p, lr):
    c = forward(p)
    g, d = backward(p, c)
    p1 = sgd(p, g, lr)
    c1 = forward(p1)
    gsq = sum(float((g[k] ** 2).sum()) for k in PARAMS)
    col = lambda v: np.asarray(v).reshape(-1, 1)
    cp = []
    add = lambda cid, title, items: cp.append((cid, title, items))
    stats = lambda ln: np.hstack([ln["mu"], ln["var"], ln["sigma"]])
    R = ROWS

    add("2.1", "埋め込み + 位置埋め込み", [("E[ids]", p["E"][INP], R, DIMS), ("P[位置]", p["P"][POSI], R, DIMS), ("X0 = E[ids] + P[位置]", c["X0"], R, DIMS)])
    add("2.2", "LayerNorm1 の統計量", [("μ, σ², σ", stats(c["ln1"]), R, ["μ", "σ²", "σ"])])
    add("2.3", "LayerNorm1 の出力", [("U1 = (X0 - μ) / σ", c["U1"], R, DIMS)])
    add("3.1", "Q, K, V", [("Q = U1 Wq", c["Q"], R, DIMS), ("K = U1 Wk", c["K"], R, DIMS), ("V = U1 Wv", c["V"], R, DIMS)])
    add("3.2", "スコア（文ごと・マスク付き）", sum((blocks(c["S"][h], f"S{h} = Q{h} K{h}ᵀ / √2") for h in range(H)), []))
    add("3.3", "Attention重み", sum((blocks(c["A"][h], f"A{h} = softmax(S{h})") for h in range(H)), []))
    add("3.4", "ヘッド出力と連結", [(f"O{h} = A{h} V{h}", c["O"][h], R, DIMS[h * DK:(h + 1) * DK]) for h in range(H)]
        + [("Ocat = [O0 | O1]", c["Ocat"], R, DIMS)])
    add("3.5", "出力射影", [("Attn = Ocat Wo", c["Attn"], R, DIMS)])
    add("4.1", "残差接続1", [("X1 = X0 + Attn", c["X1"], R, DIMS)])
    add("4.2", "LayerNorm2", [("μ, σ², σ", stats(c["ln2"]), R, ["μ", "σ²", "σ"]), ("U2 = (X1 - μ) / σ", c["U2"], R, DIMS)])
    add("4.3", "FFN 1層目 + ReLU", [("Z = U2 W1", c["Z"], R, HID), ("R = ReLU(Z)", c["R"], R, HID)])
    add("4.4", "FFN 2層目", [("F = R W2", c["F"], R, DIMS)])
    add("4.5", "残差接続2", [("X2 = X1 + F", c["X2"], R, DIMS)])
    add("5.1", "ロジット", [("logits = X2 Wout", c["logits"], R, VOCAB)])
    add("5.2", "softmax", [("p = softmax(logits)", c["probs"], R, VOCAB)])
    add("5.3", "損失", [("-log p[正解]", col(c["nll"]), R, ["nll"]), ("L（6位置の平均）", np.array([[c["loss"]]]), ["L"], ["値"])])
    add("5.4", "文脈の使い分けとアブレーション", contrast_items(p, c))

    add("6.1", "出力層の逆伝播", [("dlogits = (p - y) / 6", d["logits"], R, VOCAB), ("dWout = X2ᵀ dlogits", g["Wout"], DIMS, VOCAB),
                               ("dX2 = dlogits Woutᵀ", d["X2"], R, DIMS)])
    add("6.2", "FFN の逆伝播", [("dF = dX2（残差で素通り）", d["F"], R, DIMS), ("dW2 = Rᵀ dF", g["W2"], HID, DIMS),
                              ("dR = dF W2ᵀ", d["R"], R, HID), ("dZ = dR ⊙ [Z>0]", d["Z"], R, HID),
                              ("dW1 = U2ᵀ dZ", g["W1"], DIMS, HID), ("dU2 = dZ W1ᵀ", d["U2"], R, DIMS)])
    add("6.3", "LayerNorm2 の逆伝播と残差の合流", [("LN2経由", d["X1_ln2"], R, DIMS), ("dX1 = dX2 + LN2経由", d["X1"], R, DIMS)])
    add("6.4", "出力射影の逆伝播", [("dAttn = dX1", d["Attn"], R, DIMS), ("dWo = Ocatᵀ dAttn", g["Wo"], DIMS, DIMS),
                                ("dOcat = dAttn Woᵀ", d["Ocat"], R, DIMS)])
    items = []
    for h in range(H):
        items += blocks(d["A"][h], f"dA{h} = dO{h} V{h}ᵀ") + blocks(d["S"][h], f"dS{h} = A{h} ⊙ (dA{h} - rowsum(dA{h} ⊙ A{h}))")
    add("6.5", "Attention重み・softmax の逆伝播", items)
    items = []
    for h in range(H):
        sl = m.head_slice(h)
        items += [(f"dQ{h} = dS{h} K{h} / √2", d["Q"][:, sl], R, DIMS[sl]), (f"dK{h} = dS{h}ᵀ Q{h} / √2", d["K"][:, sl], R, DIMS[sl]),
                  (f"dV{h} = A{h}ᵀ dO{h}", d["V"][:, sl], R, DIMS[sl])]
    add("6.6", "Q, K, V の勾配", items)
    add("6.7", "射影行列の勾配と dU1", [("dWq = U1ᵀ dQ", g["Wq"], DIMS, DIMS), ("dWk = U1ᵀ dK", g["Wk"], DIMS, DIMS),
                                   ("dWv = U1ᵀ dV", g["Wv"], DIMS, DIMS), ("dU1 = dQ Wqᵀ + dK Wkᵀ + dV Wvᵀ", d["U1"], R, DIMS)])
    add("6.8", "LayerNorm1 の逆伝播と残差の合流", [("LN1経由", d["X0_ln1"], R, DIMS), ("dX0 = dX1 + LN1経由", d["X0"], R, DIMS)])
    add("6.9", "埋め込みの勾配", [("dP（同じ位置の A, B の行を足す）", g["P"], POS3, DIMS), ("dE（使われた行にだけ加算）", g["E"], VOCAB, DIMS)])

    add("7.1", "SGD更新後のパラメータ", [(f"{k}' = {k} - {lr:g}·d{k}", p1[k], *param_labels(k)) for k in PARAMS])
    add("7.2", "損失変化の1次近似", [("‖g‖², 予測 ΔL = -lr‖g‖²", np.array([[gsq, -lr * gsq]]), ["値"], ["‖g‖²", "予測ΔL"])])
    items = [("X0'", c1["X0"], R, DIMS), ("μ', σ²', σ'（LN1）", stats(c1["ln1"]), R, ["μ", "σ²", "σ"]), ("U1'", c1["U1"], R, DIMS),
             ("Q'", c1["Q"], R, DIMS), ("K'", c1["K"], R, DIMS), ("V'", c1["V"], R, DIMS)]
    items += sum((blocks(c1["S"][h], f"S{h}'") for h in range(H)), []) + sum((blocks(c1["A"][h], f"A{h}'") for h in range(H)), [])
    items += [("Ocat'", c1["Ocat"], R, DIMS), ("Attn'", c1["Attn"], R, DIMS), ("X1'", c1["X1"], R, DIMS),
              ("μ', σ²', σ'（LN2）", stats(c1["ln2"]), R, ["μ", "σ²", "σ"]), ("U2'", c1["U2"], R, DIMS), ("Z'", c1["Z"], R, HID),
              ("F'", c1["F"], R, DIMS), ("X2'", c1["X2"], R, DIMS), ("logits'", c1["logits"], R, VOCAB), ("p'", c1["probs"], R, VOCAB),
              ("-log p'[正解]", col(c1["nll"]), R, ["nll"])]
    add("7.3", "再順伝播", items)
    add("7.4", "比較", [("L → L'", np.array([[c["loss"], c1["loss"], c1["loss"] - c["loss"], -lr * gsq]]), ["値"],
                       ["L（更新前）", "L'（更新後）", "実際のΔL", "予測ΔL"]),
                      ("正解トークンの確率 前→後", np.vstack([c["probs"][np.arange(N), TGT], c1["probs"][np.arange(N), TGT]]).T, R, ["更新前", "更新後"])]
        + contrast_items(p1, c1, "'"))
    add("8.1", "回路で読む（更新前）", circuit_items(p, c))
    add("8.2", "回路で読む（更新後）", circuit_items(p1, c1, "'"))
    return cp


def param_labels(k):
    rows = {"E": VOCAB, "P": POS3, "W2": HID}.get(k, DIMS)
    cols = {"W1": HID, "Wout": VOCAB}.get(k, DIMS)
    return rows, cols


SHARED = {  # 順伝播で「A の行と同じ」になる B の行（計算は1回でよい）
    **{cid: [3] for cid in ("2.1", "2.2", "2.3", "3.4", "3.5", "4.1", "4.2", "4.3", "4.4", "4.5", "5.1", "5.2")},
    "3.1": [3, 5],
}


def skip_cells(cid, label, M):
    """記入用シートの印：'—' = 計算不要、'=' = A の同じ位置の行と同じ値。"""
    k = np.full(np.atleast_2d(M).shape, "", dtype=object)
    name = label.split()[0]
    if k.shape[0] == N:
        if cid in SHARED and not name.startswith("P["):
            k[SHARED[cid]] = "＝A"
            if cid == "2.1" and name == "E[ids]":
                k[5] = "＝A"
        if (cid, name) in (("3.1", "Q"), ("7.3", "Q'")):
            k[[0, 3]] = "—"                              # t0 のクエリは結果に影響しない
    if k.shape == (3, 3) and (cid in ("3.2",) or (cid == "7.3" and name.startswith("S")) or (cid == "6.5" and name.startswith("dA"))):
        k[0, 0] = "—"
    return k


def render_markdown(cp, dec=4, blank=False):
    if blank:
        out = ["# Part 2 記入用シート（mini_gpt2.py --blank が自動生成）", "",
               "「−∞」「0*」は因果マスクで決まる値、「—」は計算しなくてよいセル、「＝A」は文A の同じ位置の行と同じ値になるセルです"
               "（t0 は2文とも `<s>` だけを見るので順伝播は全部同じ。t2 は Attention の前までは同じ）。", ""]
    else:
        out = ["# Part 2 解答集（mini_gpt2.py --markdown が自動生成）", "",
               f"数値は小数点以下{dec}桁。「0」は厳密に0、「0.0000」は丸めると0になる小さな値。許容差は |自分 − 解答| ≤ 0.002 + 0.002·|解答|。", ""]
    for cid, title, items in cp:
        out += [f"## {cid} {title}", ""]
        for label, M, rl, cl in items:
            M = np.atleast_2d(M)
            masked = np.broadcast_to(m.MASK, M.shape) if m.is_masked(label, M) else np.zeros(M.shape, dtype=bool)
            skip = skip_cells(cid, label, M)
            out += [f"**{m.md_escape(label)}**", "", "| | " + " | ".join(map(m.md_escape, cl)) + " |", "|---|" + "---:|" * len(cl)]
            for r, row, mrow, krow in zip(rl, M, masked, skip):
                if blank:
                    cells = [("−∞" if np.isneginf(v) else "0*") if mk else (k or " ") for v, mk, k in zip(row, mrow, krow)]
                else:
                    cells = [m.fmt(v, dec) for v in row]
                out.append("| " + m.md_escape(r) + " | " + " | ".join(cells) + " |")
            out.append("")
    return "\n".join(out)


def render_params(p, lr):
    out = ["# Part 2 初期パラメータ（mini_gpt2.py --params-md が自動生成）", "", f"学習率 lr = {lr:g}（SGD）。正本は params.json。", ""]
    for k in PARAMS:
        rl, cl = param_labels(k)
        out += [f"**{k}**", "", "| | " + " | ".join(map(m.md_escape, cl)) + " |", "|---|" + "---:|" * len(cl)]
        out += ["| " + m.md_escape(r) + " | " + " | ".join(f"{v:g}".replace("-", "−") for v in row) + " |" for r, row in zip(rl, p[k])]
        out.append("")
    return "\n".join(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(HERE / "params.json"))
    ap.add_argument("--decimals", type=int, default=4)
    ap.add_argument("--step", type=int)
    ap.add_argument("--only")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--blank", action="store_true")
    ap.add_argument("--params-md", action="store_true")
    ap.add_argument("--gradcheck", action="store_true")
    ap.add_argument("--train", type=int, metavar="STEPS")
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    p, lr = load_params(a.params)
    lr = a.lr if a.lr is not None else lr
    if a.gradcheck:
        c = forward(p)
        g, _ = backward(p, c)
        for k, e in m.grad_check(p, g, task=TASK2).items():
            print(f"{k:5s} 数値微分との最大誤差 {e:.2e}")
        return
    if a.train:
        q = p
        for s in range(a.train + 1):
            c = forward(q)
            if s % max(1, a.train // 20) == 0 or s == a.train:
                pred = " ".join(VOCAB[i] for i in c["probs"].argmax(-1))
                print(f"step {s:4d}  loss {c['loss']:.4f}  C {contrast(c):7.3f}  予測: {pred}")
            if s < a.train:
                q = sgd(q, backward(q, c)[0], lr)
        return
    if a.params_md:
        print(render_params(p, lr))
        return
    cp = checkpoints(p, lr)
    if a.markdown or a.blank:
        print(render_markdown(cp, a.decimals, blank=a.blank))
    else:
        print(m.render_text(cp, a.decimals, a.only, a.step))


if __name__ == "__main__":
    main()
