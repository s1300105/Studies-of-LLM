"""K3 行列レベルの自前実装 — あなたが埋める骨組み。

**mini_gpt.py を見ずに**、README.md §2〜3（仕様と記号）と workbook/derivations.md（逆伝播の式）だけを見て書きます。
書けたら照合します（hand_transformer/ で）:

    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --part 1     # まず Part 1（3行）
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --part 2     # 次に Part 2（2文・6行）に一般化
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py              # 両方
    python3 part3/k3_check.py part3/k3_matrix/my_gpt.py --gradcheck  # 自分の forward だけで数値微分（正解を使わない検算）

---------------------------------------------------------------- API（k3_check.py が呼ぶもの）

forward(p, task) -> c（辞書）
    p    : {"E": V×4, "P": 3×4, "Wq", "Wk", "Wv", "Wo": 4×4, "W1": 4×8, "W2": 8×4, "Wout": 4×V} の NumPy 配列（V = 語彙数。Part 1 は5、Part 2 は6）
    task : {"inp": 長さ N のトークン番号, "tgt": 長さ N の正解, "posi": 各行が使う P の行番号,
            "mask": N×N の bool（True = 行 i は行 j を見てはいけない）,
            "batch": {"inp": B×T, "tgt": B×T}（バッチ形式で書きたい人向けの同じデータ）}
    c に必ず入れるもの : "loss"（float、N 行の平均）, "logits"（N×V）
    入れれば照合されるもの（名前は README §1・§3 の記号と同じ）:
        "X0", "ln1"（{"mu", "var", "sigma"}、各 N×1）, "U1", "Q", "K", "V",
        "S"（ヘッドごとの N×N のリスト。マスク部は −∞）, "A"（同）, "O"（ヘッドごとの N×2 のリスト）, "Ocat", "Attn",
        "X1", "ln2", "U2", "Z", "R", "F", "X2", "probs", "nll"（N 行の −log p[正解]）
    backward が必要とするもの（task など）も c に入れておいてよい。

backward(p, c) -> g または (g, d)
    g : p と同じキー・同じ形の勾配 ∂L/∂θ（9個すべて必須）
    d : 中間変数の勾配（任意）。"logits", "X2", "F", "R", "Z", "U2", "X1_ln2"（LN2 経由の分）, "X1", "Attn", "Ocat",
        "A"（ヘッドごとのリスト）, "S"（同）, "Q", "K", "V", "U1", "X0_ln1"（LN1 経由の分）, "X0"

sgd(p, g, lr) -> p'（任意。無ければ k3_check.py が p − lr·g を使う）

Part 2 の分析（任意。Part 2 のセクション C・F をコードで確かめたい人向け。行は 0..2 = 猫文、3..5 = 犬文）
    ablate_t2(p, c)          -> 2×V   5.4：t2 の Attention 出力を0にしたときの 猫文 t2・犬文 t2 の logits
    head_contributions(p, c) -> H×4   5.5：ヘッドごとの [パターン項, 値項, 直接効果, パッチングの総効果]
    qk_circuit(p, h)         -> 4×4   8.1：Wq_h Wk_hᵀ
    ov_circuit(p, h)         -> 4×V   8.1：Wv_h Wo_h Wout

バッチ形式（B×T×d）で書いた人へ：k3_check.py は (B, T, …) の配列を N = B·T 行に、(B, T, T) の注意の配列を
ブロック対角の N×N に並べ直してから照合するので、そのまま返してかまいません。

---------------------------------------------------------------- 仕様の落とし穴（README §2 より）
- LN は γ=1, β=0, ε=0、**母分散**（÷4）。最終 LN は無い。Pre-LN（X1 = X0 + Attn(LN(X0))）
- ヘッド h は Q, K, V の**列 2h, 2h+1**。スコアは ÷√2、マスクは −∞
- 損失は N 行の**平均**。行ベクトル規約（X @ W）
- 埋め込みの勾配：同じトークン・同じ位置が複数の行で使われたら**足し込む**（Part 2 では P の各行が2回使われる）
"""
import numpy as np


def forward(p, task):
    # TODO: 2.1 → 5.3 を順に。まず Part 1（N = 3、ふつうの因果マスク）で L = 1.3949 を出す
    raise NotImplementedError("forward をまだ書いていません")


def backward(p, c):
    # TODO: 6.1 → 6.9 を順に（derivations.md 問1〜8）。g と（できれば）d を返す
    raise NotImplementedError("backward をまだ書いていません")
