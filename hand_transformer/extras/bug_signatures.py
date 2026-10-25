"""発展課題 M4 の補助：よくある実装ミスが「どのチェックポイントで最初に」「どう」現れるか。

自分で実装したコード（や手計算）が answers.md と合わないとき、最初にずれたチェックポイントと
L, L' の値をこの表と見比べると、原因の見当がつけやすい。

仕組み：mini_gpt.py の関数のソースを1行だけ書き換えて（= わざとバグを入れて）全チェックポイントを再計算し、
正解と許容差 0.002 + 0.002·|正解| を超えてずれた最初のチェックポイントを表示する。
mini_gpt.py のファイル自体は変更しない（メモリ上で関数を差し替えて、終わったら戻す）。

実行:  python3 extras/bug_signatures.py
"""
import inspect
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as mg  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# (説明, 書き換える関数, 正しいコード, バグ入りのコード)
BUGS = [
    ("LN の分散を ÷3（不偏分散）にした", "layernorm",
     "var = (xc ** 2).mean(-1, keepdims=True)", "var = (xc ** 2).sum(-1, keepdims=True) / 3"),
    ("スコアを √d_k で割り忘れた", "forward",
     'S = c["Q"][:, sl] @ c["K"][:, sl].T / np.sqrt(DK)', 'S = c["Q"][:, sl] @ c["K"][:, sl].T'),
    ("Attention の softmax を列方向にとった", "forward",
     'c["A"].append(softmax(S))', 'c["A"].append(softmax(S.T).T)'),
    ("因果マスクを付け忘れた", "forward",
     "S = np.where(mask, -np.inf, S)", "S = S"),
    ("損失を平均でなく和にした（dlogits の ÷3 忘れ）", "backward",
     'd["logits"] = (c["probs"] - Y) / n', 'd["logits"] = (c["probs"] - Y)'),
    ("残差の分岐で勾配を足し忘れた（dX1 = LN2経由だけ）", "backward",
     'd["X1"] = d["X2"] + d["X1_ln2"]', 'd["X1"] = d["X1_ln2"]'),
    ("LN の逆伝播で平均の2項を忘れた（dx = dy/σ）", "layernorm_backward",
     "return (dy - dy.mean(-1, keepdims=True) - y * (dy * y).mean(-1, keepdims=True)) / sigma", "return dy / sigma"),
    ("dV = A dO と転置を忘れた", "backward",
     'd["V"][:, sl] = A.T @ dO', 'd["V"][:, sl] = A @ dO'),
    ("dK = dS Q と転置を忘れた", "backward",
     'd["K"][:, sl] = dS.T @ c["Q"][:, sl] / np.sqrt(DK)', 'd["K"][:, sl] = dS @ c["Q"][:, sl] / np.sqrt(DK)'),
    ("dQ, dK の 1/√d_k を忘れた", "backward",
     'd["Q"][:, sl] = dS @ c["K"][:, sl] / np.sqrt(DK)\n        d["K"][:, sl] = dS.T @ c["Q"][:, sl] / np.sqrt(DK)',
     'd["Q"][:, sl] = dS @ c["K"][:, sl]\n        d["K"][:, sl] = dS.T @ c["Q"][:, sl]'),
    ("SGD の符号を逆にした（θ + lr·g）", "sgd",
     "p[k] - lr * g[k]", "p[k] + lr * g[k]"),
]


def mutated(name, old, new):
    """mini_gpt の関数 name のソースの old を new に置き換えた版を、mini_gpt の名前空間に定義する。"""
    src = inspect.getsource(getattr(mg, name))
    assert src.count(old) == 1, f"{name} に置き換え対象が見つからない: {old}"
    saved = getattr(mg, name)
    exec(src.replace(old, new), mg.__dict__)
    return saved


def first_deviation(ref, got):
    for (cid, _, items_r), (_, _, items_g) in zip(ref, got):
        for (label, Mr, *_), (_, Mg, *_) in zip(items_r, items_g):
            Mr, Mg = np.atleast_2d(Mr), np.atleast_2d(Mg)
            both_inf = np.isneginf(Mr) & np.isneginf(Mg)
            err = np.where(both_inf, 0.0, np.abs(np.nan_to_num(Mg - Mr, nan=np.inf)))
            if (err > 0.002 + 0.002 * np.abs(np.where(both_inf, 0, Mr))).any():
                return cid, label, float(err.max())
    return None


def summary(cp):
    get = {cid: items for cid, _, items in cp}
    L = float(get["5.3"][1][1][0, 0])
    L1 = float(get["7.4"][0][1][0, 1])
    return L, L1


def main():
    p, lr = mg.load_params()
    ref = mg.checkpoints(p, lr)
    L, L1 = summary(ref)
    print(f"正解: L = {L:.4f}, L' = {L1:.4f}\n")
    print(f"  {'バグ':<44} {'最初にずれるチェックポイント':<40} {'最大のずれ':>8}   L        L'")
    for desc, name, old, new in BUGS:
        saved = mutated(name, old, new)
        try:
            got = mg.checkpoints(p, lr)
        finally:
            mg.__dict__[name] = saved
        dev = first_deviation(ref, got)
        Lb, L1b = summary(got)
        where = f"{dev[0]} {dev[1]}" if dev else "（ずれない）"
        fmt = lambda x: f"{x:.4f}" if np.isfinite(x) else "nan"
        print(f"  {desc:<44} {where[:40]:<40} {dev[2] if dev else 0:8.4f}   {fmt(Lb)}   {fmt(L1b)}")


if __name__ == "__main__":
    main()
