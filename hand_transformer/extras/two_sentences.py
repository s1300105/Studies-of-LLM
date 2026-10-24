"""発展課題：Attention が「本当に必要な」タスクで学習させる。

本編のタスク（<s> 猫 が → 猫 が 鳴く）は「直前の1トークンだけ見れば次が決まる」ので、
実は Attention が無くても（埋め込み + FFN + 出力層だけで）解けてしまう。

ここでは2文を同時に学習させる：
    <s> 猫 が → 猫 が 鳴く
    <s> 犬 が → 犬 が 吠える
位置2の入力はどちらも「が」なので、次が「鳴く」か「吠える」かは
**位置1を振り返らないと決められない**。Attention を外したモデルはここで50%に張り付く。

なお位置0（<s> の次が 猫 か 犬 か）はどちらのモデルでも原理的に当てられないので、
損失の下限は  Attentionあり: 2·ln2/6 ≈ 0.231、Attentionなし: 4·ln2/6 ≈ 0.462。

実行:  python extras/two_sentences.py
"""
import numpy as np

from general_model import grad_check, init_params, train

VOCAB = ["<s>", "猫", "犬", "が", "鳴く", "吠える", "。"]
ids = {w: i for i, w in enumerate(VOCAB)}
enc = lambda s: [ids[w] for w in s.split()]
INP = np.array([enc("<s> 猫 が"), enc("<s> 犬 が")])
TGT = np.array([enc("猫 が 鳴く"), enc("犬 が 吠える")])


def report(name, c):
    print(f"\n[{name}] 最終損失 {c['loss']:.4f}")
    for b in range(2):
        pr = c["probs"][b, 2]
        print(f"  文{b}: 位置2（が）の次  P(鳴く)={pr[ids['鳴く']]:.3f}  P(吠える)={pr[ids['吠える']]:.3f}")


def main():
    p0 = init_params(len(VOCAB), 3, seed=0)
    print(f"逆伝播の数値微分チェック: 最大誤差 {grad_check(p0, INP, TGT):.1e}")
    print(f"損失の理論下限: Attentionあり {2 * np.log(2) / 6:.4f} / Attentionなし {4 * np.log(2) / 6:.4f}")

    _, c_full, _ = train(p0, INP, TGT, steps=3000, use_attn=True)
    report("Attention あり", c_full)
    print("  位置2（が）のクエリが各位置に払う注意（列 = <s>, 猫/犬, が）")
    for h in range(2):
        for b in range(2):
            print(f"    ヘッド{h} 文{b}: {np.round(c_full['A'][h][b, 2], 3)}")

    _, c_none, _ = train(p0, INP, TGT, steps=3000, use_attn=False)
    report("Attention なし", c_none)


if __name__ == "__main__":
    main()
