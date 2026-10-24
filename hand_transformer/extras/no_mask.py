"""発展課題 S1：因果マスクを外すとどうなるか（カンニングと、生成の破綻）。

学習時は「<s> 猫 が」を一度に入れて3位置の次トークンを同時に当てる（teacher forcing）。
因果マスクが無いと、位置0は「自分が当てるべき答え（位置1の 猫/犬）」をそのまま見られる。
  - 学習損失は、マスクありでは原理的に越えられない下限（2文タスクで 2·ln2/6 ≈ 0.231）を下回る
  - しかし生成時は1語ずつ入力するので、見るはずだった「未来」はまだ存在しない
    → 学習時と生成時で同じ位置の出力が変わり、貪欲生成が崩れる
因果マスクの役割は「生成時と同じ条件で学習させること」。KV キャッシュが使えるのもこのおかげ（kv_cache.py）。

実行:  python3 extras/no_mask.py          （約20秒）
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mini_gpt as mg  # noqa: E402
from general_model import forward, init_params, train  # noqa: E402
from two_sentences import INP, TGT, VOCAB, ids  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
VALID = {("猫", "が", "鳴く"), ("犬", "が", "吠える")}


def generate(p, causal, vocab, ln_eps=1e-5):
    """<s> から貪欲に1語ずつ生成。P の行数（=3）までしか位置が無いので3語で止まる。"""
    seq = [0]
    while len(seq) <= p["P"].shape[0]:
        c = forward(p, [seq], None, causal=causal, ln_eps=ln_eps)
        seq.append(int(c["probs"][0, -1].argmax()))
    return tuple(vocab[i] for i in seq[1:])


def two_sentence_task(seeds):
    print(f"[1] 2文タスク（two_sentences.py と同じデータ、Adam 3000ステップ）。マスクありの損失下限 = {2 * np.log(2) / 6:.4f}")
    print("  seed | マスクあり: 学習損失  生成（<s> から貪欲）    | マスクなし: 学習損失  生成")
    n_ok = {True: 0, False: 0}
    models = {}
    for seed in seeds:
        p0 = init_params(len(VOCAB), 3, seed=seed)
        row = f"  {seed:4d} |"
        for causal in (True, False):
            p, c, _ = train(p0, INP, TGT, steps=3000, causal=causal)
            sent = generate(p, causal, VOCAB)
            n_ok[causal] += sent in VALID
            models[seed, causal] = p
            row += f" {c['loss']:10.4f}  {' '.join(sent):<12}{'○' if sent in VALID else '×'}       |"
        print(row)
    print(f"  正しい文を生成できた seed の数: マスクあり {n_ok[True]}/{len(seeds)}, マスクなし {n_ok[False]}/{len(seeds)}")
    return models


def inspect(p_mask, p_free):
    print("\n[2] seed 0 の中身：位置0（<s>）は何を見て、何を予測しているか")
    for name, p, causal in (("マスクあり", p_mask, True), ("マスクなし", p_free, False)):
        full = forward(p, INP, None, causal=causal)          # 学習時と同じ：3語まとめて入力
        alone = forward(p, INP[:, :1], None, causal=causal)  # 生成の1歩目：<s> だけ
        print(f"  {name}")
        for b in range(2):
            att = " / ".join(f"ヘッド{h} {np.round(full['A'][h][b, 0], 3)}" for h in range(2))
            pf, pa = full["probs"][b, 0], alone["probs"][b, 0]
            print(f"    文{b}: 位置0 の注意（列 = <s>, 猫/犬, が）{att}")
            print(f"          位置0 の P(猫), P(犬): 3語入力 ({pf[ids['猫']]:.3f}, {pf[ids['犬']]:.3f})  "
                  f"<s>だけ入力 ({pa[ids['猫']]:.3f}, {pa[ids['犬']]:.3f})、"
                  f"<s>だけ入力での最有力候補「{VOCAB[pa.argmax()]}」({pa.max():.3f})")
        diff = np.abs(full["probs"][:, 0] - alone["probs"][:, 0]).max()
        print(f"    位置0 の確率分布が「後ろに語を足したか」で変わる量（最大差）: {diff:.1e}")


def main_task():
    print("\n[3] 本編の1文（params.json、SGD 100ステップ、LN の ε=0）でマスクを外すと")
    p0, _ = mg.load_params()
    for lr in (0.25, 0.5, 1.0):
        for causal in (True, False):
            p, c, hist = train(p0, [mg.INP], [mg.TGT], steps=100, lr=lr, optimizer="sgd", causal=causal, ln_eps=0)
            loss = f"{c['loss']:.4f}" if np.isfinite(c["loss"]) else "発散"
            sent = " ".join(generate(p, causal, mg.VOCAB, ln_eps=0)) if np.isfinite(c["loss"]) else "-"
            a0 = " ".join(f"{np.round(c['A'][h][0, 0], 3)}" for h in range(2)) if np.isfinite(c["loss"]) else "-"
            print(f"  lr={lr:<4g} {'マスクあり' if causal else 'マスクなし'}  初期損失 {hist[0]:.4f}  100ステップ後 {loss:>7}  "
                  f"生成: {sent:<12} 位置0の注意 A0,A1 = {a0}")


def main():
    models = two_sentence_task(range(10))
    inspect(models[0, True], models[0, False])
    main_task()


if __name__ == "__main__":
    main()
