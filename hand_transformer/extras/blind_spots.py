"""「手計算では学べないこと」を数字で確かめる小さな実験集。

  [1] 汎化：1文を丸暗記したモデルは、見たことのない入力にも自信満々で答える
  [2] 初期化のスケール：本編の 0.5 刻みの重みと、実際の LLM の標準偏差 0.02 の初期化の違い
  [3] Adam の1ステップ目：勾配の大きさによらず、ほぼ全パラメータが lr だけ動く（warmup が要る理由の1つ）
  [4] 数値の安定性：LN の ε、softmax の最大値引き、−log p と logsumexp
  [5] トークン化：日本語1文字は UTF-8 で何バイトか（バイト単位 BPE の出発点）
  [6] 長い文脈のコスト：スコアの個数と KV キャッシュのメモリ

実行:  python3 extras/blind_spots.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mini_gpt as mg  # noqa: E402
from general_model import backward, forward, init_params  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)


def generalization():
    print("[1] 本編のモデルを lr=1 で100ステップ学習した後、いろいろな入力を与える（各位置で最も確率の高い次トークン）")
    p, lr = mg.load_params()
    for _ in range(100):
        g, _ = mg.backward(p, mg.forward(p))
        p = mg.sgd(p, g, lr)
    for sent in ("<s> 猫 が", "<s> が 猫", "<s> 鳴く 。", "<s> 。 。"):
        ids = [mg.VOCAB.index(w) for w in sent.split()]
        pr = forward(p, [ids], None, ln_eps=0)["probs"][0]
        nxt = " ".join(f"{mg.VOCAB[i]}({pr[t, i]:.2f})" for t, i in enumerate(pr.argmax(-1)))
        print(f"  入力 {sent:<10} → {nxt}" + ("   ← 学習した文" if sent == "<s> 猫 が" else ""))


def init_scale():
    print("\n[2] 初期化のスケールと、初期状態の Attention・損失（本編のタスク、標準偏差 s の正規乱数で初期化、seed 10個の平均）")
    p, _ = mg.load_params()
    c = mg.forward(p)
    print(f"  本編の重み（0.5 刻み）: 初期損失 {c['loss']:.4f}  位置2の注意の最大値 "
          f"ヘッド0 {c['A'][0][2].max():.4f} / ヘッド1 {c['A'][1][2].max():.4f}（一様なら 0.3333）")
    for s in (0.02, 0.5):
        loss, amax = [], []
        for seed in range(10):
            q = init_params(5, 3, scale=s, seed=seed)
            cc = forward(q, [mg.INP], [mg.TGT], ln_eps=0)
            loss.append(cc["loss"])
            amax.append(max(cc["A"][h][0, 2].max() for h in range(2)))
        print(f"  s = {s:<4g}: 初期損失 {np.mean(loss):.4f}（ln5 = {np.log(5):.4f}）  位置2の注意の最大値 {np.mean(amax):.4f}")


def adam_first_step():
    print("\n[3] Adam の1ステップ目（β1=0.9, β2=0.999, ε=1e-8、本編の重み）")
    p, _ = mg.load_params()
    g = backward(p, forward(p, [mg.INP], [mg.TGT], ln_eps=0))
    allg = np.concatenate([g[k].ravel() for k in mg.PARAMS])
    m, v = 0.1 * allg, 0.001 * allg ** 2
    step = (m / 0.1) / (np.sqrt(v / 0.001) + 1e-8)          # バイアス補正後 = g / (|g| + ε)
    nz = np.abs(allg) > 1e-10
    print(f"  |勾配| > 1e-10 の {nz.sum()} 個: |勾配| は {np.abs(allg[nz]).min():.1e}〜{np.abs(allg[nz]).max():.3f}、"
          f"更新量 |Δθ|/lr は {np.abs(step[nz]).min():.4f}〜{np.abs(step[nz]).max():.4f}")
    print("  → SGD なら勾配に比例して動くが、Adam の1歩目は全員がほぼ lr ずつ動く（符号だけを見た更新に近い）")


def stability():
    print("\n[4] 数値の安定性")
    x = np.array([[0.5, 0.5, 0.5, 0.5]])
    print(f"  全成分が同じ行の LN: ε=0 → {mg.layernorm(x)[0][0]}（0/0）,  ε=1e-5 → "
          f"{np.round(((x - x.mean()) / np.sqrt(x.var() + 1e-5))[0], 4)}")
    z = np.array([1000.0, 999.0, 0.0])
    naive = np.exp(z) / np.exp(z).sum()
    print(f"  softmax([1000, 999, 0]): そのまま → {naive},  最大値を引く → {np.round(mg.softmax(z), 4)}")
    z = np.array([0.0, 800.0])
    p_ = mg.softmax(z)
    lse = z.max() + np.log(np.exp(z - z.max()).sum())
    print(f"  正解の logit が 0、他が 800 のとき nll: −log(softmax) → {-np.log(p_[0])},  logsumexp − logit → {lse - z[0]:.1f}")


def tokenization():
    print("\n[5] トークン化")
    for s in ("猫が鳴く。", "cat"):
        print(f"  「{s}」: {len(s)} 文字、UTF-8 で {len(s.encode('utf-8'))} バイト")


def context_cost():
    print("\n[6] 長い文脈のコスト（1ヘッド・1層あたりのスコア q·k の個数、因果マスクで下三角だけ）")
    for T in (3, 1024, 8192, 131072):
        print(f"  T = {T:>7,}: スコア {T * (T + 1) // 2:>17,} 個")
    L_, d, T = 12, 768, 1024
    for T in (1024, 131072):
        print(f"  KV キャッシュ（12層・d_model 768、K と V、16bit）T = {T:>7,}: {2 * L_ * T * d * 2 / 2 ** 20:,.0f} MiB / 1系列")


def main():
    generalization()
    init_scale()
    adam_first_step()
    stability()
    tokenization()
    context_cost()


if __name__ == "__main__":
    main()
