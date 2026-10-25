# K6 nanoGPT を日本語の文字単位で学習し、Part 1 と対応づける（6時間）

Andrej Karpathy の [nanoGPT](https://github.com/karpathy/nanoGPT) は、GPT-2 と同じ形の Transformer を `model.py` 1ファイル（約330行）で書いたものです。
この課題では、夏目漱石『吾輩は猫である』（青空文庫）を**1文字 = 1トークン**で CPU 学習させ、

1. `model.py` の各行が Part 1 のどのチェックポイント（2.1〜5.3）にあたるかを対応づけ、
2. 学習したモデルを Part 1 の記号（E, P, Wq, …, Wout）に読み替えて **NumPy で順伝播し、PyTorch の logits と一致させ**、
3. 部品を1つずつ外す比較実験（位置埋め込みなし・因果マスクなし・ヘッド1本と4本・最終 LayerNorm なし）をします。

> **この環境で実際に確かめた範囲**：下の手順はすべてこのリポジトリの作成時に実行し、数値はその出力です
> （nanoGPT コミット `3adf61e`（2025-11-12）、PyTorch 2.14.1 を CPU で、Python 3.13.17、NumPy 2.5.4、4コアの CPU）。
> ただし**青空文庫のサイト（www.aozora.gr.jp）にはこの環境から接続できなかった**ので、同じファイルを
> 青空文庫のテキストを集めたミラー（[aozorahack/aozorabunko_text](https://github.com/aozorahack/aozorabunko_text)、GitHub）から取得しました。
> 公式サイトの zip（§1 の手順2）からは取得を確かめていません。どちらでも中身は同じ青空文庫形式（Shift_JIS）のテキストです。

補助スクリプトは [`k6_nanogpt/`](k6_nanogpt/) にあります。

| ファイル | 中身 |
|---|---|
| [`prepare_aozora.py`](k6_nanogpt/prepare_aozora.py) | 青空文庫のテキストからルビ・注記を除き、nanoGPT の `train.bin` / `val.bin` / `meta.pkl` を作る |
| [`patch_nanogpt.py`](k6_nanogpt/patch_nanogpt.py) | `model.py` と `train.py` に比較実験用のスイッチ（`use_wpe`, `causal`, `use_ln_f`, `seed`）を足す。既定値では元と同じ計算 |
| [`run_experiments.sh`](k6_nanogpt/run_experiments.sh) | 比較実験7本とサンプル生成をまとめて回す |
| [`summarize_logs.py`](k6_nanogpt/summarize_logs.py) | ログから train / val 損失の表を作る |
| [`numpy_forward.py`](k6_nanogpt/numpy_forward.py) | チェックポイントを Part 1 の記号に読み替えて NumPy で順伝播し、PyTorch と照合する |

## 0. 時間の使い方（6時間）

| # | 内容 | 目安 |
|---|---|---:|
| 1 | 準備：nanoGPT、PyTorch、パッチ、青空文庫のテキスト、データ作り（§1） | 40分 |
| 2 | `model.py` を読み、対応表（§2）を自分で埋めてから答えと比べる。§2.3 の問い | 1時間20分 |
| 3 | 基準の学習とサンプル生成（§3） | 30分 |
| 4 | `numpy_forward.py` で対応表を数値で確かめる。余力があれば自分で書く（§4） | 1時間 |
| 5 | 比較実験：予想を書く → 7本回す → 記録 → 考察（§5） | 2時間 |
| 6 | まとめ：Part 1 で省いたものが実物のどこにあったかを1枚に書く | 30分 |

---

## 1. 準備

以下、このリポジトリを `~/Studies-of-LLM` に置いた場合のパスで書きます。

```bash
K6=~/Studies-of-LLM/hand_transformer/part3/k6_nanogpt

# 1) nanoGPT と PyTorch（GPU は不要）
git clone https://github.com/karpathy/nanoGPT.git ~/nanoGPT
cd ~/nanoGPT
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy tiktoken          # sample.py が tiktoken を import するので入れておく
python3 $K6/patch_nanogpt.py .            # model.py と train.py にスイッチを足す（元は *.orig に残る）

# 2) 青空文庫のテキスト（どちらか一方）
#   a. 公式：図書カード https://www.aozora.gr.jp/cards/000148/card789.html の
#      「ファイルのダウンロード」から「テキストファイル(ルビあり)」（789_ruby_5639.zip）を保存する
#   b. ミラー（この環境で取得を確かめた方法）：
curl -L -o 789_ruby_5639.txt \
  https://raw.githubusercontent.com/aozorahack/aozorabunko_text/master/cards/000148/files/789_ruby_5639/789_ruby_5639.txt

# 3) nanoGPT の文字単位データにする（zip でも txt でもよい）
python3 $K6/prepare_aozora.py --out data/aozora_char 789_ruby_5639.txt
```

- 夏目漱石の作品は著作権の保護期間が終わっています。利用は青空文庫の「[青空文庫収録ファイルの取り扱い規準](https://www.aozora.gr.jp/guide/kijyunn.html)」に従ってください。
- `--compile=False` を付けるので PyTorch 2.0 の `torch.compile` は使いません。Apple Silicon の Mac なら、nanoGPT の README のとおり `--device=mps` で GPU も使えます（ここでは確かめていません）。
- `patch_nanogpt.py` は書き換える場所の文字列が**ちょうど1回**見つかることを確かめてから書き換えます。nanoGPT の版が違って見つからなければ、何も変えずに止まります。

<details><summary>prepare_aozora.py の出力（このリポジトリの作成時）</summary>

```
789_ruby_5639.txt: cp932、321,105 文字
語彙（異なる文字の数）: 3,044
多い文字 20 個: 'の':12476 'い':10299 'る':8724 'て':8601 'な':8571 'と':8113 'に':7616 '。':7486 'か':7303 'し':6947 ...
1回しか出ない文字: 533 種類
train 288,994 トークン、val 32,111 トークン（val にだけ出る文字 77 種類）
```

先頭は `一\n\n　吾輩は猫である。名前はまだ無い。\n　どこで生れたかとんと見当がつかぬ。…`（ルビ《わがはい》《けんとう》や ［＃８字下げ］ などの注記が消えている）。
ファイル全体は 377,325 文字（改行は CR+LF）で、冒頭の記号の説明と末尾の「底本：」以降を除き、改行を LF にそろえて 321,105 文字になりました。
語彙 3,044 は、Part 1 の語彙 5 の約600倍、Shakespeare（nanoGPT の例、65 文字）の約47倍です。
</details>

---

## 2. `model.py` を Part 1 と対応づける

`model.py` を上から読み、各部品が Part 1 のどこにあたるかを、**下の表を見る前に**自分で書き出してください。
nanoGPT は行ベクトル規約（`x @ W`）ではなく PyTorch の `nn.Linear`（`y = x·Wᵀ + b`、`weight` の形は（出力, 入力））を使うので、
**Part 1 の W は `weight` の転置**です。

### 2.1 対応表

形は §3 の設定（語彙 3,044、`n_embd` C = 128、`n_head` 4、ヘッドの次元 hs = 32、`n_layer` 4、`block_size` 64、`bias=False`）のときの値です。`h[l]` は `transformer.h[l]`（l 番目のブロック）の略です。

| nanoGPT（`model.py`） | 形 | Part 1 | チェックポイント | Part 1 との違い |
|---|---|---|---|---|
| `transformer.wte`（`nn.Embedding`） | 3044 × 128 | E | 2.1 | なし（表引き） |
| `transformer.wpe`（`nn.Embedding`） | 64 × 128 | P | 2.1 | なし（学習型の位置埋め込み） |
| `tok_emb + pos_emb` → `transformer.drop` | | $X_0 = E[\text{ids}] + P$ | 2.1 | dropout（§3 では 0） |
| `h[l].ln_1`（`LayerNorm`） | 128 | LN1 | 2.2–2.3 | γ（`weight`）を学習する。β（`bias`）は `bias=True` のときだけ。ε = 1e-5（Part 1 は γ=1, β=0 固定、ε = 0） |
| `h[l].attn.c_attn`（`nn.Linear(C, 3C)`） | 384 × 128 | $[W_q \mid W_k \mid W_v]^\top$ | 3.1 | 3つを1つの行列で同時に計算し、`split(n_embd, dim=2)` で q, k, v に分ける。$W_q$ = `weight[:C].T` |
| `q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)` | | ヘッドの分割（ヘッド h = 列 h·hs〜(h+1)·hs−1） | 3.1 | 全ヘッドを1つのテンソルの次元に並べて同時に計算する |
| `(q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))` | | $S_h = Q_h K_h^\top/\sqrt{d_k}$ | 3.2 | 同じ（$d_k$ = hs = 32） |
| `att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))` | | 因果マスク | 3.2 | 同じ。PyTorch 2.0 以上では下の `scaled_dot_product_attention` の中（`is_causal=True`） |
| `F.softmax(att, dim=-1)` → `attn_dropout` → `att @ v` | | $A_h$、$O_h = A_h V_h$ | 3.3–3.4 | dropout |
| `torch.nn.functional.scaled_dot_product_attention(q, k, v, …, is_causal=True)` | | 3.2〜3.4 をまとめたもの | 3.2–3.4 | PyTorch 2.0 以上ではこちらが使われる（`self.flash`）。中身は上の4行と同じ計算 |
| `y.transpose(1, 2).contiguous().view(B, T, C)` | | $O_\text{cat} = [O_0 \mid O_1 \mid \dots]$ | 3.4 | |
| `h[l].attn.c_proj` → `resid_dropout` | 128 × 128 | $W_o$ | 3.5 | dropout |
| `x = x + self.attn(self.ln_1(x))` | | $X_1 = X_0 + \text{Attn}$ | 4.1 | なし（同じ Pre-LN） |
| `h[l].ln_2` | 128 | LN2 | 4.2 | LN1 と同じ |
| `h[l].mlp.c_fc` → `nn.GELU()` | 512 × 128 | $W_1$、ReLU | 4.3 | 幅は 4C。ReLU ではなく GELU（`nn.GELU()` の既定 = erf を使う厳密版） |
| `h[l].mlp.c_proj` → `dropout` | 128 × 512 | $W_2$ | 4.4 | dropout |
| `x = x + self.mlp(self.ln_2(x))` | | $X_2 = X_1 + F$ | 4.5 | なし |
| `for block in self.transformer.h` | | （Part 1 は1層） | — | ブロックを `n_layer` 回くり返す |
| `transformer.ln_f` | 128 | **なし** | — | Part 1 は省いた（logits を残差の各成分の和に分解するため。README §2） |
| `lm_head`（`nn.Linear(C, vocab, bias=False)`） | 3044 × 128 | $W_\text{out}$ | 5.1 | **`wte` と同じテンソル**（`self.transformer.wte.weight = self.lm_head.weight`）なので $W_\text{out} = E^\top$。Part 1 は共有しない（extras C3） |
| `F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)` | | softmax ＋ 損失の平均 | 5.2–5.3 | バッチ内の全系列・全位置の平均 |

Part 1 で省いたもののうち、学習に関わるものは `train.py` と `configure_optimizers` にあります：
AdamW（β₁ = 0.9、β₂ = 0.95。文字単位の設定 `config/train_shakespeare_char.py` では β₂ = 0.99）、重み減衰 0.1（2次元以上のパラメータだけ。バイアスと LayerNorm にはかけない）、
勾配クリップ 1.0、学習率の warmup（100 反復）と cosine 減衰、バッチ（§3 では 12 系列 × 64 文字）。
初期値は `nn.Linear` と `nn.Embedding` の重みが N(0, 0.02²)（バイアスは 0、LayerNorm は γ = 1）で、残差に書き込む射影（名前が `c_proj.weight` で終わるもの）だけ標準偏差を $0.02/\sqrt{2\cdot n_\text{layer}}$ に小さくします。

### 2.2 パラメータ数

§3 の設定で学習を始めると `number of parameters: 1.18M` と表示されます（`get_num_params()` は既定で `wpe` を数えない）。内訳：

| 部品 | 数 |
|---|---:|
| `wte`（= `lm_head`） | 3,044 × 128 = 389,632 |
| 1ブロック：`ln_1` 128 ＋ `c_attn` 49,152 ＋ `attn.c_proj` 16,384 ＋ `ln_2` 128 ＋ `c_fc` 65,536 ＋ `mlp.c_proj` 65,536 | 196,864 |
| 4ブロック | 787,456 |
| `ln_f` | 128 |
| **合計（`wpe` を除く）** | **1,177,216** |

文字単位の日本語では、語彙が大きいので **埋め込みだけで全体の 33%** になります。Part 1（180 個中 E が 20 個、11%）とは割合が違います。
`train.py` の表示 `num decayed parameter tensors: 18, with 1,184,256 parameters` は `wpe`（8,192 個）を含み、LayerNorm の 9 個 × 128 = 1,152 個（`num non-decayed`）を除いた数です。

### 2.3 読みながら答える問い

<details><summary>Q1. Part 1 の「t0 のクエリは学習されない」（6.5–6.6）は nanoGPT でも起きるか</summary>

起きます。位置0 のクエリが見られるキーは自分自身だけなので、softmax の入力が1つになり、重みは常に 1 です。
バッチの各系列の先頭位置でこの勾配は 0 になります（ただし nanoGPT は系列を文章の途中から切り出すので、「先頭」は毎回違う文字です）。
</details>

<details><summary>Q2. `lm_head` と `wte` を共有すると、E の勾配はどうなるか</summary>

E は入力側（表引き）と出力側（$W_\text{out} = E^\top$）の2か所で使われるので、分岐の規則どおり勾配は2つの和になります：
$dE = dE_\text{入力側} + dW_\text{out}^\top$。入力に一度も出てこない文字の行にも、出力側から勾配が届きます（extras C3 `weight_tying.py`）。
</details>

<details><summary>Q3. `targets` を渡さないとき（生成のとき）、`forward` は何を返すか</summary>

`logits = self.lm_head(x[:, [-1], :])`、つまり**最後の位置の logits だけ**です。生成では次の1文字しか使わないので、残りの位置の出力層の計算を省いています。
学習では `targets` を渡すので、全位置の logits と損失が返ります。
</details>

<details><summary>Q4. dropout は何か所にあるか</summary>

4種類：埋め込みの直後（`transformer.drop`）、Attention の重み（`attn_dropout`）、Attention の出力射影の後（`resid_dropout`）、MLP の出力（`mlp.dropout`）。
§3 では `--dropout=0.0` なので全部効きません。
</details>

---

## 3. 基準の学習とサンプル

nanoGPT の README（quick start）にある「CPU で動かす小さな設定」を、データを替え、評価を正確にするため `--eval_iters` を 20 → 200、表示を減らすため `--log_interval` を 1 → 100 にして使います。

```bash
cd ~/nanoGPT
python3 train.py config/train_shakespeare_char.py --dataset=aozora_char --out_dir=out-base \
  --device=cpu --compile=False --eval_iters=200 --log_interval=100 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python3 sample.py --out_dir=out-base --device=cpu --num_samples=1 --max_new_tokens=200 --start=吾輩は
```

CPU では `torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.` などの警告が出ますが、無視してかまいません（CPU では使われない）。

<details><summary>実際の結果</summary>

- 1反復 33 ms、評価を含めて約1分40秒。1反復 768 文字なので、2000 反復で 1,536,000 文字 = train の約 5.3 周分
- 損失：最初 8.0482（ln 3044 = 8.0209：一様に当て推量）→ 2000 反復で **train 3.3821、val 3.6775**
- サンプル（`--start=吾輩は`、先頭部分）：

  > 吾輩は面白いたそのであっていた。これから君の黒い所へ行って見るところに座敷の所の詩が出すのから坊ばしい事だろう。

  「座敷」「主人」「人間」などの語や句読点の置き方は小説らしくなりますが、文としては意味が通りません。120万パラメータ・2分弱の学習ではこの程度です。
</details>

---

## 4. NumPy で同じ logits を出す

§2.1 の対応表が正しければ、学習した重みを Part 1 の記号に読み替えて、Part 1 と同じ順（2.1 → 5.1）に NumPy で計算すると、PyTorch と同じ logits が出るはずです。

```bash
cd ~/nanoGPT
python3 $K6/numpy_forward.py --out_dir out-base --start 吾輩は猫である
```

`numpy_forward.py` は、`c_attn.weight.T` を3等分して $W_q, W_k, W_v$ にする、`weight` を転置して $W_o, W_1, W_2$ にする、$W_\text{out} = E^\top$ にする、
LN に γ, β と ε = 1e-5 を入れる、ReLU の代わりに GELU を使う、最後に `ln_f` を通す、
ブロック（LN1 → Attention → LN2 → MLP）を `n_layer` 回（§3 の設定では 4 回）くり返す、ヘッドを `n_head` 本・hs 次元（§3 では 4 本・32 次元）にする（スコアは ÷√hs、マスクは入力の長さ T の T×T）、の8点だけが Part 1 の `mini_gpt.py` の順伝播と違います
（`mini_gpt.py` の `forward()` は1層で、ヘッド数 H = 2・次元 DK = 2 がモジュールの定数です）。
**まず自分で書いて**（`mini_gpt.py` の `forward()` を出発点にするとよい）、一致しなければこのスクリプトと比べてください。

<details><summary>実際の結果</summary>

```
lm_head.weight と wte.weight は同じテンソルか: True
入力 '吾輩は猫である'（7 文字）で、NumPy と PyTorch の logits の最大差 = 4.87e-06（float32 の丸め程度なら一致）
各位置の次の文字の予測（上位3つと確率）:
               '吾' → '輩' 0.846  'の' 0.011  '等' 0.011
              '吾輩' → 'の' 0.342  'は' 0.314  'も' 0.066
             '吾輩は' → 'こ' 0.042  '一' 0.024  '、' 0.017
            '吾輩は猫' → 'の' 0.304  'な' 0.096  'に' 0.096
           '吾輩は猫で' → 'あ' 0.354  'も' 0.130  'は' 0.088
          '吾輩は猫であ' → 'る' 0.734  'っ' 0.088  'ら' 0.034
         '吾輩は猫である' → '。' 0.642  'か' 0.096  'が' 0.030
```

PyTorch は float32、NumPy 側は float64 で計算しているので、差 10⁻⁶ 程度は丸めの差です。§5 の4つの変種（nowpe、nomask、head1、nolnf）のチェックポイントでも、最大差は 4.90×10⁻⁶〜6.63×10⁻⁶ でした。
「吾輩は」の次（0.042 が最大）だけ予測がばらけているのは、小説の中で「吾輩は」の後に来る文字が多様だからです。
</details>

---

## 5. 比較実験：部品を1つずつ外す

`patch_nanogpt.py` が足したスイッチで、部品を1つずつ外します。**回す前に、train / val 損失とサンプルがどう変わるかを予想して書いてください。**

| 名前 | 変更 | 外れるもの |
|---|---|---|
| `base`、`base_s1`、`base_s2` | なし（`--seed=1`、`--seed=2`） | — （ばらつきの大きさを測る） |
| `nowpe` | `--use_wpe=False` | 位置埋め込み P（2.1） |
| `nomask` | `--causal=False` | 因果マスク（3.2） |
| `head1` | `--n_head=1` | ヘッドの分割（1本 × 128 次元。4本 × 32 次元と同じパラメータ数） |
| `nolnf` | `--use_ln_f=False` | 最終 LayerNorm |

```bash
cd ~/nanoGPT
bash $K6/run_experiments.sh          # 7本の学習と5本のサンプル（4コアの CPU で約12分。1本あたり約1分40秒）
python3 $K6/summarize_logs.py logs   # 表にする
```

### 5.1 記録すること

実験ごとに次の欄を埋めます。**差が意味を持つかは、`base` を seed だけ変えた3本の幅と比べて判断します**（`train.py` の乱数の種は元は 1337 に固定なので、`seed` スイッチを足しました）。

| 実験 | train（最後） | val（最後） | val − train | 1反復の時間 | サンプルの先頭 50 文字 | 気づいたこと |
|---|---:|---:|---:|---:|---|---|
| base（3本の幅） | | | | | | |
| nowpe | | | | | | |
| nomask | | | | | | |
| head1 | | | | | | |
| nolnf | | | | | | |

`nomask` では、`numpy_forward.py --out_dir out-nomask` の「各位置の予測」も記録してください。

<details><summary>実際の結果（このリポジトリの作成時）</summary>

| 実験 | train（最後） | val（最後） | val − train | 1反復 |
|---|---:|---:|---:|---:|
| base（seed 1337） | 3.3821 | 3.6775 | 0.2954 | 33 ms |
| base_s1（seed 1） | 3.3660 | 3.6660 | 0.3000 | 38 ms |
| base_s2（seed 2） | 3.3866 | 3.6985 | 0.3119 | 38 ms |
| **nowpe** | 3.5053 | **3.7878** | 0.2825 | 35 ms |
| **nomask** | **0.0825** | **0.1117** | 0.0292 | 33 ms |
| head1 | 3.3627 | 3.6615 | 0.2988 | 35 ms |
| nolnf | 3.3483 | 3.7070 | 0.3587 | 31 ms |

（2000 反復後、`eval_iters=200` で測った値。val の最小は7本とも最後の 2000 反復目でした。1反復の時間は中央値で、ほかの処理と同時に動かした回もあるので目安です。）

**ばらつき**：base の3本の val は 3.6660〜3.6985（幅 0.0325）。これより小さい差は、この実験では区別できません。

**nowpe（位置埋め込みなし）**：val 3.7878。base の3本のどれより 0.09〜0.12 悪く、ばらつきの幅の約3倍です。
それでも学習は進みました（8.05 → 3.79）。位置埋め込みが無くても、因果マスクがあると「自分より前に何文字あるか」で注意の平均のされ方が変わるので、位置の手がかりがゼロにはなりません
（Haviv et al. 2022 が、位置埋め込みなしの因果言語モデルでも位置の情報が学ばれることを報告しています）。
サンプル：「吾輩は面白いたその一度に取りの男がやって君の大変化すともります。」

**nomask（因果マスクなし）**：train 0.0825、val 0.1117。**損失だけ見ると圧倒的に「良い」**のに、サンプルは

> 吾輩は輩輩輩輩はは輩輩輩輩輩輩輩輩輩輩輩はは輩輩輩はは輩輩輩輩輩輩は輩輩輩…

と壊れました。各位置の予測を見ると理由が分かります（`numpy_forward.py --out_dir out-nomask --start 吾輩は猫である`）：

```
               '吾' → '輩' 0.999
              '吾輩' → 'は' 1.000
             '吾輩は' → '猫' 0.998
            '吾輩は猫' → 'で' 1.000
           '吾輩は猫で' → 'あ' 1.000
          '吾輩は猫であ' → 'る' 1.000
         '吾輩は猫である' → 'は' 0.922
```

マスクが無いので、位置 t は**入力の中の位置 t+1 の文字（＝正解）を見て写すだけ**です。最後の位置だけは見るべき未来が無いので、当て推量になります（ここでは「は」0.922）。
生成では予測に使うのが常に最後の位置なので、写す相手の無い予測ばかりが続き、壊れます。
**val 損失も同じ順伝播で測るので、評価もカンニングしています**。「val 損失が低い」はモデルが良いことの証明にならない、という例です（extras S1 `no_mask.py` と同じ現象）。

**head1（ヘッド1本 × 128 次元）**：val 3.6615。base の3本の幅（3.6660〜3.6985）とほぼ同じで、**この大きさ・この文脈長（64 文字）・2000 反復では差が見えませんでした**。

**nolnf（最終 LN なし）**：val 3.7070。base の最大（3.6985）より 0.0085 悪いだけで、ばらつきの幅の中か、すぐ外です。**この実験だけでは差があるとは言えません**。
train は 3.3483 と7本の中で（nomask を除いて）一番低く、val − train は 0.3587 と一番大きくなりました。

まとめると、この小さな設定で**はっきり差が出たのは「因果マスク」と「位置埋め込み」だけ**でした。
ヘッドの本数と最終 LN の効果を見るには、もっと大きなモデル・長い文脈・長い学習が要ります（たとえば `--block_size=256` や `--max_iters=5000`。ここでは確かめていません）。
</details>

### 5.2 考察の問い

<details><summary>Q5. nowpe のモデルに「吾輩は猫である」と、同じ7文字を並べ替えた「猫は吾輩である」を入れると、最後の位置（どちらも「る」）の予測は変わるか</summary>

変わります（`numpy_forward.py --out_dir out-nowpe --start …`）：

```
'吾輩は猫である' → '。' 0.586  'か' 0.077  'と' 0.047
'猫は吾輩である' → '。' 0.546  'か' 0.076  'と' 0.063
```

層0 だけなら変わりません。最後の位置のクエリはどちらも「る」で、見るキーと値は同じ7文字の集合（並び順が違うだけ）なので、Attention の和は並べ替えても同じです。
しかし層1 以降のキーと値は、各位置の**層0 の出力**で、それは「その位置より前に何があったか」（因果マスクで決まる、並び順に依存する集合）で変わります。
こうして位置埋め込みが無くても、層を重ねると並び順の情報が入ります。base（位置埋め込みあり）では 0.642 と 0.614 でした。
</details>

<details><summary>Q6. nomask の train 損失は、base と比べてどう下がったか</summary>

| step | base | nomask |
|---:|---:|---:|
| 250 | 4.2592 | 4.1182 |
| 500 | 3.9670 | **0.4689** |
| 2000 | 3.3821 | 0.0825 |

step 250 まではほぼ同じで、250〜500 の間に一気に下がりました。「1つ右の位置の文字を写す」ことを一度覚えると、損失はほぼ 0 まで落ちます（§5.1 の各位置の予測が 0.998〜1.000）。
「正解をコピーする」は「次を予測する」よりずっと易しい課題です。K7 の phase change に似て、ある働きができた所で損失が急に下がる例です。
</details>

---

## 参考

- nanoGPT：https://github.com/karpathy/nanoGPT （`model.py`、`train.py`、`config/train_shakespeare_char.py`、README の「quick start」の節にある CPU 用の設定）
- Andrej Karpathy, "Let's build GPT: from scratch, in code, spelled out."（YouTube）
- Radford et al., "Language Models are Unsupervised Multitask Learners", 2019（GPT-2）
- Press, Wolf, "Using the Output Embedding to Improve Language Models", EACL 2017（重み共有）
- Haviv, Ram, Press, Izsak, Levy, "Transformer Language Models without Positional Encodings Still Learn Positional Information", Findings of EMNLP 2022
- 青空文庫：https://www.aozora.gr.jp/ 、aozorahack/aozorabunko_text：https://github.com/aozorahack/aozorabunko_text
