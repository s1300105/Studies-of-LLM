# 導出ノート：逆伝播で使う微分を自分で導く

ステップ6（逆伝播）に入る**前に**、このノートの問いを紙の上で解いてください（目安 2.5 時間）。
答えは各問の「答え」を開くと見られますが、まず自分で導出してから確認するのがおすすめです。

記号：$L$ は損失（スカラー）。任意の変数 $X$ について $dX := \partial L / \partial X$（$X$ と同じ形の行列）と書きます。
行ベクトル規約（$Y = XW$）を使います。

---

## 問1 行列積 $Y = XW$ の逆伝播

$X$ は $n \times k$、$W$ は $k \times m$ とします。$dY$ が分かっているとき、$dX$ と $dW$ を求めてください。

ヒント：成分で書くと $Y_{ij} = \sum_k X_{ik} W_{kj}$。$W_{kj}$ はどの $Y$ の成分に効いているか？

<details><summary>答え</summary>

$$\frac{\partial L}{\partial W_{kj}} = \sum_i \frac{\partial L}{\partial Y_{ij}} X_{ik} \quad\Rightarrow\quad dW = X^\top dY,\qquad dX = dY\, W^\top$$

**形合わせで覚える**：$dW$ は $W$ と同じ $k\times m$ にならなければならない。$X^\top$（$k\times n$）と $dY$（$n\times m$）を掛ければ $k\times m$ になる。
$dX$ も同様に $n\times k$ = $(n\times m)(m\times k)$。

$dW = X^\top dY$ は「位置ごとの外積 $x_t^\top\, dy_t$ を全位置で足したもの」でもあります。
**同じ重みを3つの位置で共有しているから、勾配は3位置分の和になる**、というのが Transformer の重み共有の意味です。
</details>

---

## 問2 分岐と残差接続

ある変数が2か所で使われるとき（例：$X_1 = X_0 + \mathrm{Attn}(\mathrm{LN}(X_0))$ の $X_0$）、$dX_0$ はどうなりますか？

<details><summary>答え</summary>

多変数の連鎖律により、**それぞれの経路から来た勾配を足す**：

$$dX_0 = \underbrace{dX_1}_{\text{残差（恒等写像）経由}} + \underbrace{\mathrm{LN}^\top\text{-backward}(\cdots)}_{\text{Attention 経由}}$$

恒等写像の微分は単位行列なので、残差経路では $dX_1$ が**何も掛けられずにそのまま**届きます。
このモデルは最終LNが無いので、$dX_2$（出力層の直前の勾配）が $dX_1$、$dX_0$ に**そのまま含まれる**ことを手計算で確認してください（6.3, 6.8）。
</details>

---

## 問3 softmax + 交差エントロピー

位置 $t$ のロジット $z \in \mathbb{R}^5$、$p = \mathrm{softmax}(z)$、正解 $y$ に対して $L_t = -\log p_y$。$\partial L_t/\partial z_j$ を求めてください。
さらに損失が3位置の平均 $L = \frac13\sum_t L_t$ のとき $d\,\mathrm{logits}$ はどうなりますか？

<details><summary>答え</summary>

$\log p_y = z_y - \log\sum_k e^{z_k}$ なので

$$\frac{\partial L_t}{\partial z_j} = p_j - \mathbb{1}[j = y], \qquad d\,\mathrm{logits} = \frac{1}{3}(P - Y)$$

（$Y$ は正解の one-hot 行列）。「予測確率 − 正解」という驚くほど単純な形。
**検算**：各行の和は $\sum_j p_j - 1 = 0$ になる。
</details>

---

## 問4 softmax のヤコビアン（Attention 内部）

$a = \mathrm{softmax}(s)$ について $\partial a_i / \partial s_j$ を求め、$da$ から $ds$ を計算する式を作ってください。

<details><summary>答え</summary>

$$\frac{\partial a_i}{\partial s_j} = a_i(\delta_{ij} - a_j)$$

$$ds_j = \sum_i da_i\, a_i(\delta_{ij} - a_j) = a_j\Big(da_j - \sum_i a_i\, da_i\Big)$$

行列でまとめると $dS = A \odot \big(dA - \mathrm{rowsum}(dA \odot A)\big)$。

**検算**：各行の和は $\sum_j a_j da_j - (\sum_j a_j)(\sum_i a_i da_i) = 0$。

**観察ポイント**：位置0の行は要素が1つ（$a_{00} = 1$）なので $ds_{00} = 1\cdot(da_{00} - da_{00}) = 0$。
つまり **位置0のクエリ $Q[0]$ には勾配が流れない**（6.6 で $dQ$ の0行目（t0）が0になることを確認）。
</details>

---

## 問5 スケーリングと因果マスク

$S = QK^\top/\sqrt{d_k}$（マスク位置は $-\infty$）。$dS$ から $dQ$、$dK$ を求めてください。マスク位置の勾配はどう扱えばよいですか？

<details><summary>答え</summary>

$$dQ = \frac{1}{\sqrt{d_k}}\, dS\, K, \qquad dK = \frac{1}{\sqrt{d_k}}\, dS^\top Q$$

マスク位置は定数 $-\infty$ を代入しているだけなのでパラメータに依存しない。また $A$ がそこで0なので、問4の式から $dS$ も自動的に0になる。
**未来のトークンへは勾配も流れない**＝マスクは順伝播だけでなく逆伝播でも「未来を見ない」ことを保証している。

$dK = dS^\top Q / \sqrt{d_k}$ の転置に注意：キー $k_j$ の勾配は「$k_j$ を見た全てのクエリ」から集まる（列方向の和）。

**検算**：$\sum_j dK_j = \frac{1}{\sqrt{d_k}}\sum_i \big(\sum_j dS_{ij}\big) Q_i = 0$（dS の行和が0だから）。
全キーに同じベクトルを足してもスコアが行ごとに定数ずれるだけで softmax は変わらない、ということの裏返し。

**順伝播での節約**：位置0の行は要素1つの softmax なので、$Q$ の0行目が何であっても $A$ の0行目は $[1,0,0]$。
つまり**順伝播で $Q$ の0行目は計算しなくてよい**（逆伝播でも勾配0）。
</details>

---

## 問6 Attention の出力 $O = AV$

$dO$ から $dA$ と $dV$ を求めてください（問1の特別な場合です）。

<details><summary>答え</summary>

$$dA = dO\, V^\top \quad(\text{マスク位置は 0 にする}), \qquad dV = A^\top dO$$

$dV$ の行 $j$ は「位置 $j$ の値ベクトルを、どれだけの重みで使ったクエリたちの勾配の加重和」。
</details>

---

## 問7 ヘッドの分割・連結、ReLU、埋め込み

1. $Q$ の列0,1 をヘッド0、列2,3 をヘッド1 として使い、出力を $[O_0 \mid O_1]$ と連結した。逆伝播ではどうする？
2. $R = \mathrm{ReLU}(Z)$ の逆伝播は？
3. $X_0 = E[\text{ids}] + P$ の逆伝播で $dE$、$dP$ は？

<details><summary>答え</summary>

1. 連結の逆は分割、分割の逆は該当列への書き戻し。$dO_{\text{cat}}$ の列0,1 が $dO_0$、列2,3 が $dO_1$。$dQ$ はヘッドごとの $dQ_h$ を列方向に並べる。
2. $dZ = dR \odot \mathbb{1}[Z > 0]$。**一度も発火しなかったユニットは、W1 の列と W2 の行に勾配が全く来ない**（死んだユニット）。
3. $dP = dX_0$。$dE$ は入力に現れたトークンの行にだけ $dX_0$ の行を**足し込む**（同じトークンが2回出たら2回足す）。
   **入力に現れない「鳴く」「。」の行は勾配0**——出力層 Wout の列としては学習されるのに、入力側の埋め込みは変わらない。
</details>

---

## 問8 LayerNorm（$\gamma=1, \beta=0, \varepsilon=0$）の逆伝播 ★山場

1行（$d=4$ 次元）について $\mu = \frac1d\sum_i x_i$、$\sigma^2 = \frac1d\sum_i (x_i-\mu)^2$、$y_i = (x_i-\mu)/\sigma$。
$dy$ から $dx$ を求めてください。

ヒント：(a) $\partial\mu/\partial x_i$、(b) $\partial\sigma/\partial x_i$（$y_i$ を使って書ける）、(c) $\partial y_j/\partial x_i$ の順に。

<details><summary>答え</summary>

(a) $\partial\mu/\partial x_i = 1/d$

(b) $\partial\sigma/\partial x_i = \dfrac{x_i-\mu}{d\,\sigma} = \dfrac{y_i}{d}$（$\sum_j(x_j-\mu) = 0$ を使う）

(c) $\dfrac{\partial y_j}{\partial x_i} = \dfrac{1}{\sigma}\Big(\delta_{ij} - \dfrac1d - \dfrac{y_i y_j}{d}\Big)$

よって

$$dx_i = \sum_j dy_j \frac{\partial y_j}{\partial x_i} = \frac{1}{\sigma}\Big(dy_i - \overline{dy} - y_i\,\overline{dy\odot y}\Big)$$

（$\overline{\,\cdot\,}$ はその行の4成分の平均）。

**検算（必ずやる）**：各行で
- $\sum_i dx_i = 0$（平均を引いているので、全成分を同じだけずらす方向には勾配が無い）
- $\sum_i dx_i\, y_i = 0$（LNは入力のスケールに不変なので、$x-\mu$ 方向にも勾配が無い）

この2つが成り立たなければどこかで計算ミスしています。

**このワークブックの LN1 は出力 $U_1$ が全て ±1 になるよう設計してある**ので、$y_i^2 = 1$ となり式の計算が少し楽です。
</details>

---

## 問9 1ステップでどれだけ損失が下がるか（テイラー展開）

SGD で $\theta' = \theta - \eta\, g$（$g = \nabla_\theta L$）と更新したとき、$L(\theta')$ を1次まで展開してください。

<details><summary>答え</summary>

$$L(\theta - \eta g) \approx L(\theta) - \eta\, g^\top g = L(\theta) - \eta \lVert g\rVert^2$$

勾配の二乗和を全パラメータ（このモデルでは180個）について足せば、**再順伝播をする前に損失の下がり幅を予言できる**（7.2）。
実際の下がり幅（7.4）とのずれは2次以上の項（曲率）によるもの。$\eta \to 0$ ではずれは $\eta^2$ に比例して小さくなるが、$\eta$ が大きいと曲率しだいで予測より多くも少なくも下がりうる。このモデルでは lr = 0.5 だと実際の方が約8%多く下がり、lr = 1 だと約12%少ない。lr を大きくしすぎると損失は逆に**上がる**（lr = 10 で $L' = 2.1463$）。
</details>

---

## 検算チェックリスト（計算中いつでも使える）

| 場所 | 成り立つべき性質 |
|---|---|
| 2.3 / 4.2 LN出力 | 各行の平均 = 0、二乗平均 = 1 |
| 3.3 Attention重み | 各行の和 = 1、マスク位置 = 0、位置0の行は [1, 0, 0] |
| 5.2 確率 | 各行の和 = 1 |
| 6.1 dlogits | 各行の和 = 0 → したがって dWout の**各行の和も 0** |
| 6.3 / 6.8 LN逆伝播 | 「LN2経由」「LN1経由」の表の各行の和 = 0、かつ LN出力との内積 = 0（合流後の dX1, dX0 では成り立たない） |
| 6.3 / 6.8 残差の合流 | dX2、dX1、dX0 の**各行の和はすべて等しい**（LN経由の分の行和が0だから） |
| 6.5 dS | 各行の和 = 0、位置0の行は全て0 |
| 6.6 dQ | 位置0の行 = 0 |
| 6.6 dK | ヘッドごとに**列の和（3位置の和）= 0**（全キーに同じベクトルを足しても Attention は変わらないから） |
| 6.6 dV | ヘッドごとに列の和 = dO の列の和（A の行和が1だから） |
| 6.9 dE | 入力に無いトークン（鳴く、。）の行 = 0。使われた行は dP と完全に一致（各トークンが1回ずつ別の位置に出るため） |
| 7.1 更新 | E と P が同じだけ動くので、X0' = X0 − 2·lr·dX0 |
| 7.2 → 7.4 | 予測 ΔL = −lr‖g‖² は必ず負。実際の ΔL は lr が十分小さければ負で、予測に近い（lr = 1 では予測 −1.4701、実際 −1.2893） |

### 行列積のチェックサム（どの行列積にも使える）

$Y = XW$ の各行の和は $X$ の行と「$W$ の行和ベクトル」$w = W\mathbf{1}$ の内積に等しい：

$$\sum_j Y_{tj} = x_t \cdot w,\qquad w_k = \sum_j W_{kj}$$

計算を始める前に各重み行列の行和 $w$ を求めておけば（1回目の順伝播では重みが0.5刻みなので暗算でできる）、出力1行につき内側の次元の数（4、FFN の2層目 $R W_2$ では8）回の積和で検算できます。
一致しなければその行のどこかに誤りがあります（一致しても、同じ行で誤差が打ち消し合うまれな場合は見逃します）。
逆伝播の $dX = dY\,W^\top$ には $W$ の**列和**を使います。
