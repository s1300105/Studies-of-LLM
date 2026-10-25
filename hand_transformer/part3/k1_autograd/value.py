"""K1 スカラー自動微分 — あなたが埋める骨組み。

ゴール：Part 1 のステップ6で手で回した逆伝播を、「1つ1つの数（スカラー）」の単位で機械にやらせる。
完成したら K2（../k2_verify_part1.py --mine）で、Part 1 の全180個の勾配を手計算の正解と照合します。

使い方（完成後）:
    from value import Value
    x = Value(2.0); y = Value(-3.0)
    z = (x * y + x.exp()).relu()
    z.backward()
    print(x.grad, y.grad)            # ∂z/∂x, ∂z/∂y

テスト:
    python3 test_value.py            # このファイルをテストする。目標：最大誤差 ≤ 1e-8
    python3 test_value.py --ref      # 参考解答（../solutions/value.py）をテストする（テスト自体の確認用）

やること（TODO は5か所 + backward）:
    1. __mul__   ∂(ab)/∂a = b, ∂(ab)/∂b = a
    2. __pow__   定数乗 x**k（k は int / float）。∂(x^k)/∂x = k x^(k-1)
    3. exp       ∂e^x/∂x = e^x
    4. log       ∂ln x/∂x = 1/x
    5. relu      ∂relu(x)/∂x = [x > 0]（x = 0 では 0 とする。Part 1 と同じ）
    6. backward  トポロジカル順に並べ、出力 → 入力の順に _backward を呼ぶ

ルール（ここが K1 の学びどころ）:
    - _backward の中では、子（入力）の grad に **+=** で足し込む。= で上書きしない。
      同じ値が2か所で使われたら勾配は足される（Part 0 D2「分岐」、derivations.md 問2）。
      テストの x * x（同じオブジェクトを2回使う）で、= だと必ず間違えます。
    - 局所微分は「順伝播の値（self.data, out.data）」と「上から来た勾配 out.grad」の積で書く。
    - __add__ は書き方の見本として実装済み。残りもこれと同じ形で書けます。
    - __neg__, __sub__, __truediv__ などは基本演算の組み合わせなので実装済み（局所微分を書く必要がない）。
"""
import math


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)      # 順伝播の値
        self.grad = 0.0              # ∂(出力)/∂(この値)。backward() で埋まる
        self._prev = tuple(_children)  # この値を作った入力（グラフの辺）
        self._op = _op               # 作った演算の名前（表示・数え上げ用）
        self._backward = lambda: None  # 自分の grad を、入力の grad に配る関数（葉では何もしない）

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------ 見本（実装済み）

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad        # ∂(a+b)/∂a = 1 → 上からの勾配をそのまま足す
            other.grad += out.grad       # ∂(a+b)/∂b = 1
        out._backward = _backward
        return out

    # ------------------------------------------------------------ TODO 1〜5

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # TODO 1: self.grad と other.grad に、それぞれの局所微分 × out.grad を足し込む
            raise NotImplementedError("TODO 1: __mul__ の _backward")
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "指数は定数（int / float）だけ"
        # TODO 2: out = Value(順伝播の値, (self,), f"**{k:g}") を作り、_backward を書く
        raise NotImplementedError("TODO 2: __pow__")

    def exp(self):
        # TODO 3: 順伝播の値 math.exp(self.data) を一度だけ計算し、_backward でも使い回す
        raise NotImplementedError("TODO 3: exp")

    def log(self):
        # TODO 4
        raise NotImplementedError("TODO 4: log")

    def relu(self):
        # TODO 5
        raise NotImplementedError("TODO 5: relu")

    # ------------------------------------------------------------ TODO 6：逆伝播

    def backward(self):
        """self（ふつうは損失 L）から、グラフ中の全ノードの grad を計算する。

        手順:
          (a) self から _prev をたどって到達できる全ノードを、「入力側が先、出力側が後」の順（トポロジカル順）に並べる。
              同じノードを2回入れないこと（id(v) の集合で訪問済みを管理する）。
          (b) self.grad = 1.0（∂L/∂L = 1）
          (c) 並べたノードを逆順（出力側から）にたどり、各ノードの _backward() を呼ぶ。
              逆順にするのは、v._backward() を呼ぶ時点で v.grad が「v を使う全ノードからの寄与の合計」になっているため。

        ヒント：再帰（def build(v): for c in v._prev: build(c); topo.append(v)）で書くのが一番短い。
        K2 のグラフ（Part 1 の順伝播全体で 1,448 ノード、深さ 79）は再帰の上限（既定 1000）より浅いので再帰でも動くが、
        test_value.py の「深い鎖」テスト（任意）は再帰だと RecursionError になる。余裕があればスタックで書き直す。
        """
        raise NotImplementedError("TODO 6: backward")

    # ------------------------------------------------------------ 組み合わせで作れる演算（実装済み）

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):           # 3 + x
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):           # 3 - x
        return other + (-self)

    def __rmul__(self, other):           # 3 * x
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):       # 3 / x
        return other * self ** -1
