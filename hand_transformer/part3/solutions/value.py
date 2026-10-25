"""K1 の参考解答：スカラー自動微分（リバースモード）。

まず part3/k1_autograd/value.py の TODO を自分で埋めてください。詰まったときだけ、ここを見ます。

    from value import Value
    x = Value(2.0); y = Value(-3.0)
    z = (x * y + x.exp()).relu()      # 順伝播：計算しながらグラフ（どの値が、どの値から、どの演算で作られたか）を記録する
    z.backward()                      # 逆伝播：z.grad = 1 から始め、トポロジカル順の逆に局所微分を掛けて配る
    x.grad, y.grad                    # ∂z/∂x, ∂z/∂y

設計（Part 1 の手計算との対応）:
- 各 Value は「数値 data」「勾配 grad = ∂(出力)/∂(この値)」「親（入力）」「局所微分の配り方 _backward」を持つ。
- _backward は勾配を **+=** で足し込む。同じ値が2か所で使われたら（分岐・残差・重み共有）勾配は足される
  （Part 0 D2・D3、derivations.md 問2）。= で上書きすると、後から来た経路の勾配だけが残る。
- backward() は、出力から見て「自分を使う全ノードの勾配が出そろってから」自分の _backward を呼ぶ。
  その順番がトポロジカル順の逆。Part 1 のステップ6を「下から上へ」解いたのと同じ順番。
- トポロジカルソートは再帰ではなくスタックで書く（深いグラフで RecursionError にならないように）。
"""
import math


class Value:
    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = tuple(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # ------------------------------------------------------------ 基本演算（局所微分を自分で書くもの）

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad                 # ∂(a+b)/∂a = 1
            other.grad += out.grad                # ∂(a+b)/∂b = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad    # ∂(ab)/∂a = b
            other.grad += self.data * out.grad    # ∂(ab)/∂b = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "指数は定数（int / float）だけ"
        out = Value(self.data ** k, (self,), f"**{k:g}")

        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad             # (e^x)' = e^x（順伝播の値を使い回す）
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += out.grad / self.data
        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            self.grad += (self.data > 0) * out.grad   # Part 1 と同じく ReLU'(0) = 0
        out._backward = _backward
        return out

    # ------------------------------------------------------------ 逆伝播

    def backward(self):
        """self（ふつうは損失）について、グラフ中の全ノードの grad を計算する。grad は呼ぶ前に 0 にしておく。"""
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:                              # 帰りがけ順の DFS（再帰を使わない版）
            v, children_done = stack.pop()
            if children_done:
                topo.append(v)                    # 子（入力側）がすべて topo に入ったあとで自分を入れる
                continue
            if id(v) in seen:
                continue
            seen.add(id(v))
            stack.append((v, True))
            for child in v._prev:
                if id(child) not in seen:
                    stack.append((child, False))
        self.grad = 1.0                           # ∂L/∂L = 1
        for v in reversed(topo):                  # 出力側から入力側へ
            v._backward()

    # ------------------------------------------------------------ 組み合わせで作れる演算（局所微分は不要）

    def __neg__(self):
        return self * -1.0

    def __radd__(self, other):                    # 3 + x
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):                    # 3 - x
        return other + (-self)

    def __rmul__(self, other):                    # 3 * x
        return self * other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):                # 3 / x
        return other * self ** -1


def topological_order(root):
    """root から到達できる全ノードを、入力側 → 出力側の順に返す（K2 でノード数を数えるのに使う）。"""
    topo, seen, stack = [], set(), [(root, False)]
    while stack:
        v, done = stack.pop()
        if done:
            topo.append(v)
        elif id(v) not in seen:
            seen.add(id(v))
            stack.append((v, True))
            stack.extend((c, False) for c in v._prev if id(c) not in seen)
    return topo
