"""K1 のテスト：自分の Value（value.py）の勾配を、数値微分（中心差分）と突き合わせる。

実行（part3/k1_autograd/ で）:
    python3 test_value.py              # 同じフォルダの value.py（あなたの実装）をテスト
    python3 test_value.py --ref        # 参考解答 ../solutions/value.py をテスト（テスト自体が正しいことの確認）
    python3 test_value.py --n 2000     # ランダムな式の数を増やす

マイルストーン：「ランダムな式」テストの最大誤差 ≤ 1e-8（誤差 = |自動微分 − 数値微分| / max(1, |数値微分|)）。
任意の「深い鎖」テストは、backward() を再帰で書いていると RecursionError で落ちます（K2 には不要）。

数値微分の作り方（Part 0 D1 の中心差分と同じ式）:
    D = (f(x+h) − f(x−h)) / 2h。ただし float で計算すると、h を小さくすれば丸め誤差が、大きくすれば打ち切り誤差
    （h² に比例）が効いて、1e-8 の判定には精度が足りない。そこで数値微分の側だけ Python の decimal（50桁）で
    計算し、h = 1e-15 にする。こうすると数値微分の誤差は 1e-20 より小さくなり、残る差は自動微分（float）の側だけになる。
"""
import argparse
import importlib.util
import math
import random
import sys
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_value(path):
    spec = importlib.util.spec_from_file_location("value_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Value


# ---------------------------------------------------------------- 小さなテスト（式が分かっているもの）

def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def grads(Value, f, *xs):
    vs = [Value(x) for x in xs]
    out = f(*vs)
    out.backward()
    return out.data, [v.grad for v in vs]


def unit_tests(Value):
    """(名前, 関数, 入力, 期待する値, 期待する勾配) を順に確かめる。"""
    e2 = math.exp(2.0)
    cases = [
        ("a + b", lambda a, b: a + b, (2.0, -3.0), -1.0, [1.0, 1.0]),
        ("a * b", lambda a, b: a * b, (2.0, -3.0), -6.0, [-3.0, 2.0]),
        ("a ** 3", lambda a: a ** 3, (2.0,), 8.0, [12.0]),
        ("a ** -1", lambda a: a ** -1, (2.0,), 0.5, [-0.25]),
        ("a ** 0.5", lambda a: a ** 0.5, (4.0,), 2.0, [0.25]),
        ("exp(a)", lambda a: a.exp(), (2.0,), e2, [e2]),
        ("log(a)", lambda a: a.log(), (4.0,), math.log(4.0), [0.25]),
        ("relu(a), a > 0", lambda a: a.relu(), (1.5,), 1.5, [1.0]),
        ("relu(a), a < 0", lambda a: a.relu(), (-1.5,), 0.0, [0.0]),
        ("3 − a, 3 / a, −a", lambda a: (3 - a) + 3 / a + (-a), (2.0,), 1.0 + 1.5 - 2.0, [-1.0 - 0.75 - 1.0]),
        # ↓ ここからが「+=」のテスト。同じオブジェクトが2か所で使われる
        ("a * a（同じ値を2回）", lambda a: a * a, (3.0,), 9.0, [6.0]),
        ("a + a（同じ値を2回）", lambda a: a + a, (3.0,), 6.0, [2.0]),
        ("e = a·b; e + e·e（分岐）", lambda a, b: (lambda e: e + e * e)(a * b), (2.0, 0.5), 2.0, [1.5, 6.0]),
        ("残差 x + relu(x·w)", lambda x, w: x + (x * w).relu(), (2.0, 0.5), 3.0, [1.5, 2.0]),
        ("使われない入力の勾配は0", lambda a, b: a * 2.0, (1.0, 5.0), 2.0, [2.0, 0.0]),
    ]
    ok_all = True
    for name, f, xs, val, g_exp in cases:
        try:
            v, g = grads(Value, f, *xs)
            ok = close(v, val) and all(close(a, b) for a, b in zip(g, g_exp))
            msg = "" if ok else f"  値 {v:.6g}（正解 {val:.6g}）、勾配 {[round(x, 6) for x in g]}（正解 {g_exp}）"
        except NotImplementedError as e:
            ok, msg = False, f"  未実装: {e}"
        except Exception as e:  # noqa: BLE001  学習者のコードの例外はすべて表示して続ける
            ok, msg = False, f"  {type(e).__name__}: {e}"
        ok_all &= ok
        print(f"  [{'OK' if ok else 'NG'}] {name}{msg}")
    return ok_all


# ---------------------------------------------------------------- ランダムな式（DAG）

UNARY = ["exp", "log", "relu", "neg", "pow"]
BINARY = ["add", "sub", "mul", "div"]
WITH_CONST = ["radd", "rsub", "rmul", "rdiv"]
POWERS = [2, 3, -1, -2, 0.5, 1.5]


class Reject(Exception):
    """数値微分が当てにならない点（log の引数が0に近い、ReLU の折れ目の近く、値が大きすぎる）を避ける。"""


def make_program(rng, n_vars, n_ops):
    """ノード列を作る。各ノードは既存のノードを参照するので、同じノードが何度も使われる DAG になる。"""
    prog = [("var", i) for i in range(n_vars)]
    for _ in range(n_ops):
        pick = lambda: rng.randrange(max(0, len(prog) - 6), len(prog)) if rng.random() < 0.7 else rng.randrange(len(prog))
        r = rng.random()
        if r < 0.45:
            prog.append((rng.choice(BINARY), pick(), pick()))
        elif r < 0.8:
            op = rng.choice(UNARY)
            prog.append((op, pick(), rng.choice(POWERS)) if op == "pow" else (op, pick()))
        else:
            prog.append((rng.choice(WITH_CONST), pick(), round(rng.uniform(-2, 2), 2) or 0.5))
    return prog


def dec_pow(x, k):
    if k == int(k):
        return x ** int(k)
    return x.sqrt() if k == 0.5 else x * x.sqrt()      # 0.5 乗と 1.5 乗


# 同じ式を3つの「数の体系」で評価する：float（定義域の確認）、Value（自動微分）、Decimal 50桁（数値微分）
FLOAT = dict(lift=float, num=float, exp=math.exp, log=math.log, relu=lambda x: max(x, 0.0), pow=lambda x, k: x ** k)
VALUE = dict(lift=float, num=lambda v: v.data, exp=lambda v: v.exp(), log=lambda v: v.log(), relu=lambda v: v.relu(),
             pow=lambda v, k: v ** k)
DEC = dict(lift=Decimal, num=float, exp=lambda d: d.exp(), log=lambda d: d.ln(), relu=lambda d: max(d, Decimal(0)), pow=dec_pow)


def run(prog, xs, B, relu_signs=None, check=False):
    """prog を数の体系 B で評価する。check=True なら定義域と大きさを確かめる（float のときだけ使う）。"""
    vals, signs = [], []
    for node in prog:
        op = node[0]
        if op == "var":
            vals.append(xs[node[1]])
            continue
        a = vals[node[1]]
        x = B["num"](a)
        if op in BINARY:
            b = vals[node[2]]
            if check and op == "div" and abs(B["num"](b)) < 0.2:
                raise Reject
            v = {"add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b, "div": lambda: a / b}[op]()
        elif op in WITH_CONST:
            c = B["lift"](node[2])
            if check and op == "rdiv" and abs(x) < 0.2:
                raise Reject
            v = {"radd": lambda: c + a, "rsub": lambda: c - a, "rmul": lambda: c * a, "rdiv": lambda: c / a}[op]()
        elif op == "pow":
            k = node[2]
            if check and (abs(x) < 0.2 or (k != int(k) and x < 0.2)):
                raise Reject
            v = B["pow"](a, k)
        elif op == "exp":
            if check and x > 4:
                raise Reject
            v = B["exp"](a)
        elif op == "log":
            if check and x < 0.2:
                raise Reject
            v = B["log"](a)
        elif op == "relu":
            if check and abs(x) < 0.05:
                raise Reject
            signs.append(x > 0)
            v = B["relu"](a)
        else:  # neg
            v = -a
        if check and abs(B["num"](v)) > 100:
            raise Reject
        vals.append(v)
    if relu_signs is not None and signs != relu_signs:
        raise Reject                     # 数値微分で x を動かしたら ReLU の折れ目をまたいだ
    out = vals[-1]
    for v in vals[-4:-1]:                # 最後の数ノードを足して出力にする（多くのノードに勾配が届くように）
        out = out + v
    return out, signs


def numeric_grad(prog, xs, i, signs):
    """50桁の十進演算で中心差分 (f(x+h) − f(x−h)) / 2h をとる（h = 1e-15）。

    float の中心差分は、丸め誤差と打ち切り誤差の兼ね合いで 1e-10〜1e-8 程度が精度の限界（Part 0 D1）。
    桁数を増やせば h を小さくでき、打ち切り誤差（h² に比例）も丸め誤差も 1e-25 以下にできる。
    """
    h = Decimal("1e-15")
    with localcontext() as ctx:
        ctx.prec = 50
        def f(dx):
            ys = [Decimal(x) for x in xs]
            ys[i] += dx
            return run(prog, ys, DEC, relu_signs=signs)[0]
        return float((f(h) - f(-h)) / (2 * h))


def random_tests(Value, n, seed):
    rng = random.Random(seed)
    worst, done, nodes = 0.0, 0, 0
    while done < n:
        n_vars = rng.randint(1, 4)
        prog = make_program(rng, n_vars, rng.randint(3, 25))
        xs = [round(rng.uniform(-2, 2), 3) for _ in range(n_vars)]
        try:
            fx, signs = run(prog, xs, FLOAT, check=True)
            fd = [numeric_grad(prog, xs, i, signs) for i in range(n_vars)]
        except (Reject, OverflowError, ValueError, ZeroDivisionError, InvalidOperation):
            continue
        vs = [Value(x) for x in xs]
        out, _ = run(prog, vs, VALUE)
        if abs(out.data - fx) > 1e-12 * max(1, abs(fx)):
            raise AssertionError(f"順伝播の値が違う: {out.data} != {fx}（式 {prog}）")
        out.backward()
        for v, g in zip(vs, fd):
            worst = max(worst, abs(v.grad - g) / max(1.0, abs(g)))
        done += 1
        nodes += len(prog)
    return worst, nodes


def deep_chain(Value, n=5000):
    x = Value(1.0)
    y = x
    for _ in range(n):
        y = y + x
    y.backward()
    return x.grad == n + 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", action="store_true", help="参考解答 ../solutions/value.py をテストする")
    ap.add_argument("--path", help="テストする value.py のパス（既定：同じフォルダの value.py）")
    ap.add_argument("--n", type=int, default=500, help="ランダムな式の数")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    path = Path(a.path) if a.path else (HERE.parent / "solutions" / "value.py" if a.ref else HERE / "value.py")
    print(f"テスト対象: {path}")
    Value = load_value(path)

    print("\n1. 基本演算と「+=」（同じ値を2回使う）")
    ok_unit = unit_tests(Value)

    print(f"\n2. ランダムな式 {a.n} 個（seed {a.seed}）を数値微分と比べる")
    try:
        worst, nodes = random_tests(Value, a.n, a.seed)
        ok_rand = worst <= 1e-8
        print(f"  ノード数の合計 {nodes}、最大誤差 {worst:.1e}  → {'OK（≤ 1e-8）' if ok_rand else 'NG（> 1e-8）'}")
    except NotImplementedError as e:
        ok_rand = False
        print(f"  未実装: {e}")

    print("\n3. 深い鎖（任意）：y = x + x + … + x（5001項）を backward。正解 ∂y/∂x = 5001")
    try:
        print(f"  [{'OK' if deep_chain(Value) else 'NG'}]")
    except RecursionError:
        print("  RecursionError：backward() が再帰。K2 には影響しないが、スタックを使う形に書き直すと深いグラフでも動く")
    except NotImplementedError as e:
        print(f"  未実装: {e}")

    print("\nマイルストーン:", "達成（K2 へ進む: python3 ../k2_verify_part1.py --mine）" if ok_unit and ok_rand else "未達")
    sys.exit(0 if ok_unit and ok_rand else 1)


if __name__ == "__main__":
    main()
