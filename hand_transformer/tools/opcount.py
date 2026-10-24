#!/usr/bin/env python3
"""Exact hand-cost accounting for hand_transformer/params.json.

Replays every scalar operation a human does at each checkpoint (2.1 .. 7.4), in worksheet order,
with the structural skips a smart learner uses, and classifies each product by its factors:
  skip   : one factor is 0
  sign   : one factor is +-1           (sign_n: other factor nice, sign_d: other factor decimal)
  nn     : both factors multiples of 0.5
  nd     : one factor a multiple of 0.5
  dd     : both decimal                (dd_sq: a square of a decimal, x^2 key)
Special ops: exp, log, sqrt, div (by data value / by nice constant), scale (x 1/sqrt2).
Also: standalone additions (outside M+ accumulation), numbers written, structurally skipped products.

Values are classified as the learner records them (4 significant digits), unless --exact.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mini_gpt as M  # noqa: E402

INP, TGT = M.INP, M.TGT
MODE = {'view': 'r4'}
TOL = 1e-12


def view(x):
    x = float(x)
    if MODE['view'] == 'exact' or x == 0 or not np.isfinite(x):
        return x
    return float(f"{x:.4g}")


def is0(x): return abs(view(x)) < TOL
def is1(x): return abs(abs(view(x)) - 1) < TOL
def nice(x): v = view(x); return abs(2 * v - round(2 * v)) < TOL
def short(x): v = view(x); return abs(8 * v - round(8 * v)) < TOL


# ------------------------------------------------------------------ time model (seconds)
TM = {
    'skip': 1,                       # notice the 0 and move on
    'sign_n': 2, 'sign_d': 4,        # +-1 x nice: mental; +-1 x decimal: key the number into M+/M-
    'nn': 3,                         # 0.5-multiples: mental (1.5 x -0.5)
    'nd': 7,                         # 0.5 x 0.7428 on the calculator, M+
    'dd': 16,                        # 0.2572 x 1.0686 M+ (two 4-digit entries)
    'dd_sq': 10,                     # x^2 key: one entry
    'a_triv': 0, 'a_nice': 2, 'a_dec': 5,
    'd_triv': 2, 'd_nn': 3, 'd_nd': 7, 'd_dec': 15,
    'exp': 15, 'exp_triv': 2, 'log': 15, 'log_triv': 2, 'sqrt': 15, 'sqrt_triv': 2,
    'scale': 10,                     # x 0.70711 (constant kept in a calculator variable)
    'scale_skip': 1, 'scratch': 0,
    'w_short': 3, 'w_dec': 8, 'w_copy': 4,   # write a result (read display, round to 4 sig. digits, write, MC)
    'struct': 0,
}
MUL_KEYS = ['skip', 'sign_n', 'sign_d', 'nn', 'nd', 'dd', 'dd_sq']


class Tally:
    def __init__(self):
        self.c = defaultdict(Counter)
        self.order = []
        self.cur = None

    def at(self, cid):
        self.cur = cid
        if cid not in self.order:
            self.order.append(cid)

    def n(self, k, m=1):
        self.c[self.cur][k] += m

    def mul(self, a, b):
        if is0(a) or is0(b):
            k = 'skip'
        elif is1(a) or is1(b):
            o = b if is1(a) else a
            k = 'sign_n' if nice(o) else 'sign_d'
        elif nice(a) and nice(b):
            k = 'nn'
        elif nice(a) or nice(b):
            k = 'nd'
        else:
            k = 'dd'
        self.n(k)

    def dot(self, xs, ys):
        for a, b in zip(xs, ys):
            self.mul(a, b)

    def sq(self, a):
        if is0(a):
            k = 'skip'
        elif is1(a):
            k = 'sign_n'
        elif nice(a):
            k = 'nn'
        else:
            k = 'dd_sq'
        self.n(k)

    def add(self, a, b):
        if is0(a) or is0(b):
            k = 'a_triv'
        elif nice(a) and nice(b):
            k = 'a_nice'
        else:
            k = 'a_dec'
        self.n(k)

    def sum(self, xs):
        xs = list(xs)
        s = xs[0]
        for x in xs[1:]:
            self.add(s, x)
            s = s + x
        return s

    def div(self, a, b):
        if is0(a) or is1(b):
            k = 'd_triv'
        elif nice(b):
            k = 'd_nn' if nice(a) else 'd_nd'
        else:
            k = 'd_dec'
        self.n(k)

    def exp(self, x): self.n('exp_triv' if is0(x) else 'exp')
    def log(self, x): self.n('log_triv' if is1(x) else 'log')

    def sqrt(self, x):
        self.n('sqrt_triv' if nice(np.sqrt(x)) else 'sqrt')   # sqrt(0.25), sqrt(4): known squares

    def scale(self, x): self.n('scale_skip' if is0(x) else 'scale')
    def write(self, x): self.n('w_short' if short(x) else 'w_dec')

    def scratch(self, x):                       # written on paper but not a worksheet cell
        self.write(x); self.n('scratch')
    def copy(self, m=1): self.n('w_copy', m)
    def struct(self, m): self.n('struct', m)


# ------------------------------------------------------------------ forward pass (2.1 - 5.3)

def ln_fwd(T, X, st, U, cid_stats, cid_out):
    T.at(cid_stats)
    xcs = []
    for t in range(3):
        x = X[t]
        mu, var, sig = st['mu'][t, 0], st['var'][t, 0], st['sigma'][t, 0]
        s = T.sum(x)
        T.div(s, 4); T.write(mu)
        xc = x - mu
        if not is0(mu):                          # centred values are needed twice: write them
            for i in range(4):
                T.add(x[i], -mu); T.scratch(xc[i])
        for i in range(4):
            T.sq(xc[i])                         # x^2 M+  (no separate additions)
        T.div(var * 4, 4); T.write(var)
        T.sqrt(var); T.write(sig)
        xcs.append(xc)
    T.at(cid_out)
    for t in range(3):
        for i in range(4):
            T.div(xcs[t][i], st['sigma'][t, 0]); T.write(U[t, i])


def softmax_row(T, s, out):
    m = max(s)
    es = []
    for v in s:
        if v == m:
            T.exp(0.0); es.append(1.0)
        else:
            T.add(v, -m); T.exp(v - m); e = np.exp(v - m); T.scratch(e); es.append(e)
    tot = T.sum(es)
    if len(es) > 1:
        T.scratch(tot)
    for e, o in zip(es, out):
        T.div(e, tot); T.write(o)
    return m, tot


def dead_units(c):
    return [k for k in range(M.FF) if np.all(c['Z'][:, k] <= 0)]


def fwd(T, p, c, tag, first):
    Q, K, V = c['Q'], c['K'], c['V']
    T.at(tag('2.1'))
    Eids = p['E'][INP]
    if first:
        T.copy(12)                               # E[ids] table on the worksheet
    for t in range(3):
        for j in range(4):
            T.add(Eids[t, j], p['P'][t, j]); T.write(c['X0'][t, j])
    ln_fwd(T, c['X0'], c['ln1'], c['U1'], tag('2.2'), tag('2.3'))

    T.at(tag('3.1'))
    for name, W, Mx in (('Q', p['Wq'], Q), ('K', p['Wk'], K), ('V', p['Wv'], V)):
        rows = [1, 2] if name == 'Q' else [0, 1, 2]
        if name == 'Q':
            T.struct(16)                         # Q row 0 is never needed
        for t in rows:
            for j in range(4):
                T.dot(c['U1'][t], W[:, j]); T.write(Mx[t, j])

    T.at(tag('3.2'))
    for h in range(M.H):
        sl = M.head_slice(h)
        T.struct(2 * (1 + 3))                    # S[0,0] (needs Q row 0) + 3 masked entries
        for i in (1, 2):
            for j in range(i + 1):
                T.dot(Q[i, sl], K[j, sl])
                raw = float(Q[i, sl] @ K[j, sl]); T.scratch(raw)
                T.scale(raw); T.write(c['S'][h][i, j])

    T.at(tag('3.3'))
    for h in range(M.H):
        T.write(1.0)                             # A[0,0] = 1 by structure
        for i in (1, 2):
            softmax_row(T, list(c['S'][h][i, :i + 1]), list(c['A'][h][i, :i + 1]))

    T.at(tag('3.4'))
    for h in range(M.H):
        sl = M.head_slice(h)
        T.struct(2 * 3)                          # masked A entries
        for i in range(3):
            for col in range(sl.start, sl.stop):
                T.dot(c['A'][h][i, :i + 1], V[:i + 1, col]); T.write(c['O'][h][i, col - sl.start])
    if first:
        T.copy(12)                               # Ocat table

    T.at(tag('3.5'))
    for t in range(3):
        for j in range(4):
            T.dot(c['Ocat'][t], p['Wo'][:, j]); T.write(c['Attn'][t, j])

    T.at(tag('4.1'))
    for t in range(3):
        for j in range(4):
            T.add(c['X0'][t, j], c['Attn'][t, j]); T.write(c['X1'][t, j])
    ln_fwd(T, c['X1'], c['ln2'], c['U2'], tag('4.2'), tag('4.2'))

    T.at(tag('4.3'))
    for t in range(3):
        for j in range(M.FF):
            T.dot(c['U2'][t], p['W1'][:, j]); T.write(c['Z'][t, j])
    if first:
        for v in c['Z'].ravel():
            T.copy() if v > 0 else T.write(0.0)  # R table (2nd pass: read R' off Z')

    T.at(tag('4.4'))
    dead = dead_units(c)
    for t in range(3):
        for j in range(4):
            for k in range(M.FF):
                if k in dead:
                    T.struct(1)
                else:
                    T.mul(c['R'][t, k], p['W2'][k, j])
            T.write(c['F'][t, j])

    T.at(tag('4.5'))
    for t in range(3):
        for j in range(4):
            T.add(c['X1'][t, j], c['F'][t, j]); T.write(c['X2'][t, j])

    T.at(tag('5.1'))
    for t in range(3):
        for j in range(M.V):
            T.dot(c['X2'][t], p['Wout'][:, j]); T.write(c['logits'][t, j])

    T.at(tag('5.2'))
    lse = []
    for t in range(3):
        m, tot = softmax_row(T, list(c['logits'][t]), list(c['probs'][t]))
        lse.append((m, tot))

    T.at(tag('5.3'))                             # nll = logsumexp - logit_target
    for t in range(3):
        m, tot = lse[t]
        T.log(tot); T.add(np.log(tot), m); T.add(m + np.log(tot), -c['logits'][t, TGT[t]]); T.write(c['nll'][t])
    s = T.sum(c['nll']); T.div(s, 3); T.write(c['loss'])


# ------------------------------------------------------------------ backward pass (6.1 - 6.9)

def ln_bwd(T, dy_all, y_all, sig_all, out):
    for t in range(3):
        dy, y, sig = dy_all[t], y_all[t], sig_all[t, 0]
        s = T.sum(dy); m1 = s / 4; T.div(s, 4); T.scratch(m1)
        T.dot(dy, y); s2 = float(dy @ y); m2 = s2 / 4; T.div(s2, 4); T.scratch(m2)
        for i in range(4):
            T.mul(y[i], m2)
            a = dy[i] - m1; T.add(dy[i], -m1)
            T.add(a, -y[i] * m2)
            T.div(a - y[i] * m2, sig); T.write(out[t, i])


def bwd(T, p, c, g, d):
    dead = dead_units(c)
    act = c['Z'] > 0
    T.at('6.1')
    Y = np.zeros((3, M.V)); Y[np.arange(3), TGT] = 1
    for t in range(3):
        for j in range(M.V):
            if Y[t, j]:
                T.add(c['probs'][t, j], -1.0)
            T.div(c['probs'][t, j] - Y[t, j], 3); T.write(d['logits'][t, j])
    for k in range(4):
        for j in range(M.V):
            T.dot(c['X2'][:, k], d['logits'][:, j]); T.write(g['Wout'][k, j])
    for t in range(3):
        for k in range(4):
            T.dot(d['logits'][t], p['Wout'][k]); T.write(d['X2'][t, k])

    T.at('6.2')
    T.copy(12)                                   # dF = dX2
    for k in range(M.FF):
        for j in range(4):
            if k in dead:
                T.struct(3); T.write(0.0)
            else:
                T.dot(c['R'][:, k], d['F'][:, j]); T.write(g['W2'][k, j])
    for t in range(3):
        for k in range(M.FF):
            if act[t, k]:
                T.dot(d['F'][t], p['W2'][k]); T.write(d['R'][t, k])
            else:
                T.struct(4)                      # dZ = 0 there (incl. dead column h3)
    for t in range(3):
        for k in range(M.FF):
            T.copy() if act[t, k] else T.write(0.0)   # dZ table
    for k in range(4):
        for j in range(M.FF):
            if j in dead:
                T.struct(3); T.write(0.0)
            else:
                T.dot(c['U2'][:, k], d['Z'][:, j]); T.write(g['W1'][k, j])
    for t in range(3):
        for k in range(4):
            for j in range(M.FF):
                if j in dead:
                    T.struct(1)
                else:
                    T.mul(d['Z'][t, j], p['W1'][k, j])
            T.write(d['U2'][t, k])

    T.at('6.3')
    ln_bwd(T, d['U2'], c['U2'], c['ln2']['sigma'], d['X1_ln2'])
    for t in range(3):
        for i in range(4):
            T.add(d['X2'][t, i], d['X1_ln2'][t, i]); T.write(d['X1'][t, i])

    T.at('6.4')
    T.copy(12)                                   # dAttn = dX1
    for k in range(4):
        for j in range(4):
            T.dot(c['Ocat'][:, k], d['Attn'][:, j]); T.write(g['Wo'][k, j])
    for t in range(3):
        for k in range(4):
            T.dot(d['Attn'][t], p['Wo'][k]); T.write(d['Ocat'][t, k])

    T.at('6.5')
    for h in range(M.H):
        sl = M.head_slice(h)
        dO, V, A = d['Ocat'][:, sl], c['V'][:, sl], c['A'][h]
        T.struct(2 * (1 + 3))                    # dA row 0 + 3 masked entries
        for i in (1, 2):
            for j in range(i + 1):
                T.dot(dO[i], V[j]); T.write(d['A'][h][i, j])
        T.write(0.0)                             # dS[0,0] = 0 by structure
        T.struct(2 * 4)                          # rowsum(dA*A) and A*(dA-r) for row 0 + masked entries
        for i in (1, 2):
            dA = d['A'][h][i, :i + 1]
            T.dot(dA, A[i, :i + 1]); r = float(dA @ A[i, :i + 1]); T.scratch(r)
            for j in range(i + 1):
                T.add(dA[j], -r); T.mul(A[i, j], dA[j] - r); T.write(d['S'][h][i, j])

    T.at('6.6')
    for h in range(M.H):
        sl = M.head_slice(h)
        dS, Qh, Kh, A, dO = d['S'][h], c['Q'][:, sl], c['K'][:, sl], c['A'][h], d['Ocat'][:, sl]
        T.struct(2 * 3); T.write(0.0); T.write(0.0)     # dQ row 0 = 0
        T.struct(2 + 6 + 6)                      # masked dS / A entries in dQ, dK, dV
        for i in (1, 2):
            for col in range(2):
                T.dot(dS[i, :i + 1], Kh[:i + 1, col]); raw = float(dS[i, :i + 1] @ Kh[:i + 1, col])
                T.scale(raw); T.write(d['Q'][i, sl.start + col])
        for j in range(3):
            qs = [i for i in range(max(j, 1), 3)]
            T.struct(2 * (1 if j == 0 else 0))          # query row 0 contributes nothing
            for col in range(2):
                T.dot(dS[qs, j], Qh[qs, col]); raw = float(dS[qs, j] @ Qh[qs, col])
                T.scale(raw); T.write(d['K'][j, sl.start + col])
        for j in range(3):
            for col in range(2):
                T.dot(A[j:, j], dO[j:, col]); T.write(d['V'][j, sl.start + col])

    T.at('6.7')
    U1 = c['U1']
    for name in ('Q', 'K', 'V'):
        rows = [1, 2] if name == 'Q' else [0, 1, 2]
        if name == 'Q':
            T.struct(16)
        for k in range(4):
            for j in range(4):
                T.dot(U1[rows, k], d[name][rows, j]); T.write(g['W' + name.lower()][k, j])
    for t in range(3):
        for k in range(4):
            for name in ('Q', 'K', 'V'):
                if name == 'Q' and t == 0:
                    T.struct(4); continue
                T.dot(d[name][t], p['W' + name.lower()][k])
            T.write(d['U1'][t, k])

    T.at('6.8')
    ln_bwd(T, d['U1'], c['U1'], c['ln1']['sigma'], d['X0_ln1'])
    for t in range(3):
        for i in range(4):
            T.add(d['X1'][t, i], d['X0_ln1'][t, i]); T.write(d['X0'][t, i])

    T.at('6.9')
    T.copy(12)                                   # dP = dX0
    T.copy(12); T.n('w_short', 8)                # dE: used rows copied, unused rows 0


# ------------------------------------------------------------------ update (7.1, 7.2)

def update(T, p, g, p1, lr):
    T.at('7.1')
    for k in M.PARAMS:
        for idx in np.ndindex(p[k].shape):
            th, gr = p[k][idx], g[k][idx]
            if is0(gr):
                T.write(th)                      # unchanged
            elif is0(th):
                T.n('a_triv'); T.copy()          # th' = -g : sign flip copy
            else:
                T.add(th, -lr * gr); T.write(p1[k][idx])
    T.at('7.2')
    for k in M.PARAMS:
        if k == 'E':
            T.struct(g['E'].size); T.mul(2.0, float((g['P'] ** 2).sum()))   # ||dE||^2 = ||dP||^2
        else:
            for v in g[k].ravel():
                T.sq(v)
        T.write(float((g[k] ** 2).sum()))        # subtotal per matrix
    gsq = sum(float((g[k] ** 2).sum()) for k in M.PARAMS)
    T.write(gsq); T.write(-lr * gsq)


def loss_cmp(T, c, c1, gsq, lr):
    T.at('7.4')
    T.write(c1['loss'] - c['loss']); T.add(c1['loss'], -c['loss'])
    T.copy(2)                                    # L and predicted dL copied into the table
    T.copy(6)                                    # p[target] before / after


# ------------------------------------------------------------------ preparation (step 1) and checksums

WMATS = ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2', 'Wout']


def prep(T, p, cid):
    T.at(cid)
    for k in WMATS:
        W = p[k]
        for r in W:
            T.sum(r); T.write(r.sum())
        for col in W.T:
            T.sum(col); T.write(col.sum())


def chk_right(T, X, W, Y, rows):
    """rowsum(X W) = X (W 1)."""
    w = W.sum(1)
    for t in rows:
        T.dot(X[t], w); T.sum(Y[t])


def chk_left(T, X, dY, dW):
    """rowsum(X^T dY) = X^T (dY 1)."""
    s = dY.sum(1)
    for t in range(dY.shape[0]):
        T.sum(dY[t])
    for k in range(dW.shape[0]):
        T.dot(X[:, k], s); T.sum(dW[k])


def checks_fwd(T, p, c, tag):
    """Row-sum checksums of every matrix product + the cheap invariants of the derivations.md checklist."""
    T.at(tag('2.3'))
    for t in range(3):
        T.sum(c['U1'][t])                        # row mean of LN output = 0
    T.at(tag('3.1'))
    chk_right(T, c['U1'], p['Wq'], c['Q'], [1, 2])
    chk_right(T, c['U1'], p['Wk'], c['K'], range(3))
    chk_right(T, c['U1'], p['Wv'], c['V'], range(3))
    T.at(tag('3.3'))
    for h in range(M.H):
        for i in (1, 2):
            T.sum(c['A'][h][i, :i + 1])          # row sum = 1
    T.at(tag('3.5'))
    chk_right(T, c['Ocat'], p['Wo'], c['Attn'], range(3))
    T.at(tag('4.2'))
    for t in range(3):
        T.sum(c['U2'][t])
    T.at(tag('4.3'))
    chk_right(T, c['U2'], p['W1'], c['Z'], range(3))
    T.at(tag('4.4'))
    chk_right(T, c['R'], p['W2'], c['F'], range(3))
    T.at(tag('5.1'))
    chk_right(T, c['X2'], p['Wout'], c['logits'], range(3))
    T.at(tag('5.2'))
    for t in range(3):
        T.sum(c['probs'][t])


def checks_bwd(T, p, c, g, d):
    T.at('6.1')
    chk_left(T, c['X2'], d['logits'], g['Wout'])   # dlogits rows sum to 0 -> dWout rows sum to 0
    chk_right(T, d['logits'], p['Wout'].T, d['X2'], range(3))
    T.at('6.2')
    chk_left(T, c['R'], d['F'], g['W2'])
    chk_left(T, c['U2'], d['Z'], g['W1'])
    chk_right(T, d['Z'], p['W1'].T, d['U2'], range(3))
    T.at('6.3')
    for t in range(3):                            # LN backward: sum = 0 and dot with U2 = 0
        T.sum(d['X1_ln2'][t]); T.dot(d['X1_ln2'][t], c['U2'][t])
    T.at('6.4')
    chk_left(T, c['Ocat'], d['Attn'], g['Wo'])
    chk_right(T, d['Attn'], p['Wo'].T, d['Ocat'], range(3))
    T.at('6.5')
    for h in range(M.H):
        for i in (1, 2):
            T.sum(d['S'][h][i, :i + 1])           # dS row sum = 0
    T.at('6.6')
    for h in range(M.H):
        sl = M.head_slice(h)
        for col in range(sl.start, sl.stop):
            T.sum(d['K'][:, col])                 # dK column sum = 0
            T.sum(d['V'][:, col]); T.sum(d['Ocat'][:, col])   # dV column sum = dO column sum
    T.at('6.7')
    for n in 'qkv':
        chk_left(T, c['U1'], d[n.upper()], g['W' + n])
    for t in range(3):                            # rowsum(dU1) = dQ.cs(Wq) + dK.cs(Wk) + dV.cs(Wv)
        for n in 'QKV':
            if n == 'Q' and t == 0:
                continue
            T.dot(d[n][t], p['W' + n.lower()].sum(0))
        T.sum(d['U1'][t])
    T.at('6.8')
    for t in range(3):
        T.sum(d['X0_ln1'][t]); T.dot(d['X0_ln1'][t], c['U1'][t])


# ------------------------------------------------------------------ dense counts (no skipping at all)

def dense():
    T_, D_, F_, V_, H_, DK_ = 3, 4, 8, 5, 2, 2
    fw = {
        '2.2 LN1 squares': T_ * D_,
        '3.1 QKV': 3 * T_ * D_ * D_,
        '3.2 scores': H_ * T_ * T_ * DK_,
        '3.4 AV': H_ * T_ * T_ * DK_,
        '3.5 Wo': T_ * D_ * D_,
        '4.2 LN2 squares': T_ * D_,
        '4.3 W1': T_ * D_ * F_,
        '4.4 W2': T_ * F_ * D_,
        '5.1 Wout': T_ * D_ * V_,
    }
    bw = {
        '6.1 dWout+dX2': 2 * T_ * D_ * V_,
        '6.2 dW2+dR+dW1+dU2': 4 * T_ * D_ * F_,
        '6.3 LN2 bwd': 2 * T_ * D_,
        '6.4 dWo+dOcat': 2 * T_ * D_ * D_,
        '6.5 dA+dS': H_ * T_ * T_ * DK_ + 2 * H_ * T_ * T_,
        '6.6 dQ+dK+dV': 3 * H_ * T_ * T_ * DK_,
        '6.7 dWqkv+dU1': 6 * T_ * D_ * D_,
        '6.8 LN1 bwd': 2 * T_ * D_,
    }
    sp_fw = {'scale': H_ * T_ * T_, 'exp': H_ * T_ * T_ + T_ * V_, 'div(softmax)': H_ * T_ * T_ + T_ * V_,
             'div(LN /sigma)': 2 * T_ * D_, 'sqrt': 2 * T_, 'log': T_}
    sp_bw = {'scale': 2 * H_ * T_ * DK_, 'div(LN /sigma)': 2 * T_ * D_, 'div(/3)': T_ * V_}
    return fw, bw, sp_fw, sp_bw


# ------------------------------------------------------------------ main

STEPS = [  # (id, label, checkpoints, budget hours)
    ('1', 'step1 準備（行和・列和）', ['prep'], 1.0),
    ('2', 'step2 埋め込み+LN1', ['2.1', '2.2', '2.3'], 0.5),
    ('3', 'step3 Attention', ['3.1', '3.2', '3.3', '3.4', '3.5'], 1.75),
    ('4', 'step4 残差+LN2+FFN', ['4.1', '4.2', '4.3', '4.4', '4.5'], 1.75),
    ('5', 'step5 出力・softmax・損失', ['5.1', '5.2', '5.3'], 0.75),
    ('6a', '6a 出力層', ['6.1'], 0.75),
    ('6b', '6b FFN', ['6.2'], 1.25),
    ('6c', '6c LN2', ['6.3'], 0.75),
    ('6d', '6d Wo・softmax・QKV', ['6.4', '6.5', '6.6'], 1.5),
    ('6e', '6e 射影・LN1・埋め込み', ['6.7', '6.8', '6.9'], 1.25),
    ('7a', '7a 更新+1次近似', ['7.1', '7.2'], 1.0),
    ('7b', '7b 再順伝播+比較', ['7.3', '7.4'], 3.75),
]


def secs(cn):
    return sum(TM[k] * v for k, v in cn.items())


def summarize(cn):
    mul = {k: cn.get(k, 0) for k in MUL_KEYS}
    out = {
        'skip': mul['skip'], 'sign': mul['sign_n'] + mul['sign_d'], 'nn': mul['nn'], 'nd': mul['nd'],
        'dd': mul['dd'] + mul['dd_sq'], 'dd_sq': mul['dd_sq'],
        'exp': cn.get('exp', 0), 'exp_triv': cn.get('exp_triv', 0),
        'log': cn.get('log', 0), 'sqrt': cn.get('sqrt', 0), 'sqrt_triv': cn.get('sqrt_triv', 0),
        'div_dec': cn.get('d_dec', 0), 'div_nice': cn.get('d_nn', 0) + cn.get('d_nd', 0), 'div_triv': cn.get('d_triv', 0),
        'scale': cn.get('scale', 0),
        'add': cn.get('a_nice', 0) + cn.get('a_dec', 0), 'add_dec': cn.get('a_dec', 0), 'add_triv': cn.get('a_triv', 0),
        'write': cn.get('w_short', 0) + cn.get('w_dec', 0) + cn.get('w_copy', 0),
        'w_dec': cn.get('w_dec', 0), 'struct': cn.get('struct', 0), 'scratch': cn.get('scratch', 0),
    }
    out['nontriv_mul'] = out['sign'] + out['nn'] + out['nd'] + out['dd']
    out['all_mul'] = out['nontriv_mul'] + out['skip']
    out['sec'] = secs(cn)
    out['sec_mul'] = sum(TM[k] * cn.get(k, 0) for k in MUL_KEYS)
    out['sec_special'] = sum(TM[k] * cn.get(k, 0) for k in ('exp', 'exp_triv', 'log', 'log_triv', 'sqrt', 'sqrt_triv',
                                                              'd_triv', 'd_nn', 'd_nd', 'd_dec', 'scale', 'scale_skip'))
    out['sec_add'] = sum(TM[k] * cn.get(k, 0) for k in ('a_triv', 'a_nice', 'a_dec'))
    out['sec_write'] = sum(TM[k] * cn.get(k, 0) for k in ('w_short', 'w_dec', 'w_copy'))
    return out


def run():
    p, lr = M.load_params()
    c = M.forward(p)
    g, d = M.backward(p, c)
    p1 = M.sgd(p, g, lr)
    c1 = M.forward(p1)
    gsq = sum(float((g[k] ** 2).sum()) for k in M.PARAMS)

    T = Tally()
    prep(T, p, 'prep')
    fwd(T, p, c, lambda cid: cid, first=True)
    bwd(T, p, c, g, d)
    update(T, p, g, p1, lr)
    T2 = Tally()                                  # second pass broken down by stage
    fwd(T2, p1, c1, lambda cid: '7.3/' + cid, first=False)
    for k, cn in T2.c.items():
        if k == '7.3/5.3':
            T.c['7.4'].update(cn)                # nll', L' belong to 7.4
        else:
            T.c['7.3'].update(cn)
    loss_cmp(T, c, c1, gsq, lr)

    C = Tally()                                   # checksum / invariant (self-check) cost, by checkpoint
    checks_fwd(C, p, c, lambda cid: cid)
    checks_bwd(C, p, c, g, d)
    C.at('7.3')
    for k in WMATS:                               # row sums of updated weights W' 1
        for r in p1[k]:
            C.sum(r)
    checks_fwd(C, p1, c1, lambda cid: '7.3')
    return p, c, g, d, p1, c1, T, T2, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exact', action='store_true', help='classify exact float values instead of 4-sig-digit values')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--md', metavar='PATH', help='write Markdown tables for cost.md')
    a = ap.parse_args()
    if a.exact:
        MODE['view'] = 'exact'
    p, c, g, d, p1, c1, T, T2, C = run()

    cps = ['prep', '2.1', '2.2', '2.3', '3.1', '3.2', '3.3', '3.4', '3.5', '4.1', '4.2', '4.3', '4.4', '4.5',
           '5.1', '5.2', '5.3', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9', '7.1', '7.2', '7.3', '7.4']
    res = {'mode': MODE['view'], 'time_model_sec': TM, 'checkpoints': {}, 'steps': {}, 'pass2': {}, 'checks': {}}
    CMP = 2                                       # seconds to compare one worksheet value with answers.md

    def agg(ids):
        tot, chk = Counter(), Counter()
        for i in ids:
            tot.update(T.c[i]); chk.update(C.c[i])
        s = summarize(tot)
        cs = summarize(chk)
        s['chk_sec'] = cs['sec']; s['chk_mul'] = cs['nontriv_mul']; s['chk_dd'] = cs['dd']; s['chk_add'] = cs['add']
        s['cmp_n'] = sum(cmp_n(i) for i in ids); s['cmp_sec'] = CMP * s['cmp_n']
        return s

    # values that can be compared with answers.md: everything written on the worksheet, except the step-1 sums
    # (not in the answer key) and, in 7.3, only the cells of the answer tables (main intermediates)
    ans73 = 0
    for cid, _, items in M.checkpoints(*M.load_params()):
        if cid == '7.3':
            for label, Mx, _, _ in items:
                Mx = np.atleast_2d(Mx)
                ans73 += Mx.size - (int(M.MASK.sum()) if M.is_masked(label, Mx) else 0)

    def cmp_n(cid):
        if cid == 'prep':
            return 0
        if cid == '7.3':
            return ans73
        cn = T.c[cid]
        return cn.get('w_short', 0) + cn.get('w_dec', 0) + cn.get('w_copy', 0) - cn.get('scratch', 0)

    for cid in cps:
        res['checkpoints'][cid] = agg([cid])
    for sid, label, ids, bud in STEPS:
        s = agg(ids); s['label'] = label; s['budget_h'] = bud; s['checkpoints'] = ids
        res['steps'][sid] = s
    for k in sorted(T2.c):
        res['pass2'][k] = summarize(T2.c[k])
    groups = {'forward1': cps[1:17], 'backward': cps[17:26], 'update': ['7.1', '7.2'], 'forward2': ['7.3', '7.4']}
    for gname, ids in groups.items():
        res.setdefault('groups', {})[gname] = agg(ids)
        res['checks'][gname] = summarize(sum((C.c[i] for i in ids), Counter()))
    res['total'] = agg(cps)
    fw, bw, sp_fw, sp_bw = dense()
    res['dense'] = {'forward': fw, 'forward_total': sum(fw.values()), 'backward': bw, 'backward_total': sum(bw.values()),
                    'special_forward': sp_fw, 'special_backward': sp_bw,
                    'update_mults_lr_x_g': 180, 'update_subtractions': 180, 'grad_sq_7_2': 180,
                    'forward2_total': sum(fw.values())}
    # tiny (|x| < 5e-5) decimal factors in nontrivial products are worth knowing about
    res['tiny_grad_entries'] = {k: int(((np.abs(g[k]) < 5e-5) & (np.abs(g[k]) > 1e-12)).sum()) for k in M.PARAMS}
    if a.json:
        print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
        return
    hdr = f"{'cp':6s} {'skip':>5s} {'±1':>5s} {'nn':>5s} {'nd':>5s} {'dd':>5s} {'(sq)':>5s} {'exp':>4s} {'log':>4s} {'sqrt':>4s} " \
          f"{'divD':>4s} {'divN':>4s} {'scl':>4s} {'add':>5s} {'wr':>5s} {'strc':>5s} {'min':>7s}"
    print(f"mode = {MODE['view']}")
    print(hdr)
    for cid in cps:
        s = res['checkpoints'][cid]
        print(f"{cid:6s} {s['skip']:5d} {s['sign']:5d} {s['nn']:5d} {s['nd']:5d} {s['dd']:5d} {s['dd_sq']:5d} {s['exp']:4d} "
              f"{s['log']:4d} {s['sqrt']:4d} {s['div_dec']:4d} {s['div_nice']:4d} {s['scale']:4d} {s['add']:5d} {s['write']:5d} "
              f"{s['struct']:5d} {s['sec'] / 60:7.1f}")
    print()
    for sid, s in res['steps'].items():
        print(f"{s['label']:28s} mul {s['nontriv_mul']:4d} (skip {s['skip']:3d})  arith {s['sec'] / 3600:5.2f} h  budget {s['budget_h']:5.2f} h"
              f"  ratio {s['sec'] / 3600 / s['budget_h']:4.2f}  [mul {s['sec_mul'] / 60:5.1f} sp {s['sec_special'] / 60:5.1f}"
              f" add {s['sec_add'] / 60:5.1f} wr {s['sec_write'] / 60:5.1f} min]")
    print()
    for gname, s in res['groups'].items():
        print(f"{gname:10s} nontriv mul {s['nontriv_mul']:4d}  all mul {s['all_mul']:4d}  dd {s['dd']:4d}  struct {s['struct']:4d}  {s['sec'] / 3600:5.2f} h")
    s = res['total']
    print(f"TOTAL      nontriv mul {s['nontriv_mul']:4d}  all mul {s['all_mul']:4d}  dd {s['dd']:4d}  struct {s['struct']:4d}  {s['sec'] / 3600:5.2f} h")
    print()
    print('pass 2 by stage:')
    for k, s in res['pass2'].items():
        print(f"  {k:9s} mul {s['nontriv_mul']:4d} (skip {s['skip']:3d}, sign {s['sign']:3d}, nn {s['nn']:3d}, nd {s['nd']:3d}, dd {s['dd']:3d})  {s['sec'] / 60:6.1f} min")
    print()
    print('checksums + invariants:')
    for k, s in res['checks'].items():
        print(f"  {k:5s} mul {s['nontriv_mul']:4d} (skip {s['skip']:3d}, sign {s['sign']:3d}, nn {s['nn']:3d}, nd {s['nd']:3d}, dd {s['dd']:3d}) add {s['add']:4d}  {s['sec'] / 60:6.1f} min")
    print()
    print('dense:', json.dumps(res['dense'], ensure_ascii=False))
    print('tiny grads:', res['tiny_grad_entries'])

    # reconciliation: replayed products + structural skips must equal the dense count
    print()
    print('reconcile (replayed incl. zero-skips + structural  vs  dense):')
    dn = res['dense']
    for gname, dval in (('forward1', dn['forward_total']), ('backward', dn['backward_total']), ('forward2', dn['forward2_total'])):
        s = res['groups'][gname]
        print(f"  {gname:9s} {s['all_mul']} + {s['struct']} = {s['all_mul'] + s['struct']}   dense {dval}")
    s = res['checkpoints']['7.2']
    print(f"  7.2       {s['all_mul']} + {s['struct']} = {s['all_mul'] + s['struct']}   dense 180 squares (+1 for x2)")

    # sensitivity of the hour estimates to the dec x dec time
    print()
    base = dict(TM)
    for ddv in (12, 16, 20):
        TM['dd'] = ddv
        tot = Counter()
        for i in cps:
            tot.update(T.c[i])
        s7 = Counter(); s7.update(T.c['7.3']); s7.update(T.c['7.4'])
        print(f"  dd={ddv:2d}s  total {secs(tot) / 3600:5.2f} h   7b {secs(s7) / 3600:5.2f} h")
    TM.update(base)
    tot = res['total']
    print(f"  worksheet values {tot['cmp_n']} (+ scratch {tot['scratch']}); compare at {CMP} s each = {tot['cmp_sec'] / 60:.1f} min")
    print(f"  checks total {tot['chk_sec'] / 3600:.2f} h")
    H = 20.0
    deriv, reserve = 2.5, 1.5
    a_h, c_h, m_h = tot['sec'] / 3600, tot['chk_sec'] / 3600, tot['cmp_sec'] / 3600
    print(f"  20h = arith {a_h:.2f} + checks {c_h:.2f} + compare {m_h:.2f} + derivations {deriv} + reserve {reserve}"
          f" + rest {H - a_h - c_h - m_h - deriv - reserve:.2f}")
    allp = tot['all_mul'] + tot['struct']
    print(f"  products in replay+struct {allp}: struct {tot['struct']} ({tot['struct'] / allp:.0%}), zero {tot['skip']} "
          f"({tot['skip'] / allp:.0%}), no-calc total {(tot['struct'] + tot['skip']) / allp:.0%}; nontrivial {tot['nontriv_mul']}")
    g1, g2, gb = res['groups']['forward1'], res['groups']['forward2'], res['groups']['backward']
    print(f"  fwd2/fwd1 time {g2['sec'] / g1['sec']:.2f}x ; bwd/fwd1 nontrivial {gb['nontriv_mul'] / g1['nontriv_mul']:.2f}x, "
          f"time {gb['sec'] / g1['sec']:.2f}x ; dense bwd/fwd {res['dense']['backward_total'] / res['dense']['forward_total']:.2f}x")
    st = res['steps']
    left = lambda sid, bud: bud - (st[sid]['sec'] + st[sid]['chk_sec'] + st[sid]['cmp_sec']) / 3600
    print(f"  option A (move 0.75 h step3 -> step7: 3=1.75, 7a=1.00, 7b=3.75): left step3 {left('3', 1.75):.2f}, "
          f"7a {left('7a', 1.0):.2f}, 7b {left('7b', 3.75):.2f}")
    b7 = 3.25 - (st['7b']['sec'] + st['7b']['cmp_sec']) / 3600
    print(f"  option B (no checksums in 7.3, compare with answers.md only): 7b left {b7:.2f} h ; 7a over {left('7a', 0.75):.2f} h from reserve")
    c73 = res['checkpoints']['7.3']
    print(f"  7.3 alone {c73['sec'] / 60:.1f} min = {c73['sec'] / tot['sec']:.0%} of all arithmetic; "
          f"3.1 first pass {res['checkpoints']['3.1']['sec'] / 60:.1f} min vs second pass {res['pass2']['7.3/3.1']['sec'] / 60:.1f} min")
    for lab, sids in (('fwd steps 2-5', ['2', '3', '4', '5']), ('bwd 6a-6e', ['6a', '6b', '6c', '6d', '6e']), ('step7', ['7a', '7b'])):
        a_ = sum(st[x]['sec'] for x in sids) / 3600
        c_ = sum(st[x]['chk_sec'] for x in sids) / 3600
        m_ = sum(st[x]['cmp_sec'] for x in sids) / 3600
        b_ = sum(st[x]['budget_h'] for x in sids)
        print(f"  {lab:14s} arith {a_:.2f}  +check {c_:.2f}  +compare {m_:.2f}  = {a_ + c_ + m_:.2f}  budget {b_:.2f}")
    print(f"  7.3 answer-table cells compared: {cmp_n('7.3')} ; exp(0) skipped {tot['exp_triv']}, trivial sqrt {tot['sqrt_triv']}")
    ncp = len([x for x in cps if x != 'prep'])
    print(f"  error hunt worst case: {ncp} checkpoints x 10 min = {ncp * 10 / 60:.2f} h")
    print()
    print('per step: arith / check / compare / sum / budget / left')
    for sid, s in res['steps'].items():
        a_, c_, m_ = s['sec'] / 3600, s['chk_sec'] / 3600, s['cmp_sec'] / 3600
        print(f"  {sid:3s} {a_:5.2f} {c_:5.2f} {m_:5.2f} {a_ + c_ + m_:5.2f} {s['budget_h']:5.2f} {s['budget_h'] - a_ - c_ - m_:6.2f}")

    if a.md:
        emit_md(res, cps, p, a.md)


def special_str(s):
    parts = []
    for key, sym in (('exp', 'exp'), ('log', 'ln'), ('sqrt', '√'), ('div_dec', '÷'), ('div_nice', '÷c'), ('scale', '/√2')):
        if s[key]:
            parts.append(f"{sym} {s[key]}")
    return '、'.join(parts) if parts else '–'


def emit_md(res, cps, p, path):
    titles = {cid: t for cid, t, _ in M.checkpoints(*M.load_params())}
    titles['prep'] = '準備：重み行列の行和・列和'
    L = []
    L.append('| CP | 内容 | 0 | ±1 | n×n | n×d | d×d | 特殊演算 | 加減 | 記録 | 構造スキップ | 分 |')
    L.append('|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|')
    for cid in cps:
        s = res['checkpoints'][cid]
        L.append(f"| {cid} | {titles[cid]} | {s['skip']} | {s['sign']} | {s['nn']} | {s['nd']} | {s['dd']} | {special_str(s)} | "
                 f"{s['add']} | {s['write']} | {s['struct']} | {s['sec'] / 60:.1f} |")
    L.append('')
    L.append('| ステップ | CP | 非自明な積 | うち d×d | ① 計算 | ② 検算 | ③ 照合 | ①+②+③ | 予算 | 残り（理解・誤り探し） |')
    L.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    tot_b = 0.0
    for sid, s in res['steps'].items():
        a_, c_, m_ = s['sec'] / 3600, s['chk_sec'] / 3600, s['cmp_sec'] / 3600
        ids = s['checkpoints']
        rng = ids[0] if len(ids) == 1 else f"{ids[0]}–{ids[-1]}"
        left = s['budget_h'] - a_ - c_ - m_
        lft = f"**{left:.2f}**" if left < 0 else f"{left:.2f}"
        L.append(f"| {s['label']} | {rng} | {s['nontriv_mul']} | {s['dd']} | {a_:.2f} | {c_:.2f} | {m_:.2f} | {a_ + c_ + m_:.2f} | "
                 f"{s['budget_h']:.2f} | {lft} |")
        tot_b += s['budget_h']
    s = res['total']
    a_, c_, m_ = s['sec'] / 3600, s['chk_sec'] / 3600, s['cmp_sec'] / 3600
    L.append(f"| **計（導出・予備を除く）** | | **{s['nontriv_mul']}** | {s['dd']} | **{a_:.2f}** | {c_:.2f} | {m_:.2f} | {a_ + c_ + m_:.2f} | "
             f"{tot_b:.2f} | {tot_b - a_ - c_ - m_:.2f} |")
    L.append('')
    dn = res['dense']
    L.append('| 区間 | 密な積の数 | 構造スキップ | 0 との積 | 非自明な積 | うち d×d | 計算時間 |')
    L.append('|---|---:|---:|---:|---:|---:|---:|')
    for gname, lab, dval in (('forward1', '順伝播 2.1–5.3', dn['forward_total']), ('backward', '逆伝播 6.1–6.9', dn['backward_total']),
                             ('update', '更新 7.1–7.2', None), ('forward2', '再順伝播 7.3–7.4', dn['forward2_total'])):
        s = res['groups'][gname]
        dv = f"{dval}" if dval else '180 (lr·g) + 180²'
        L.append(f"| {lab} | {dv} | {s['struct']} | {s['skip']} | {s['nontriv_mul']} | {s['dd']} | {s['sec'] / 3600:.2f} h |")
    s = res['total']
    L.append(f"| **合計** | | {s['struct']} | {s['skip']} | **{s['nontriv_mul']}** | {s['dd']} | **{s['sec'] / 3600:.2f} h** |")
    L.append('')
    parts = []
    names = {'2.1': 'X0′', '2.2': 'LN1′統計', '2.3': 'U1′', '3.1': 'Q′K′V′', '3.2': 'S′', '3.3': 'A′', '3.4': 'O′', '3.5': 'Attn′',
             '4.1': 'X1′', '4.2': 'LN2′', '4.3': 'Z′', '4.4': 'F′', '4.5': 'X2′', '5.1': 'logits′', '5.2': 'p′', '5.3': 'L′'}
    for k, s in res['pass2'].items():
        parts.append(f"{names[k.split('/')[1]]} {s['sec'] / 60:.0f}")
    L.append('7.3 内訳（分）: ' + ' / '.join(parts))
    L.append('')
    L.append('| 検算（行和チェックサム＋不変量） | 積（非自明） | うち d×d | 加減 | 時間 |')
    L.append('|---|---:|---:|---:|---:|')
    for k, lab in (('forward1', '順伝播'), ('backward', '逆伝播（dX と dW）'), ('forward2', '再順伝播（W′ の行和の計算を含む）')):
        s = res['checks'][k]
        L.append(f"| {lab} | {s['nontriv_mul']} | {s['dd']} | {s['add']} | {s['sec'] / 60:.0f} 分 |")
    Path(path).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('wrote', path)


if __name__ == '__main__':
    main()
