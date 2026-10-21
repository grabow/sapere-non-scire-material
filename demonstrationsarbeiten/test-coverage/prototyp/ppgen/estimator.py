"""Analytische Vorhersage der Zweigueberdeckung zufaelliger Tests.

Verfahren: probabilistische abstrakte Interpretation (vgl. Kapitel 4.5).
  * Jede Programmvariable wird als lineare Form ueber unabhaengigen
    Basiszufallsvariablen dargestellt (Parameter, nichtlineare Teilausdruecke).
  * Jede Basisvariable besitzt eine explizite Wahrscheinlichkeitsfunktion (PMF).
  * Bedingungen werden exakt auf der Verteilung der linearen Form ausgewertet;
    an Verzweigungen werden die Randverteilungen exakt bedingt
    (Mean-Field-Naeherung fuer die gemeinsame Verteilung).
  * Schleifen werden symbolisch abgerollt; Trefferwahrscheinlichkeiten ueber
    Iterationen werden je nach Bedingungstyp per Maximum oder Summe kombiniert.
Ergebnis ist fuer jeden Zweig b die Wahrscheinlichkeit p_b, dass ein einzelner
Zufallstest b erreicht. Fuer N Tests gilt  E[C(N)] = mean_b 1 - (1 - p_b)^N.
"""
from __future__ import annotations

import ast
import itertools
from collections import defaultdict

import numpy as np
from scipy.signal import fftconvolve

MAX_UNROLL = 40
EPS = 1e-12
_counter = itertools.count()
ORIGIN: dict = {}          # Herkunft nichtlinearer Basisvariablen (fuer kanonische Signaturen)


def canon(lf: "LF"):
    """Kanonische Darstellung einer linearen Form (Basisnamen durch Herkunft ersetzt)."""
    return (tuple(sorted(((repr(ORIGIN.get(b, b)), k) for b, k in lf.co.items()))), lf.c)


# ---------------------------------------------------------------------------
# Diskrete Verteilungen
# ---------------------------------------------------------------------------
class PMF:
    """Wahrscheinlichkeitsfunktion auf den ganzen Zahlen lo .. lo+len(p)-1."""
    __slots__ = ("lo", "p", "_c")

    def __init__(self, lo: int, p):
        p = np.asarray(p, dtype=float)
        p = np.where(p > 1e-15, p, 0.0)
        nz = np.flatnonzero(p)
        if len(nz) == 0:
            raise ZeroDivisionError("leere Verteilung")
        p = p[nz[0]: nz[-1] + 1]
        self.lo = int(lo) + int(nz[0])
        self.p = p / p.sum()
        self._c = None

    @staticmethod
    def uniform(lo: int, hi: int) -> "PMF":
        return PMF(lo, np.full(hi - lo + 1, 1.0 / (hi - lo + 1)))

    @staticmethod
    def const(c: int) -> "PMF":
        return PMF(c, [1.0])

    @property
    def hi(self) -> int:
        return self.lo + len(self.p) - 1

    def values(self) -> np.ndarray:
        return np.arange(self.lo, self.hi + 1)

    def mean(self) -> float:
        return float(np.dot(self.values(), self.p))

    def cdf(self) -> np.ndarray:
        if self._c is None:
            self._c = np.cumsum(self.p)
        return self._c

    def p_lt(self, t):
        """P(X < t), elementweise fuer ganzzahlige t."""
        k = np.asarray(t, dtype=np.int64) - self.lo - 1
        c = self.cdf()
        out = c[np.clip(k, 0, len(c) - 1)]
        return np.where(k < 0, 0.0, np.where(k >= len(c), 1.0, out))

    def p_eq(self, t):
        k = np.asarray(t, dtype=np.int64) - self.lo
        inside = (k >= 0) & (k < len(self.p))
        return np.where(inside, self.p[np.clip(k, 0, len(self.p) - 1)], 0.0)

    def shift(self, c: int) -> "PMF":
        q = PMF.__new__(PMF)
        q.lo, q.p, q._c = self.lo + c, self.p, self._c
        return q

    def scale(self, k: int) -> "PMF":
        if k == 0:
            return PMF.const(0)
        if k == 1:
            return self
        p = self.p if k > 0 else self.p[::-1]
        a = abs(k)
        q = np.zeros(a * (len(p) - 1) + 1)
        q[::a] = p
        return PMF(k * (self.lo if k > 0 else self.hi), q)

    def add(self, other: "PMF") -> "PMF":
        if len(self.p) == 1:
            return other.shift(self.lo)
        if len(other.p) == 1:
            return self.shift(other.lo)
        if len(self.p) * len(other.p) > 200_000:
            conv = fftconvolve(self.p, other.p)
        else:
            conv = np.convolve(self.p, other.p)
        return PMF(self.lo + other.lo, conv)

    def residues(self, m: int) -> np.ndarray:
        return np.bincount(self.values() % m, weights=self.p, minlength=m)[:m]

    def map(self, fn) -> "PMF":
        v = fn(self.values())
        lo = int(v.min())
        return PMF(lo, np.bincount(v - lo, weights=self.p))

    def reweight(self, w: np.ndarray) -> "PMF | None":
        q = self.p * w
        if q.sum() <= EPS:
            return None
        return PMF(self.lo, q)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.choice(self.values(), size=n, p=self.p)

    @staticmethod
    def mixture(items: list) -> "PMF":
        items = [(d, w) for d, w in items if w > EPS]
        if len(items) == 1:
            return items[0][0]
        lo = min(d.lo for d, _ in items)
        hi = max(d.hi for d, _ in items)
        acc = np.zeros(hi - lo + 1)
        for d, w in items:
            acc[d.lo - lo: d.hi - lo + 1] += w * d.p
        return PMF(lo, acc)


def prob_op(d: PMF, op: str, t):
    """P(d op t) elementweise."""
    if op == "<":
        return d.p_lt(t)
    if op == "<=":
        return d.p_lt(np.asarray(t) + 1)
    if op == ">":
        return 1.0 - d.p_lt(np.asarray(t) + 1)
    if op == ">=":
        return 1.0 - d.p_lt(t)
    if op == "==":
        return d.p_eq(t)
    if op == "!=":
        return 1.0 - d.p_eq(t)
    raise ValueError(op)


# ---------------------------------------------------------------------------
# Lineare Formen ueber Basisvariablen
# ---------------------------------------------------------------------------
class LF:
    __slots__ = ("co", "c")

    def __init__(self, co: dict | None = None, c: int = 0):
        self.co = {k: v for k, v in (co or {}).items() if v != 0}
        self.c = c

    def __add__(self, o: "LF") -> "LF":
        co = dict(self.co)
        for k, v in o.co.items():
            co[k] = co.get(k, 0) + v
        return LF(co, self.c + o.c)

    def scale(self, k: int) -> "LF":
        return LF({b: v * k for b, v in self.co.items()}, self.c * k)

    def __sub__(self, o: "LF") -> "LF":
        return self + o.scale(-1)

    def is_const(self) -> bool:
        return not self.co

    def key(self):
        return (tuple(sorted(self.co.items())), self.c)


OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}


class Env:
    __slots__ = ("vars", "bases")

    def __init__(self, vars_=None, bases=None):
        self.vars: dict[str, LF] = vars_ or {}
        self.bases: dict[str, PMF] = bases or {}

    def copy(self) -> "Env":
        return Env(dict(self.vars), dict(self.bases))

    def new_base(self, d: PMF, origin=None) -> LF:
        name = f"#{next(_counter)}"
        self.bases[name] = d
        if origin is not None:
            ORIGIN[name] = origin
        return LF({name: 1})

    def dist(self, lf: LF) -> PMF:
        d = PMF.const(lf.c)
        for b, k in sorted(lf.co.items(), key=lambda kv: len(self.bases[kv[0]].p)):
            d = d.add(self.bases[b].scale(k))
        return d

    @staticmethod
    def mixture(items: list) -> "Env | None":
        items = [(e, w) for e, w in items if e is not None and w > EPS]
        if not items:
            return None
        if len(items) == 1:
            return items[0][0]
        first = items[0][0]
        out = Env({}, {})
        tot = sum(w for _, w in items)
        # Programmvariablen: gleiche lineare Form -> uebernehmen, sonst neue Basis
        names = set().union(*(e.vars.keys() for e, _ in items))
        base_names = set().union(*(e.bases.keys() for e, _ in items))
        for b in base_names:
            ds = [(e.bases[b], w) for e, w in items if b in e.bases]
            if all(d is ds[0][0] for d, _ in ds):
                out.bases[b] = ds[0][0]
            else:
                out.bases[b] = PMF.mixture([(d, w / tot) for d, w in ds])
        for n in names:
            lfs = [(e, e.vars[n], w) for e, w in items if n in e.vars]
            keys = {lf.key() for _, lf, _ in lfs}
            if len(keys) == 1:
                out.vars[n] = lfs[0][1]
            else:
                d = PMF.mixture([(e.dist(lf), w / tot) for e, lf, w in lfs])
                out.vars[n] = out.new_base(d)
        return out


# ---------------------------------------------------------------------------
# Ausdruecke und Bedingungen
# ---------------------------------------------------------------------------
class Analyzer:
    def __init__(self, domains: dict[str, tuple[int, int]], default_domain=(-1000, 1000)):
        self.domains = domains
        self.default_domain = default_domain
        self.rng = np.random.default_rng(0)
        self.result: dict = {}
        self.acc_stack: list = []

    # -- Ausdruecke ---------------------------------------------------------
    def unknown(self, env: Env) -> LF:
        return env.new_base(PMF.uniform(*self.default_domain))

    def eval(self, node: ast.AST, env: Env) -> LF:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int)):
                return LF({}, int(node.value))
            return self.unknown(env)
        if isinstance(node, ast.Name):
            if node.id in env.vars:
                return env.vars[node.id]
            return self.unknown(env)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand, env)
            if isinstance(node.op, ast.USub):
                return v.scale(-1)
            if isinstance(node.op, ast.UAdd):
                return v
            p, _, _ = self.cond(node, env)
            return env.new_base(PMF(0, [1 - p, p]))
        if isinstance(node, ast.BinOp):
            x, y = self.eval(node.left, env), self.eval(node.right, env)
            if isinstance(node.op, ast.Add):
                return x + y
            if isinstance(node.op, ast.Sub):
                return x - y
            if isinstance(node.op, ast.Mult):
                if y.is_const():
                    return x.scale(y.c)
                if x.is_const():
                    return y.scale(x.c)
            if isinstance(node.op, ast.Mod) and y.is_const() and y.c > 0:
                if x.is_const():
                    return LF({}, x.c % y.c)
                return env.new_base(PMF(0, env.dist(x).residues(y.c)), ("mod", canon(x), y.c))
            if isinstance(node.op, ast.FloorDiv) and y.is_const() and y.c != 0:
                k = y.c
                return env.new_base(env.dist(x).map(lambda v: v // k), ("div", canon(x), k))
            return self.sample_binop(node.op, x, y, env)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self.eval(a, env) for a in node.args]
            fn = node.func.id
            if fn == "abs" and len(args) == 1:
                return env.new_base(env.dist(args[0]).map(np.abs), ("abs", canon(args[0])))
            if fn in ("min", "max") and args:
                f = np.minimum if fn == "min" else np.maximum
                xs = [env.dist(a).sample(self.rng, 20000) for a in args]
                v = xs[0]
                for w in xs[1:]:
                    v = f(v, w)
                return env.new_base(self._from_samples(v))
            if fn == "int" and len(args) == 1:
                return args[0]
            return self.unknown(env)
        if isinstance(node, (ast.Compare, ast.BoolOp)):
            p, _, _ = self.cond(node, env)
            return env.new_base(PMF(0, [1 - p, p]))
        return self.unknown(env)

    @staticmethod
    def _from_samples(v: np.ndarray) -> PMF:
        lo = int(v.min())
        return PMF(lo, np.bincount(v - lo).astype(float))

    def sample_binop(self, op, x: LF, y: LF, env: Env) -> LF:
        """Nichtlineare Operationen: Verteilung per Stichprobe der Randverteilungen."""
        xs = env.dist(x).sample(self.rng, 20000)
        ys = env.dist(y).sample(self.rng, 20000)
        with np.errstate(all="ignore"):
            if isinstance(op, ast.Mult):
                v = xs * ys
            elif isinstance(op, (ast.Mod, ast.FloorDiv)):
                ok = ys != 0
                xs, ys = xs[ok], ys[ok]
                if len(xs) == 0:
                    return self.unknown(env)
                v = xs % ys if isinstance(op, ast.Mod) else xs // ys
            else:
                return self.unknown(env)
        return env.new_base(self._from_samples(v.astype(np.int64)))

    # -- Bedingungen --------------------------------------------------------
    def atom(self, env: Env, lf: LF, op: str, mod: tuple | None = None):
        """Bedingung  lf op 0  bzw.  (lf % m) op r.  Liefert (p, envT, envF)."""
        if mod is None:
            d = env.dist(lf)
            p = float(prob_op(d, op, 0))
        else:
            m, r = mod
            res = env.dist(lf).residues(m)
            p = float(res[r % m]) if op == "==" else float(1 - res[r % m])
        p = min(max(p, 0.0), 1.0)
        envT = env.copy() if p > EPS else None
        envF = env.copy() if p < 1 - EPS else None
        if p <= EPS or p >= 1 - EPS:
            return p, envT, envF
        for b, k in lf.co.items():
            rest = env.dist(lf - LF({b: k}))
            bd = env.bases[b]
            t = k * bd.values()
            if mod is None:
                pc = prob_op(rest, op, -t)
            else:
                m, r = mod
                rr = rest.residues(m)
                pc = rr[(r - t) % m]
                if op == "!=":
                    pc = 1 - pc
            dT, dF = bd.reweight(pc), bd.reweight(1 - pc)
            if dT is not None:
                envT.bases[b] = dT
            if dF is not None:
                envF.bases[b] = dF
        return p, envT, envF

    def compare(self, left, op_node, right, env: Env):
        op = OPS.get(type(op_node))
        if op is None:
            return 0.5, env.copy(), env.copy()
        # Spezialfall Modulo-Bedingung: (X % m) ==/!= r
        for a, b, o in ((left, right, op), (right, left, FLIP[op])):
            if (o in ("==", "!=") and isinstance(a, ast.BinOp) and isinstance(a.op, ast.Mod)):
                m = self.eval(a.right, env)
                r = self.eval(b, env)
                if m.is_const() and m.c > 0 and r.is_const():
                    x = self.eval(a.left, env)
                    if not 0 <= r.c < m.c:
                        p = 0.0 if o == "==" else 1.0
                        return p, (env.copy() if p else None), (env.copy() if p < 1 else None)
                    return self.atom(env, x, o, (m.c, r.c))
        lf = self.eval(left, env) - self.eval(right, env)
        return self.atom(env, lf, op)

    def cond(self, node: ast.AST, env: Env):
        if isinstance(node, ast.BoolOp):
            vals = node.values
            p1, T1, F1 = self.cond(vals[0], env)
            rest = vals[1] if len(vals) == 2 else ast.BoolOp(op=node.op, values=vals[1:])
            if isinstance(node.op, ast.And):
                if T1 is None:
                    return 0.0, None, F1
                p2, T2, F2 = self.cond(rest, T1)
                return p1 * p2, T2, Env.mixture([(F1, 1 - p1), (F2, p1 * (1 - p2))])
            if F1 is None:
                return 1.0, T1, None
            p2, T2, F2 = self.cond(rest, F1)
            return p1 + (1 - p1) * p2, Env.mixture([(T1, p1), (T2, (1 - p1) * p2)]), F2
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            p, T, F = self.cond(node.operand, env)
            return 1 - p, F, T
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1:
                return self.compare(node.left, node.ops[0], node.comparators[0], env)
            pairs = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                pairs.append(ast.Compare(left=left, ops=[op], comparators=[right]))
                left = right
            return self.cond(ast.BoolOp(op=ast.And(), values=pairs), env)
        if isinstance(node, ast.Constant):
            p = 1.0 if node.value else 0.0
            return p, (env.copy() if p else None), (env.copy() if not p else None)
        return self.atom(env, self.eval(node, env), "!=")

    # -- Aufzeichnung von Zweigwahrscheinlichkeiten -------------------------
    def record(self, key, p: float, sig=None) -> None:
        if self.acc_stack:
            it = self.acc_stack[-1]["iters"][-1]
            it[key] = (max(p, it.get(key, (0.0,))[0]), sig)
        else:
            self.result[key] = max(p, self.result.get(key, 0.0))

    def eqmod_signature(self, test: ast.AST, env: Env, outcome: bool):
        """Signatur eines Zweigausgangs fuer die Kombination ueber Iterationen.

        eq:  Gleichheits-/Modulo-Atome, die der Ausgang als Gleichheit erzwingt
             (aendern sie sich, sind die Ereignisse je Iteration ~ disjunkt);
        mod: uebrige Modulo-Atome (aendern sie sich: ~ unabhaengig).
        Schwellwertvergleiche erzeugen geschachtelte Ereignisse (Maximum)."""
        env = env.copy()
        eq, mod = [], []

        def visit(node, truth, forced):
            if isinstance(node, ast.BoolOp):
                conj = isinstance(node.op, ast.And)
                for v in node.values:
                    visit(v, truth, forced and conj == truth)
                return
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                visit(node.operand, not truth, forced)
                return
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                op = node.ops[0]
                left, right = node.left, node.comparators[0]
                is_eq = isinstance(op, (ast.Eq, ast.NotEq))
                equality = forced and is_eq and isinstance(op, ast.Eq) == truth
                for a, b in ((left, right), (right, left)):
                    if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Mod):
                        dividend = canon(self.eval(a.left, env))
                        if equality:
                            eq.append(("mod", dividend, canon(self.eval(a.right, env)),
                                       canon(self.eval(b, env))))
                        else:           # nur der Dividend bestimmt die Abhaengigkeit
                            mod.append(("mod", dividend))
                        return
                if equality:
                    eq.append(("eq", canon(self.eval(left, env) - self.eval(right, env))))

        visit(test, outcome, True)
        return (tuple(eq), tuple(mod))

    # -- Anweisungen ----------------------------------------------------------
    def block(self, stmts: list, env: Env, reach: float):
        cont = 1.0
        for s in stmts:
            if env is None or reach * cont <= EPS:
                return env, 0.0 if env is None else cont
            if isinstance(s, ast.Assign):
                vals = self.eval(s.value, env)
                tgt = s.targets[0]
                if isinstance(tgt, ast.Name):
                    env = env.copy()
                    env.vars[tgt.id] = vals
                elif isinstance(tgt, ast.Tuple) and isinstance(s.value, ast.Tuple):
                    lfs = [self.eval(v, env) for v in s.value.elts]
                    env = env.copy()
                    for t, lf in zip(tgt.elts, lfs):
                        if isinstance(t, ast.Name):
                            env.vars[t.id] = lf
            elif isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
                v = self.eval(ast.BinOp(left=ast.Name(id=s.target.id, ctx=ast.Load()),
                                        op=s.op, right=s.value), env)
                env = env.copy()
                env.vars[s.target.id] = v
            elif isinstance(s, ast.If):
                env, c = self.if_stmt(s, env, reach * cont)
                cont *= c
            elif isinstance(s, (ast.For, ast.While)):
                env, c = self.loop(s, env, reach * cont)
                cont *= c
            elif isinstance(s, ast.Return):
                return env, 0.0
        return env, cont

    def if_stmt(self, node: ast.If, env: Env, reach: float):
        p, T, F = self.cond(node.test, env)
        key = (node.lineno, node.col_offset)
        if self.acc_stack:
            sigT = self.eqmod_signature(node.test, env, True)
            sigF = self.eqmod_signature(node.test, env, False)
        outT, cT = self.block(node.body, T, reach * p) if T is not None else (None, 0.0)
        outF, cF = self.block(node.orelse, F, reach * (1 - p)) if F is not None else (None, 0.0)
        if self.acc_stack:
            # endet ein Zweig sicher mit return, schliessen sich Treffer in
            # verschiedenen Iterationen gegenseitig aus
            it = len(self.acc_stack[-1]["iters"])
            if T is not None and cT <= EPS:
                sigT = (("ret", it),) + sigT[1:]
            if F is not None and cF <= EPS:
                sigF = (("ret", it),) + sigF[1:]
            self.record(key + ("T",), reach * p, sigT)
            self.record(key + ("F",), reach * (1 - p), sigF)
        else:
            self.record(key + ("T",), reach * p)
            self.record(key + ("F",), reach * (1 - p))
        wT, wF = p * cT, (1 - p) * cF
        return Env.mixture([(outT, wT), (outF, wF)]), wT + wF

    @staticmethod
    def assigned_names(stmts: list) -> set:
        out = set()
        for s in stmts:
            for n in ast.walk(s):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
        return out

    def loop(self, node, env: Env, reach: float):
        key = (node.lineno, node.col_offset)
        variant = self.assigned_names(node.body)
        is_for = isinstance(node, ast.For)
        if is_for:
            variant |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
            rng_args = None
            it = node.iter
            if (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                    and it.func.id == "range" and 1 <= len(it.args) <= 3):
                args = [self.eval(a, env) for a in it.args]
                if len(args) == 1:
                    args = [LF({}, 0), args[0], LF({}, 1)]
                elif len(args) == 2:
                    args = [args[0], args[1], LF({}, 1)]
                if args[2].is_const() and args[2].c != 0:
                    rng_args = args
        self.acc_stack.append({"iters": []})
        cur, cur_reach = env, reach
        exits = []
        for j in range(MAX_UNROLL):
            if cur is None or cur_reach <= EPS:
                break
            if is_for:
                if rng_args is None:      # unbekannter Iterator: eine Iteration annehmen
                    p, T, F = (1.0, cur.copy(), None) if j == 0 else (0.0, None, cur.copy())
                    if T is not None:
                        T.vars[node.target.id] = self.unknown(T)
                else:
                    start, stop, step = rng_args
                    ival = start + LF({}, j * step.c)
                    diff = stop - ival if step.c > 0 else ival - stop
                    p, T, F = self.atom(cur, diff, ">")
                    if T is not None and isinstance(node.target, ast.Name):
                        T.vars[node.target.id] = ival
            else:
                p, T, F = self.cond(node.test, cur)
            exits.append((F, cur_reach * (1 - p)))
            if T is None:
                break
            self.acc_stack[-1]["iters"].append({})
            self.record(key + ("E",), cur_reach * p)
            out, c = self.block(node.body, T, cur_reach * p)
            cur, cur_reach = out, cur_reach * p * c
        else:
            exits.append((cur, cur_reach))
        acc = self.acc_stack.pop()
        self._combine(acc)
        p_exit = sum(w for _, w in exits)
        self.record(key + ("X",), p_exit)
        env_out = Env.mixture([(e, w) for e, w in exits])
        return env_out, (p_exit / reach if reach > EPS else 0.0)

    def _combine(self, acc: dict) -> None:
        per_key: dict = defaultdict(list)
        sigs: dict = defaultdict(list)
        for it in acc["iters"]:
            for k, (p, sig) in it.items():
                per_key[k].append(p)
                sigs[k].append(sig)
        for k, ps in per_key.items():
            ss = [x if x is not None else ((), ()) for x in sigs[k]]
            if len({x[0] for x in ss}) > 1:
                p = min(1.0, sum(ps))                  # ~ disjunkte Ereignisse
            elif len({x[1] for x in ss}) > 1:
                p = 1.0 - float(np.prod([1.0 - q for q in ps]))   # ~ unabhaengige Ereignisse
            else:
                p = max(ps)                            # gleiche bzw. geschachtelte Ereignisse
            self.record(k, p, (tuple(x[0] for x in ss), tuple(x[1] for x in ss)))


def branch_keys(source: str) -> list:
    """Alle Zweigausgaenge in derselben Kodierung wie die Instrumentierung."""
    keys = []
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    for n in ast.walk(fn):
        if isinstance(n, ast.If):
            keys += [(n.lineno, n.col_offset, "T"), (n.lineno, n.col_offset, "F")]
        elif isinstance(n, (ast.For, ast.While)):
            keys += [(n.lineno, n.col_offset, "E"), (n.lineno, n.col_offset, "X")]
    return keys


def branch_probabilities(source: str, domain=None, domains: dict | None = None) -> dict:
    """p_b je Zweig fuer gleichverteilte Eingaben."""
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef))
    params = [a.arg for a in fn.args.args]
    doms = {p: (-domain, domain) for p in params} if domain is not None else {}
    doms.update(domains or {})
    an = Analyzer(doms, default_domain=(-(domain or 1000), (domain or 1000)))
    env = Env()
    for p in params:
        lo, hi = doms[p]
        env.vars[p] = env.new_base(PMF.uniform(lo, hi))
    an.block(fn.body, env, 1.0)
    return {k: an.result.get(k, 0.0) for k in branch_keys(source)}


def expected_coverage(probs: dict, budgets) -> np.ndarray:
    p = np.array(list(probs.values()))
    if len(p) == 0:
        return np.ones(len(budgets))
    p = np.clip(p, 0.0, 1.0)
    return np.array([np.mean(1.0 - (1.0 - p) ** n) for n in budgets])
