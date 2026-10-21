"""Konstruktiver Generator fuer Python-Beispielfunktionen mit definierten Eigenschaften.

Ablauf (vgl. Kapitel 4):
  1. Budgetverteilung: Die Entscheidungsbudgets (cc - 1) werden auf Schleifen,
     if-Ketten, elif-Zweige und boolesche Operatoren verteilt.
  2. Kontrollbaum: Ein "Rueckgrat" aus max_depth geschachtelten Knoten garantiert
     die exakte Schachtelungstiefe, die restlichen Knoten werden zufaellig
     auf zulaessige Bloecke verteilt.
  3. Rendering: Beim Erzeugen des Quelltexts wird eine Intervall-Umgebung
     mitgefuehrt. Konstanten in Bedingungen werden so gewaehlt, dass beide
     Ausgaenge unter den bisherigen Pfadbedingungen erfuellbar bleiben.
"""
from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field

from .spec import Spec

PARAM_NAMES = ["a", "b", "c", "d", "e"]
LOOP_VARS = ["i", "j", "k", "m"]
MAX_ITER = 8           # obere Schranke der Iterationen von while-Schleifen
BIG = 10 ** 9


# ---------------------------------------------------------------------------
# Intervall-Umgebung
# ---------------------------------------------------------------------------
@dataclass
class Var:
    name: str
    lo: int = -BIG
    hi: int = BIG
    expr: tuple | None = None   # ('aff', src, m, o) | ('add', x, y) | ('sub', x, y) | ('mod', x, m)
    holes: frozenset = frozenset()   # ausgeschlossene Werte (nur Basisvariablen)
    res: tuple = ()                  # Restklassen ((m, erlaubte Reste), ...) (nur Basisvariablen)


SCAN = 4000   # maximale Suchschritte nach zulaessigen Werten


class Env:
    """Abbildung Variable -> Wertebereich. Basisvariablen (Parameter, Schleifenvariablen)
    tragen Intervall, ausgeschlossene Werte und Restklassen; abgeleitete Variablen werden
    aus ihren Definitionen berechnet."""

    def __init__(self, vars_: dict[str, Var] | None = None):
        self.v: dict[str, Var] = vars_ or {}

    def copy(self) -> "Env":
        return Env({k: dataclasses.replace(x) for k, x in self.v.items()})

    def add(self, var: Var) -> None:
        self.v[var.name] = var

    def names(self) -> list[str]:
        return list(self.v.keys())

    # -- Intervalle ------------------------------------------------------
    def interval(self, name: str) -> tuple[int, int]:
        x = self.v[name]
        lo, hi = x.lo, x.hi
        if x.expr is not None:
            elo, ehi = self._expr_interval(x.expr)
            lo, hi = max(lo, elo), min(hi, ehi)
            if lo > hi:            # unpraezise Verfeinerung: Klammerung verwerfen
                lo, hi = elo, ehi
        if lo > hi:
            lo = hi = x.hi if x.hi < BIG else x.lo
        return lo, hi

    def width(self, name: str) -> int:
        lo, hi = self.interval(name)
        return hi - lo + 1

    def _expr_interval(self, e: tuple) -> tuple[int, int]:
        kind = e[0]
        if kind == "aff":
            _, s, m, o = e
            slo, shi = self.interval(s)
            a, b = slo * m + o, shi * m + o
            return min(a, b), max(a, b)
        if kind == "add":
            (xlo, xhi), (ylo, yhi) = self.interval(e[1]), self.interval(e[2])
            return xlo + ylo, xhi + yhi
        if kind == "sub":
            (xlo, xhi), (ylo, yhi) = self.interval(e[1]), self.interval(e[2])
            return xlo - yhi, xhi - ylo
        if kind == "mod":
            _, s, m = e
            res = self.residues(s, m)
            return (min(res), max(res)) if res else (0, m - 1)
        raise ValueError(kind)

    def roots(self, name: str) -> set[str]:
        """Basisvariablen (Parameter/Schleifenvariablen), von denen `name` abhaengt."""
        x = self.v[name]
        if x.expr is None:
            return {name}
        if x.expr[0] in ("aff", "mod"):
            return self.roots(x.expr[1])
        return self.roots(x.expr[1]) | self.roots(x.expr[2])

    # -- zulaessige Werte (Basis bzw. affin von Basis) ----------------------
    def affine_base(self, name: str):
        x = self.v[name]
        if x.expr is None:
            return name, 1, 0
        if x.expr[0] == "aff" and self.v[x.expr[1]].expr is None:
            return x.expr[1], x.expr[2], x.expr[3]
        return None

    def _valid(self, s: str, t: int) -> bool:
        b = self.v[s]
        return t not in b.holes and all(t % m in allowed for m, allowed in b.res)

    def _scan(self, s: str, start: int, stop: int, step: int):
        t, n = start, 0
        while (t <= stop if step > 0 else t >= stop) and n < SCAN:
            if self._valid(s, t):
                return t
            t += step
            n += 1
        return None

    def extremes(self, name: str):
        """Kleinster und groesster tatsaechlich annehmbarer Wert (oder None)."""
        ab = self.affine_base(name)
        if ab is None:
            lo, hi = self.interval(name)
            return lo, hi
        s, m, o = ab
        lo, hi = self.interval(name)
        slo, shi = self.interval(s)
        a, b = self._scan(s, slo, shi, 1), self._scan(s, shi, slo, -1)
        if a is None or b is None:
            return None
        vals = sorted((m * a + o, m * b + o))
        return max(vals[0], lo), min(vals[1], hi)

    def usable(self, name: str) -> bool:
        ex = self.extremes(name)
        return ex is not None and ex[0] < ex[1]

    def random_value(self, name: str, rng: random.Random) -> int:
        ab = self.affine_base(name)
        lo, hi = self.interval(name)
        if ab is None:
            return rng.randint(lo, hi)
        s, m, o = ab
        slo, shi = self.interval(s)
        for _ in range(60):
            t = rng.randint(slo, shi)
            if self._valid(s, t) and lo <= m * t + o <= hi:
                return m * t + o
        t = self._scan(s, slo, shi, 1)
        return m * (t if t is not None else slo) + o

    def residues(self, name: str, m: int) -> list[int]:
        """Erreichbare Reste `name % m`."""
        ab = self.affine_base(name)
        lo, hi = self.interval(name)
        if ab is None:
            if hi - lo + 1 >= m:
                return list(range(m))
            return sorted({t % m for t in range(lo, hi + 1)})
        s, mm, o = ab
        slo, shi = self.interval(s)
        period = m
        for mod, _ in self.v[s].res:
            period = period * mod // math.gcd(period, mod)
        stop = min(shi, slo + 2 * period + len(self.v[s].holes) + 1)
        out = {(mm * t + o) % m for t in range(slo, stop + 1)
               if self._valid(s, t) and lo <= mm * t + o <= hi}
        return sorted(out)

    # -- Verfeinerung -----------------------------------------------------
    def exclude(self, name: str, c: int) -> None:
        ab = self.affine_base(name)
        if ab is None:
            return
        s, m, o = ab
        if (c - o) % m == 0:
            b = self.v[s]
            b.holes = b.holes | {(c - o) // m}

    def restrict_mod(self, name: str, m: int, r: int, truth: bool) -> None:
        ab = self.affine_base(name)
        if ab is None:
            return
        s, mm, o = ab
        allowed = frozenset(u for u in range(m) if ((mm * u + o) % m == r) == truth)
        b = self.v[s]
        res = dict(b.res)
        res[m] = res.get(m, frozenset(range(m))) & allowed
        b.res = tuple(sorted(res.items()))

    def refine(self, name: str, lo: int, hi: int) -> None:
        x = self.v[name]
        x.lo, x.hi = max(x.lo, lo), min(x.hi, hi)
        if x.expr is None:
            return
        clo, chi = self.interval(name)
        kind = x.expr[0]
        if kind == "aff":
            _, s, m, o = x.expr
            if m > 0:   # m*t + o in [clo, chi]  <=>  t in [ceil((clo-o)/m), floor((chi-o)/m)]
                blo, bhi = -((o - clo) // m), (chi - o) // m
            else:       # fuer m < 0 kehren sich die Grenzen um
                blo, bhi = -((o - chi) // m), (clo - o) // m
            self.refine(s, blo, bhi)
        elif kind == "add":
            xs, ys = x.expr[1], x.expr[2]
            ylo, yhi = self.interval(ys)
            self.refine(xs, clo - yhi, chi - ylo)
            xlo, xhi = self.interval(xs)
            self.refine(ys, clo - xhi, chi - xlo)
        elif kind == "sub":
            xs, ys = x.expr[1], x.expr[2]
            ylo, yhi = self.interval(ys)
            self.refine(xs, clo + ylo, chi + yhi)
            xlo, xhi = self.interval(xs)
            self.refine(ys, xlo - chi, xhi - clo)

    def merge_from(self, envs: list["Env"]) -> "Env":
        """Zusammenfuehrung nach einer Verzweigung: je Basisvariable sichere Vereinigung."""
        out = self.copy()
        if not envs:
            return out
        for name in self.v:
            if out.v[name].expr is not None:     # abgeleitete Variablen folgen ihren Basen
                continue
            group = [e.v[name] for e in envs if name in e.v]
            ivs = sorted(e.interval(name) for e in envs if name in e.v)
            if not ivs:
                continue
            comps = [list(ivs[0])]
            for lo, hi in ivs[1:]:
                if lo <= comps[-1][1] + 1:
                    comps[-1][1] = max(comps[-1][1], hi)
                else:
                    comps.append([lo, hi])
            lo, hi = max(comps, key=lambda c: c[1] - c[0])
            x = out.v[name]
            x.lo, x.hi = lo, hi
            x.holes = frozenset.intersection(*(g.holes for g in group))
            mods = set.intersection(*({m for m, _ in g.res} for g in group))
            x.res = tuple(sorted((m, frozenset().union(*(dict(g.res)[m] for g in group)))
                                 for m in mods))
        return out


# ---------------------------------------------------------------------------
# Bedingungen
# ---------------------------------------------------------------------------
REL_OPS = ["<", "<=", ">", ">="]
NEG = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


@dataclass
class Atom:
    kind: str                 # 'rel' | 'eq' | 'mod' | 'rel2'
    x: str
    op: str
    rhs: int | str
    m: int = 0

    def render(self) -> str:
        if self.kind == "mod":
            return f"{self.x} % {self.m} {self.op} {self.rhs}"
        return f"{self.x} {self.op} {self.rhs}"

    def refine(self, env: Env, truth: bool) -> None:
        op = self.op if truth else NEG[self.op]
        if self.kind == "mod":
            env.restrict_mod(self.x, self.m, int(self.rhs), op == "==")
            return
        if self.kind in ("rel", "eq"):
            _refine_const(env, self.x, op, int(self.rhs))
        else:
            _refine_vars(env, self.x, op, str(self.rhs))


def _refine_const(env: Env, x: str, op: str, c: int) -> None:
    lo, hi = env.interval(x)
    if op == "<":
        env.refine(x, -BIG, c - 1)
    elif op == "<=":
        env.refine(x, -BIG, c)
    elif op == ">":
        env.refine(x, c + 1, BIG)
    elif op == ">=":
        env.refine(x, c, BIG)
    elif op == "==":
        env.refine(x, c, c)
    elif op == "!=":
        if c == lo:
            env.refine(x, lo + 1, BIG)
        elif c == hi:
            env.refine(x, -BIG, hi - 1)
        env.exclude(x, c)


def _refine_vars(env: Env, x: str, op: str, y: str) -> None:
    if op in (">", ">="):
        x, y, op = y, x, {">": "<", ">=": "<="}[op]
    strict = 1 if op == "<" else 0
    ylo, yhi = env.interval(y)
    env.refine(x, -BIG, yhi - strict)
    xlo, xhi = env.interval(x)
    env.refine(y, xlo + strict, BIG)


@dataclass
class Compound:
    op: str                   # 'and' | 'or'
    parts: list

    def render(self) -> str:
        return f" {self.op} ".join(p.render() for p in self.parts)

    def refine(self, env: Env, truth: bool) -> None:
        if (self.op == "and" and truth) or (self.op == "or" and not truth):
            for p in self.parts:
                p.refine(env, truth)


# ---------------------------------------------------------------------------
# Kontrollbaum
# ---------------------------------------------------------------------------
@dataclass
class Node:
    kind: str                         # 'if' | 'loop'
    n_conds: int = 1                  # if + elif-Bedingungen
    boolops: list = field(default_factory=lambda: [0])
    has_else: bool = False
    loop_kind: str = ""
    blocks: list = field(default_factory=list)
    depth: int = 0

    def n_blocks(self) -> int:
        return self.n_conds + 1 if self.kind == "if" else 1

    def ensure_blocks(self) -> None:
        while len(self.blocks) < self.n_blocks():
            self.blocks.append([])


@dataclass
class GeneratedProgram:
    source: str
    spec: Spec
    func_name: str = "f"


class Generator:
    def __init__(self, spec: Spec):
        spec.validate()
        self.spec = spec
        self.rng = random.Random(spec.seed)
        self.lines: list[str] = []
        self.n_local = 0
        self.n_while = 0
        self.static = Env()      # Wertebereiche ohne Pfadbedingungen (sichere Huelle)

    # -- Schritt 1: Budgetverteilung -------------------------------------
    def allocate(self) -> list[Node]:
        s, rng = self.spec, self.rng
        loops = [Node("loop", n_conds=0, boolops=[], loop_kind=rng.choice(s.loop_kinds))
                 for _ in range(s.n_loops)]
        chains = [Node("if", has_else=rng.random() < s.p_else)
                  for _ in range(max(0, s.max_depth - s.n_loops))]
        rest = s.decisions - len(loops) - len(chains)
        while rest > 0:
            u = rng.random()
            if chains and u < s.p_compound:
                ch = rng.choice(chains)
                k = rng.randrange(ch.n_conds)
                if ch.boolops[k] < 2:
                    ch.boolops[k] += 1
                    rest -= 1
                    continue
            elif chains and u < s.p_compound + s.p_elif:
                ch = rng.choice(chains)
                if ch.n_conds < 4:
                    ch.n_conds += 1
                    ch.boolops.append(0)
                    rest -= 1
                    continue
            chains.append(Node("if", has_else=rng.random() < s.p_else))
            rest -= 1
        nodes = loops + chains
        for nd in nodes:
            nd.ensure_blocks()
        return nodes

    # -- Schritt 2: Kontrollbaum mit Rueckgrat ---------------------------
    def arrange(self, nodes: list[Node]) -> list[Node]:
        s, rng = self.spec, self.rng
        rng.shuffle(nodes)
        spine, rest = nodes[: s.max_depth], nodes[s.max_depth:]
        top: list[Node] = []
        placed: list[Node] = []
        parent = None
        for k, nd in enumerate(spine):
            nd.depth = k + 1
            if parent is None:
                top.append(nd)
            else:
                b = self._pick_block(parent)
                parent.blocks[b].append(nd)
            placed.append(nd)
            parent = nd
        for nd in rest:
            cands: list[tuple] = [(None, 0)] * 2
            for p in placed:
                if p.depth < s.max_depth:
                    for b in range(p.n_blocks()):
                        if p.kind == "if" and b == p.n_conds and not p.has_else:
                            continue
                        cands.append((p, b))
            p, b = rng.choice(cands)
            if p is None:
                nd.depth = 1
                top.insert(rng.randint(0, len(top)), nd)
            else:
                nd.depth = p.depth + 1
                p.blocks[b].insert(rng.randint(0, len(p.blocks[b])), nd)
            placed.append(nd)
        return top

    def _pick_block(self, parent: Node) -> int:
        if parent.kind == "loop":
            return 0
        if self.rng.random() < self.spec.p_else / (parent.n_conds + 1):
            parent.has_else = True
            return parent.n_conds
        return self.rng.randrange(parent.n_conds)

    # -- Schritt 3: Rendering ----------------------------------------------
    def emit(self, text: str, indent: int) -> None:
        self.lines.append("    " * indent + text)

    def usable(self, env: Env) -> list[str]:
        return [n for n in env.names() if env.usable(n)]

    def gen_atom(self, env: Env) -> Atom:
        """Atomare Bedingung, deren beide Ausgaenge erfuellbar sind und nach der
        noch mindestens eine Variable mit mehreren moeglichen Werten verbleibt."""
        atom = None
        for _ in range(8):
            atom = self._gen_atom(env)
            ok = True
            for truth in (True, False):
                e2 = env.copy()
                atom.refine(e2, truth)
                if not self.usable(e2):
                    ok = False
                    break
            if ok:
                return atom
        return atom

    def _gen_atom(self, env: Env) -> Atom:
        s, rng = self.spec, self.rng
        u = rng.random()
        if u < s.p_eq:
            kind = "eq"
        elif u < s.p_eq + s.p_mod:
            kind = "mod"
        elif u < s.p_eq + s.p_mod + s.p_rel2:
            kind = "rel2"
        else:
            kind = "rel"
        names = self.usable(env)
        if kind == "mod":
            m = rng.randint(2, 7)
            cands = [(n, r) for n in names for r in [env.residues(n, m)] if len(r) >= 2]
            if cands:
                x, res = rng.choice(cands)
                return Atom("mod", x, rng.choice(["==", "==", "!="]), rng.choice(res), m)
            kind = "rel"
        if kind == "rel2":
            pairs = []
            ext = {n: env.extremes(n) for n in names}
            for x in names:
                for y in names:
                    if x < y and not (env.roots(x) & env.roots(y)):
                        (xlo, xhi), (ylo, yhi) = ext[x], ext[y]
                        if xlo < yhi and xhi > ylo:
                            pairs.append((x, y))
            if pairs:
                x, y = rng.choice(pairs)
                if rng.random() < 0.5:
                    x, y = y, x
                return Atom("rel2", x, rng.choice(REL_OPS), y)
            kind = "rel"
        if not names:   # alle Variablen auf einen Wert festgelegt (selten)
            names = env.names()
        x = rng.choice(names)
        if kind == "eq":
            return Atom("eq", x, rng.choice(["==", "==", "!="]), env.random_value(x, rng))
        ex = env.extremes(x) or env.interval(x)
        lo, hi = ex
        op = rng.choice(REL_OPS)
        if hi <= lo:
            return Atom("rel", x, op, lo)
        if op in ("<", ">="):
            c = rng.randint(lo + 1, hi)
        else:
            c = rng.randint(lo, hi - 1)
        return Atom("rel", x, op, c)

    def gen_cond(self, env: Env, extra: int):
        first = self.gen_atom(env)
        if extra == 0:
            return first
        ops = ["and", "or"]
        self.rng.shuffle(ops)
        op = ops[0]
        for cand in ops:       # Verknuepfung so waehlen, dass Folgeatome Spielraum behalten
            sub = env.copy()
            first.refine(sub, cand == "and")
            if self.usable(sub):
                op = cand
                break
        parts = [first]
        sub = env.copy()
        for _ in range(extra):
            nxt = sub.copy()
            parts[-1].refine(nxt, op == "and")
            if self.usable(nxt):
                sub = nxt
            parts.append(self.gen_atom(sub))
        return Compound(op, parts)

    def _fmt_aff(self, src: str, m: int, o: int) -> str:
        if m == 1:
            t = src
        elif m == -1:
            t = f"-{src}"
        else:
            t = f"{src} * {m}"
        if o > 0:
            t += f" + {o}"
        elif o < 0:
            t += f" - {-o}"
        return t

    def emit_assign(self, env: Env, indent: int) -> None:
        rng = self.rng
        names = env.names()
        if rng.random() < 0.3:
            x = rng.choice(names)
            form = rng.choice([f"r = r + {x}", f"r = r - {x}", f"r = r + {x} * {rng.randint(2, 4)}",
                               f"r = r + {rng.randint(1, 9)}"])
            self.emit(form, indent)
            return
        self.n_local += 1
        t = f"t{self.n_local}"
        u = rng.random()
        if u < 0.55 or len(names) < 2:
            src = rng.choice(names)
            m = rng.choice([1, 1, 2, 3, -1, -2])
            o = rng.randint(-9, 9)
            if m == 1 and o == 0:
                o = rng.randint(1, 9)
            xs = env.v[src]
            if xs.expr is not None and xs.expr[0] == "aff":
                _, s2, m2, o2 = xs.expr
                expr = ("aff", s2, m * m2, m * o2 + o)
            else:
                expr = ("aff", src, m, o)
            self.emit(f"{t} = {self._fmt_aff(src, m, o)}", indent)
        elif u < 0.75:
            x, y = rng.sample(names, 2)
            expr = ("add", x, y)
            self.emit(f"{t} = {x} + {y}", indent)
        elif u < 0.9:
            x, y = rng.sample(names, 2)
            expr = ("sub", x, y)
            self.emit(f"{t} = {x} - {y}", indent)
        else:
            x = rng.choice(names)
            m = rng.randint(2, 7)
            expr = ("mod", x, m)
            self.emit(f"{t} = {x} % {m}", indent)
        env.add(Var(t, expr=expr))
        self.static.add(Var(t, expr=expr))

    def _static_loop_var(self, name: str, hi: int) -> None:
        old = self.static.v.get(name)
        self.static.add(Var(name, 0, max(hi, old.hi if old else 0)))

    def render_block(self, children: list[Node], env: Env, indent: int, loop_level: int,
                     may_return: bool) -> Env | None:
        s, rng = self.spec, self.rng
        env = env.copy()
        n_assign = rng.randint(s.min_stmts, s.max_stmts)
        slots = [0] * (len(children) + 1)
        slots[0] = 1
        for _ in range(n_assign - 1):
            slots[rng.randrange(len(slots))] += 1
        for idx in range(len(slots)):
            for _ in range(slots[idx]):
                self.emit_assign(env, indent)
            if idx < len(children):
                env = self.render_node(children[idx], env, indent, loop_level)
        if may_return and loop_level == 0 and rng.random() < s.p_return:
            x = rng.choice(env.names())
            self.emit(rng.choice([f"return r + {x}", f"return {x}", "return r"]), indent)
            return None
        return env

    def render_node(self, node: Node, env: Env, indent: int, loop_level: int) -> Env:
        if node.kind == "if":
            return self.render_if(node, env, indent, loop_level)
        return self.render_loop(node, env, indent, loop_level)

    def render_if(self, node: Node, env: Env, indent: int, loop_level: int) -> Env:
        cur = env.copy()
        falls: list[Env] = []
        returned = 0
        for k in range(node.n_conds):
            cond = self.gen_cond(cur, node.boolops[k])
            self.emit(("if " if k == 0 else "elif ") + cond.render() + ":", indent)
            then_env = cur.copy()
            cond.refine(then_env, True)
            out = self.render_block(node.blocks[k], then_env, indent + 1, loop_level, True)
            if out is None:
                returned += 1
            else:
                falls.append(out)
            cond.refine(cur, False)
        if node.has_else:
            self.emit("else:", indent)
            may_ret = returned < node.n_conds   # nie alle Zweige mit return beenden
            out = self.render_block(node.blocks[node.n_conds], cur, indent + 1, loop_level, may_ret)
            if out is None:
                returned += 1
            else:
                falls.append(out)
        else:
            falls.append(cur)
        return env.merge_from(falls)

    def render_loop(self, node: Node, env: Env, indent: int, loop_level: int) -> Env:
        rng = self.rng
        lv = LOOP_VARS[loop_level]
        body = env.copy()
        kind = node.loop_kind
        if kind == "for_mod":
            K = rng.randint(3, 5)
            cands = [n for n in self.usable(env) if len(env.residues(n, K)) == K]
            if cands:
                v = rng.choice(cands)
                self.emit(f"for {lv} in range({v} % {K}):", indent)
                body.add(Var(lv, 0, K - 2))
                self._static_loop_var(lv, K - 2)
            else:
                kind = "for_const"
        if kind == "while":
            cands = self.usable(env)
            if cands:
                v = rng.choice(cands)
                lo, hi = env.extremes(v)
                c = rng.randint(lo, hi - 1)
                hi_static = self.static.interval(v)[1]      # sichere Obergrenze
                d = max(1, math.ceil((max(hi, hi_static) - c) / MAX_ITER))
                self.n_while += 1
                w = f"w{self.n_while}"
                self.emit(f"{w} = {v}", indent)
                self.emit(f"while {w} > {c}:", indent)
                body.refine(v, c + 1, BIG)
            else:
                kind = "for_const"
        if kind == "for_const":
            K = rng.randint(2, 5)
            self.emit(f"for {lv} in range({K}):", indent)
            body.add(Var(lv, 0, K - 1))
            self._static_loop_var(lv, K - 1)
        self.render_block(node.blocks[0], body, indent + 1, loop_level + 1, False)
        if kind == "while":
            self.emit(f"{w} = {w} - {d}", indent + 1)
        return env

    def generate(self, header: bool = False) -> GeneratedProgram:
        s = self.spec
        top = self.arrange(self.allocate())
        params = PARAM_NAMES[: s.n_params]
        env = Env()
        for p in params:
            env.add(Var(p, -s.domain, s.domain))
            self.static.add(Var(p, -s.domain, s.domain))
        if header:
            self.lines.append(f"# cc={s.cc}, Tiefe={s.max_depth}, "
                              f"Schleifen={s.n_loops}, Parameter={s.n_params}, "
                              f"Bereich=[-{s.domain}, {s.domain}]")
        self.lines.append(f"def f({', '.join(params)}):")
        self.emit("r = 0", 1)
        self.render_block(top, env, 1, 0, False)
        self.emit("return r", 1)
        return GeneratedProgram("\n".join(self.lines) + "\n", s)


def generate(spec: Spec, header: bool = False) -> GeneratedProgram:
    return Generator(spec).generate(header=header)
