"""Ausfuehrung von Tests und Messung der Zweigueberdeckung per AST-Instrumentierung.

Jeder Zweigausgang (if: wahr/falsch, Schleife: betreten/verlassen) erhaelt eine
Sonde ``_F[i] = 1``. Pro Testlauf wird festgehalten, welche Zweige erstmals bei
welchem Testfall erreicht wurden; daraus ergibt sich die gesamte
Coverage-Kurve C(N) fuer alle Budgets N in einem einzigen Durchlauf.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np

from .metrics import find_function, int_constants

BranchKey = tuple  # (lineno, col_offset, outcome)


class Instrumenter(ast.NodeTransformer):
    def __init__(self):
        self.keys: list[BranchKey] = []

    def _new(self, node: ast.AST, outcome: str) -> int:
        self.keys.append((node.lineno, node.col_offset, outcome))
        return len(self.keys) - 1

    @staticmethod
    def _probe(idx: int) -> ast.stmt:
        return ast.parse(f"_F[{idx}] = 1").body[0]

    def visit_If(self, node: ast.If):
        t, f = self._new(node, "T"), self._new(node, "F")
        self.generic_visit(node)
        node.body.insert(0, self._probe(t))
        node.orelse.insert(0, self._probe(f))
        return node

    def _loop(self, node):
        e, x = self._new(node, "E"), self._new(node, "X")
        self.generic_visit(node)
        node.body.insert(0, self._probe(e))
        node.orelse.insert(0, self._probe(x))
        return node

    visit_For = _loop
    visit_While = _loop


@dataclass
class Compiled:
    fn: object
    flags: bytearray
    keys: list
    n_params: int


def compile_instrumented(source: str) -> Compiled:
    tree = ast.parse(source)
    fdef = find_function(tree)
    inst = Instrumenter()
    tree = inst.visit(tree)
    ast.fix_missing_locations(tree)
    flags = bytearray(len(inst.keys))
    ns: dict = {"_F": flags}
    exec(compile(tree, "<generiert>", "exec"), ns)
    return Compiled(ns[fdef.name], flags, inst.keys, len(fdef.args.args))


@dataclass
class RunResult:
    hits: np.ndarray           # bool-Matrix (Tests x Zweige): Zweig im Test erreicht
    n_exceptions: int

    @property
    def n_tests(self) -> int:
        return self.hits.shape[0]

    @property
    def hit_count(self) -> np.ndarray:
        return self.hits.sum(axis=0)

    @property
    def first_hit(self) -> np.ndarray:
        any_hit = self.hits.any(axis=0)
        return np.where(any_hit, self.hits.argmax(axis=0), -1)

    def coverage_at(self, budgets) -> np.ndarray:
        """Coverage der ersten N Tests fuer jedes Budget N."""
        fh = self.first_hit
        if len(fh) == 0:
            return np.ones(len(budgets))
        return np.array([np.mean((fh >= 0) & (fh < n)) for n in budgets])

    def coverage_chunked(self, budgets) -> np.ndarray:
        """Mittlere Coverage ueber disjunkte Teilfolgen der Laenge N
        (reduziert die Varianz fuer kleine Budgets)."""
        nt, nb = self.hits.shape
        if nb == 0:
            return np.ones(len(budgets))
        out = []
        for n in budgets:
            k = nt // n
            h = self.hits[: k * n].reshape(k, n, nb).any(axis=1)
            out.append(h.mean())
        return np.array(out)


def run_inputs(comp: Compiled, inputs: list[tuple]) -> RunResult:
    nb = len(comp.flags)
    view = np.frombuffer(comp.flags, dtype=np.uint8)
    hits = np.zeros((len(inputs), nb), dtype=bool)
    zero = bytes(nb)
    fn, flags = comp.fn, comp.flags
    n_exc = 0
    for t, args in enumerate(inputs):
        flags[:] = zero
        try:
            fn(*args)
        except Exception:      # generierte Programme sollten nie Ausnahmen werfen
            n_exc += 1
        hits[t] = view
    return RunResult(hits, n_exc)


def random_inputs(rng: np.random.Generator, n_params: int, domain: int, n: int) -> list[tuple]:
    return [tuple(row) for row in rng.integers(-domain, domain + 1, size=(n, n_params)).tolist()]


def seeded_inputs(rng: np.random.Generator, n_params: int, domain: int, n: int,
                  constants: list[int], p_seed: float = 0.5) -> list[tuple]:
    """Zufallseingaben, bei denen je Parameter mit Wahrscheinlichkeit p_seed eine
    Konstante aus dem Programm (oder ihr Nachbar +-1) verwendet wird."""
    base = rng.integers(-domain, domain + 1, size=(n, n_params))
    if constants:
        pool = np.array(sorted({c + d for c in constants for d in (-1, 0, 1)}))
        pool = np.clip(pool, -domain, domain)
        mask = rng.random((n, n_params)) < p_seed
        picks = rng.choice(pool, size=(n, n_params))
        base = np.where(mask, picks, base)
    return [tuple(row) for row in base.tolist()]


def measure(source: str, domain: int, n_tests: int, seed: int, strategy: str = "random") -> RunResult:
    comp = compile_instrumented(source)
    rng = np.random.default_rng(seed)
    if strategy == "random":
        inputs = random_inputs(rng, comp.n_params, domain, n_tests)
    elif strategy == "seeded":
        inputs = seeded_inputs(rng, comp.n_params, domain, n_tests, int_constants(source))
    else:
        raise ValueError(strategy)
    return run_inputs(comp, inputs)


def range_inputs(rng: np.random.Generator, ranges: list[tuple[int, int]], n: int) -> list[tuple]:
    """Gleichverteilte Eingaben mit individuellem Bereich je Parameter."""
    cols = [rng.integers(lo, hi + 1, size=n) for lo, hi in ranges]
    return list(zip(*(c.tolist() for c in cols)))
