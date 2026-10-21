"""Unabhaengige statische Analyse: Metriken und Merkmale einer Python-Funktion.

Wird sowohl zur Pruefung der Spezifikationstreue als auch als Merkmalsquelle
fuer die lernbasierten Vorhersagemodelle verwendet.
"""
from __future__ import annotations

import ast
import math


def _const_value(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const_value(node.operand)
        return None if v is None else -v
    return None


def _has_name(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) for n in ast.walk(node))


def _has_mod(node: ast.AST) -> bool:
    return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod) for n in ast.walk(node))


def find_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("keine Funktion gefunden")


def int_constants(source: str) -> list[int]:
    """Ganzzahlige Konstanten aus Vergleichen (fuer Konstanten-Seeding)."""
    fn = find_function(ast.parse(source))
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            for side in [node.left, *node.comparators]:
                for sub in ast.walk(side):
                    v = _const_value(sub)
                    if v is not None:
                        out.add(v)
    return sorted(out)


class _Walker:
    def __init__(self):
        self.n_if = self.n_elif = self.n_else = 0
        self.n_for = self.n_while = 0
        self.max_depth = 0
        self.decision_depths: list[int] = []
        self.n_return_early = 0
        self.cond = {"rel": 0, "eq": 0, "mod": 0, "rel2": 0}

    def classify(self, test: ast.AST) -> None:
        for node in ast.walk(test):
            if isinstance(node, ast.Compare):
                left = node.left
                for op, right in zip(node.ops, node.comparators):
                    pair = ast.BinOp(left=left, op=ast.Add(), right=right)
                    if _has_mod(pair):
                        self.cond["mod"] += 1
                    elif isinstance(op, (ast.Eq, ast.NotEq)):
                        self.cond["eq"] += 1
                    elif _has_name(left) and _has_name(right):
                        self.cond["rel2"] += 1
                    else:
                        self.cond["rel"] += 1
                    left = right

    def block(self, stmts: list[ast.stmt], depth: int) -> None:
        for s in stmts:
            if isinstance(s, ast.If):
                self.if_chain(s, depth)
            elif isinstance(s, (ast.For, ast.While)):
                if isinstance(s, ast.For):
                    self.n_for += 1
                else:
                    self.n_while += 1
                    self.classify(s.test)
                self.max_depth = max(self.max_depth, depth + 1)
                self.decision_depths.append(depth + 1)
                self.block(s.body, depth + 1)
                self.block(s.orelse, depth + 1)
            elif isinstance(s, ast.Return) and depth > 0:
                self.n_return_early += 1

    def if_chain(self, node: ast.If, depth: int) -> None:
        self.n_if += 1
        self.classify(node.test)
        self.max_depth = max(self.max_depth, depth + 1)
        self.decision_depths.append(depth + 1)
        self.block(node.body, depth + 1)
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            self.n_elif += 1
            self.if_chain(node.orelse[0], depth)
        elif node.orelse:
            self.n_else += 1
            self.block(node.orelse, depth + 1)


def analyze(source: str) -> dict:
    """Statische Metriken der ersten Funktion im Quelltext."""
    fn = find_function(ast.parse(source))
    w = _Walker()
    w.block(fn.body, 0)
    n_boolops = sum(len(n.values) - 1 for n in ast.walk(fn) if isinstance(n, ast.BoolOp))
    n_and = sum(len(n.values) - 1 for n in ast.walk(fn)
                if isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And))
    n_loops = w.n_for + w.n_while
    n_stmts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.stmt)) - 1
    lines = ast.get_source_segment(source, fn) or source
    loc = sum(1 for ln in lines.splitlines() if ln.strip() and not ln.strip().startswith("#"))
    assigned = {t.id for n in ast.walk(fn) if isinstance(n, (ast.Assign, ast.AugAssign))
                for t in ast.walk(n.targets[0] if isinstance(n, ast.Assign) else n.target)
                if isinstance(t, ast.Name)}
    n_atoms = sum(w.cond.values())
    n_consts = sum(1 for n in ast.walk(fn) if isinstance(n, ast.Constant)
                   and isinstance(n.value, int))
    return {
        "n_params": len(fn.args.args),
        "cc": 1 + w.n_if + n_loops + n_boolops,
        "max_depth": w.max_depth,
        "n_loops": n_loops,
        "n_for": w.n_for,
        "n_while": w.n_while,
        "n_if": w.n_if,
        "n_elif": w.n_elif,
        "n_else": w.n_else,
        "n_boolops": n_boolops,
        "n_and": n_and,
        "n_or": n_boolops - n_and,
        "n_branches": 2 * (w.n_if + n_loops),
        "loc": loc,
        "n_stmts": n_stmts,
        "n_locals": len(assigned),
        "n_return_early": w.n_return_early,
        "n_cond_rel": w.cond["rel"],
        "n_cond_eq": w.cond["eq"],
        "n_cond_mod": w.cond["mod"],
        "n_cond_rel2": w.cond["rel2"],
        "frac_eq": w.cond["eq"] / n_atoms if n_atoms else 0.0,
        "frac_mod": w.cond["mod"] / n_atoms if n_atoms else 0.0,
        "frac_rel2": w.cond["rel2"] / n_atoms if n_atoms else 0.0,
        "mean_decision_depth": (sum(w.decision_depths) / len(w.decision_depths)
                                if w.decision_depths else 0.0),
        "n_consts": n_consts,
    }
