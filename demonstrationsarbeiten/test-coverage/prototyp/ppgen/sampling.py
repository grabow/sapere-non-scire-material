"""Zufaellige Spezifikationen fuer die Versuchsdatensaetze."""
from __future__ import annotations

import numpy as np

from .spec import Spec

BUDGETS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def sample_spec(rng: np.random.Generator, seed: int) -> Spec:
    cc = int(rng.integers(2, 26))
    p_eq, p_mod, p_rel2 = rng.uniform(0, 0.5), rng.uniform(0, 0.3), rng.uniform(0, 0.3)
    return Spec(
        n_params=int(rng.integers(1, 5)),
        cc=cc,
        max_depth=int(rng.integers(1, min(6, cc - 1) + 1)),
        n_loops=int(rng.integers(0, min(3, cc - 1) + 1)),
        domain=int(rng.choice([10, 100, 1000])),
        p_eq=float(p_eq), p_mod=float(p_mod), p_rel2=float(p_rel2),
        p_compound=float(rng.uniform(0, 0.4)),
        p_elif=float(rng.uniform(0, 0.4)),
        p_else=float(rng.uniform(0.2, 0.8)),
        p_return=float(rng.uniform(0, 0.3)),
        max_stmts=int(rng.integers(1, 4)),
        seed=seed,
    )
