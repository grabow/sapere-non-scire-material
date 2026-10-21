"""Naiver grammatikbasierter Generator (Vergleichsbasis fuer Generate-and-Test).

Erzeugt Kontrollstrukturen nach einer probabilistischen Grammatik ohne
Budgetsteuerung. Fuer die Akzeptanzrate einer Spezifikation genuegt die
Verteilung von (cc, Tiefe, Schleifen), daher wird nur der Strukturbaum gezogen.
"""
from __future__ import annotations

import random


def sample_structure(rng: random.Random, p_ctrl: float = 0.45, decay: float = 0.75,
                     p_loop: float = 0.25, p_elif: float = 0.2, p_bool: float = 0.2,
                     p_else: float = 0.5, max_depth: int = 8) -> tuple[int, int, int]:
    """Liefert (cc, max_depth, n_loops) eines zufaellig gezogenen Programms."""

    def block(depth: int) -> tuple[int, int, int]:
        dec, md, loops = 0, depth, 0
        for _ in range(rng.randint(1, 3)):
            if depth >= max_depth or rng.random() >= p_ctrl * decay ** depth:
                continue                       # Zuweisung
            if rng.random() < p_loop:
                d, m, l = block(depth + 1)
                dec, loops, md = dec + 1 + d, loops + 1 + l, max(md, m)
                continue
            n_conds = 1
            while n_conds < 4 and rng.random() < p_elif:
                n_conds += 1
            dec += n_conds
            for _ in range(n_conds):
                while rng.random() < p_bool:
                    dec += 1
                d, m, l = block(depth + 1)
                dec, loops, md = dec + d, loops + l, max(md, m)
            if rng.random() < p_else:
                d, m, l = block(depth + 1)
                dec, loops, md = dec + d, loops + l, max(md, m)
        return dec, md, loops

    d, m, l = block(0)
    return 1 + d, m, l
