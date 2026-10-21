"""Versuch 3: Akzeptanzrate eines naiven Generate-and-Test-Generators.

Fuer jede Spezifikation (cc, Tiefe, Schleifen) wird geschaetzt, mit welcher
Wahrscheinlichkeit ein naiver grammatikbasierter Generator sie zufaellig trifft.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.naive import sample_structure      # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_SAMPLES = 1_000_000


def main():
    rng = random.Random(1)
    t0 = time.perf_counter()
    cnt = Counter(sample_structure(rng) for _ in range(N_SAMPLES))
    t = (time.perf_counter() - t0) / N_SAMPLES
    progs = [json.loads(l) for l in open(os.path.join(DATA, "programs.jsonl"))]
    rates = []
    for p in progs:
        s = p["spec"]
        rates.append(cnt.get((s["cc"], s["max_depth"], s["n_loops"]), 0) / N_SAMPLES)
    hist = {}
    for (cc, d, l), c in cnt.items():
        hist[f"{cc},{d}"] = hist.get(f"{cc},{d}", 0) + c
    res = {"n_samples": N_SAMPLES, "t_per_sample": t, "rates": rates,
           "hist_cc_depth": hist,
           "frac_specs_never_hit": sum(r == 0 for r in rates) / len(rates)}
    json.dump(res, open(os.path.join(DATA, "naive_baseline.json"), "w"))
    print({k: v for k, v in res.items() if k not in ("rates", "hist_cc_depth")})


if __name__ == "__main__":
    main()
