"""Versuch 6: Coverage als Zieleigenschaft (vorhersagegesteuerte Generierung).

Kandidaten werden aus der Spezifikationsverteilung erzeugt und nur akzeptiert,
wenn die analytisch vorhergesagte Coverage fuer N = 100 Zufallstests im
Toleranzband um den Zielwert liegt. Danach wird die tatsaechliche Coverage gemessen.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.generator import generate                                       # noqa: E402
from ppgen.runner import compile_instrumented, run_inputs, random_inputs  # noqa: E402
from ppgen.estimator import branch_probabilities, expected_coverage       # noqa: E402
from ppgen.sampling import sample_spec                                    # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N, TOL, PER_TARGET = 100, 0.025, 60


def measure(src: str, domain: int) -> float:
    comp = compile_instrumented(src)
    runs = [run_inputs(comp, random_inputs(np.random.default_rng([5, r]), comp.n_params, domain, 1000))
            for r in range(20)]
    return float(np.mean([r.coverage_chunked([N])[0] for r in runs]))


def main():
    res = []
    seed = 10_000_000
    for target in TARGETS:
        t0 = time.perf_counter()
        tried, acc = 0, []
        while len(acc) < PER_TARGET:
            seed += 1
            spec = sample_spec(np.random.default_rng([seed]), seed)
            src = generate(spec).source
            tried += 1
            est = float(expected_coverage(branch_probabilities(src, spec.domain), [N])[0])
            ok = est >= 1 - TOL if target == 1.0 else abs(est - target) <= TOL
            if ok:
                acc.append({"seed": seed, "est": est, "cc": spec.cc, "domain": spec.domain,
                            "measured": measure(src, spec.domain)})
        dt = time.perf_counter() - t0
        meas = np.array([a["measured"] for a in acc])
        res.append({"target": target, "tried": tried, "accepted": len(acc), "time": dt,
                    "programs": acc, "mae": float(np.mean(np.abs(meas - target))),
                    "mean": float(meas.mean())})
        print(f"Ziel {target:.1f}: {tried} Kandidaten, MAE {res[-1]['mae']:.4f}, "
              f"Mittel {meas.mean():.3f}, {dt:.1f}s")
    json.dump(res, open(os.path.join(DATA, "targeted.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
