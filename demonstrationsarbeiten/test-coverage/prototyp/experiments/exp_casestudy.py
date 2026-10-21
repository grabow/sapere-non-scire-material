"""Versuch 4: Uebertragbarkeit auf handgeschriebene Lehrbuchfunktionen (Messung + Analytik)."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.casestudy import CASES                                       # noqa: E402
from ppgen.metrics import analyze                                       # noqa: E402
from ppgen.runner import compile_instrumented, run_inputs, range_inputs  # noqa: E402
from ppgen.estimator import branch_probabilities, expected_coverage     # noqa: E402
from ppgen.sampling import BUDGETS                                      # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    out = []
    for name, doms, src in CASES:
        comp = compile_instrumented(src)
        ranges = list(doms.values())
        runs = [run_inputs(comp, range_inputs(np.random.default_rng([11, r]), ranges, 1000))
                for r in range(20)]
        meas = np.mean([r.coverage_chunked(BUDGETS) for r in runs], axis=0)
        t0 = time.perf_counter()
        probs = branch_probabilities(src, domains=doms)
        t_est = time.perf_counter() - t0
        est = expected_coverage(probs, BUDGETS)
        m = analyze(src)
        half = float(np.mean([(hi - lo) / 2 for lo, hi in ranges]))
        out.append({"name": name, "source": src.strip(), "domains": doms, "metrics": m,
                    "domain_equiv": half, "measured": meas.tolist(), "analytic": est.tolist(),
                    "t_est": t_est})
        print(f"{name:15s} {np.round(meas[[3, 6]], 3)} {np.round(est[[3, 6]], 3)}")
    json.dump(out, open(os.path.join(DATA, "casestudy.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
