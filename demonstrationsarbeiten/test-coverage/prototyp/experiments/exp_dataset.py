"""Versuch 1: Datensatz generierter Programme mit gemessener und geschaetzter Coverage.

Aufruf:  python experiments/exp_dataset.py [Anzahl] [Prozesse]
Ausgabe: data/programs.jsonl, data/coverage.csv, data/branches.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.generator import generate                                   # noqa: E402
from ppgen.metrics import analyze, int_constants                      # noqa: E402
from ppgen.runner import compile_instrumented, run_inputs, random_inputs, seeded_inputs  # noqa: E402
from ppgen.estimator import branch_probabilities, expected_coverage   # noqa: E402
from ppgen.sampling import sample_spec, BUDGETS                       # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
REPS_RANDOM, REPS_SEEDED, N_TESTS = 20, 10, 1000
N_EXTRA = 30000
MASTER_SEED = 20261016


def work(idx: int) -> dict:
    rng = np.random.default_rng([MASTER_SEED, idx])
    spec = sample_spec(rng, seed=idx)
    t0 = time.perf_counter()
    prog = generate(spec)
    t_gen = time.perf_counter() - t0
    src = prog.source
    m = analyze(src)
    conform = {k: m[k] == getattr(spec, k) for k in ("cc", "max_depth", "n_loops", "n_params")}

    comp = compile_instrumented(src)
    consts = int_constants(src)
    rand_runs = [run_inputs(comp, random_inputs(np.random.default_rng([idx, 1, r]), comp.n_params,
                                                spec.domain, N_TESTS)) for r in range(REPS_RANDOM)]
    seed_runs = [run_inputs(comp, seeded_inputs(np.random.default_rng([idx, 2, r]), comp.n_params,
                                                spec.domain, N_TESTS, consts)) for r in range(REPS_SEEDED)]
    extra = [run_inputs(comp, random_inputs(np.random.default_rng([idx, 3]), comp.n_params,
                                            spec.domain, N_EXTRA)),
             run_inputs(comp, seeded_inputs(np.random.default_rng([idx, 4]), comp.n_params,
                                            spec.domain, N_EXTRA, consts))]
    all_runs = rand_runs + seed_runs + extra
    feasible = np.any([r.hits.any(axis=0) for r in all_runs], axis=0)
    n_exc = sum(r.n_exceptions for r in all_runs)

    t0 = time.perf_counter()
    probs = branch_probabilities(src, spec.domain)
    t_est = time.perf_counter() - t0
    est = expected_coverage(probs, BUDGETS)

    cov_rows = []
    rc = np.array([r.coverage_chunked(BUDGETS) for r in rand_runs])
    rf = np.array([r.coverage_at(BUDGETS) for r in rand_runs])
    sc = np.array([r.coverage_chunked(BUDGETS) for r in seed_runs])
    for j, n in enumerate(BUDGETS):
        cov_rows.append([idx, "random", n, rc[:, j].mean(), rf[:, j].std()])
        cov_rows.append([idx, "seeded", n, sc[:, j].mean(), 0.0])
        cov_rows.append([idx, "analytic", n, est[j], 0.0])

    hits = np.sum([r.hit_count for r in rand_runs], axis=0)
    n_rand = REPS_RANDOM * N_TESTS
    br_rows = [[idx, f"{k[0]}:{k[1]}:{k[2]}", hits[b] / n_rand, probs[k], int(feasible[b])]
               for b, k in enumerate(comp.keys)]

    rec = {"id": idx, "spec": spec.to_dict(), "source": src, "metrics": m, "conform": conform,
           "t_gen": t_gen, "t_est": t_est, "n_exceptions": int(n_exc),
           "n_branches": len(comp.keys), "max_cov": float(feasible.mean()) if len(feasible) else 1.0}
    return {"rec": rec, "cov": cov_rows, "br": br_rows}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    with Pool(procs) as pool, \
            open(os.path.join(DATA, "programs.jsonl"), "w") as fp, \
            open(os.path.join(DATA, "coverage.csv"), "w", newline="") as fc, \
            open(os.path.join(DATA, "branches.csv"), "w", newline="") as fb:
        wc, wb = csv.writer(fc), csv.writer(fb)
        wc.writerow(["id", "strategy", "budget", "coverage", "std"])
        wb.writerow(["id", "key", "p_emp", "p_pred", "feasible"])
        for k, res in enumerate(pool.imap_unordered(work, range(n), chunksize=4)):
            fp.write(json.dumps(res["rec"]) + "\n")
            wc.writerows(res["cov"])
            wb.writerows(res["br"])
            if (k + 1) % 250 == 0:
                print(f"{k + 1}/{n} Programme, {time.time() - t0:.0f}s", flush=True)
    print(f"fertig in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
