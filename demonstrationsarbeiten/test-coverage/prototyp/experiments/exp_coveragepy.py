"""Versuch 2: Validierung der eigenen Zweig-Instrumentierung gegen coverage.py.

Fuer generierte Programme und die Lehrbuchfunktionen wird dieselbe Eingabefolge
einmal mit der eigenen Sonden-Instrumentierung und einmal unter coverage.py
(branch=True) ausgefuehrt und die Zweigueberdeckung verglichen.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

import coverage
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppgen.casestudy import CASES                                       # noqa: E402
from ppgen.runner import compile_instrumented, run_inputs, random_inputs, range_inputs  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def covpy_branch(source: str, inputs: list[tuple]) -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe_mod.py")
        with open(path, "w") as f:
            f.write(source)
        cov = coverage.Coverage(branch=True, data_file=None)
        cov.start()
        spec = importlib.util.spec_from_file_location("probe_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = [v for k, v in vars(mod).items() if callable(v) and not k.startswith("_")][0]
        for args in inputs:
            fn(*args)
        cov.stop()
        out = os.path.join(tmp, "cov.json")
        cov.json_report(morfs=[path], outfile=out)
        summ = json.load(open(out))["files"][path]["summary"] if path in json.load(open(out))["files"] \
            else list(json.load(open(out))["files"].values())[0]["summary"]
        return summ["covered_branches"], summ["num_branches"]


def main():
    progs = [json.loads(l) for l in open(os.path.join(DATA, "programs.jsonl"))][:300]
    rows = []
    for p in progs:
        rng = np.random.default_rng(p["id"])
        comp = compile_instrumented(p["source"])
        for n in (5, 50):
            inputs = random_inputs(rng, comp.n_params, p["spec"]["domain"], n)
            own = run_inputs(comp, inputs).hits.any(axis=0)
            cb, nb = covpy_branch(p["source"], inputs)
            rows.append(("generiert", p["id"], n, int(own.sum()), len(own), cb, nb))
    for name, doms, src in CASES:
        rng = np.random.default_rng(7)
        comp = compile_instrumented(src)
        for n in (5, 50):
            inputs = range_inputs(rng, list(doms.values()), n)
            own = run_inputs(comp, inputs).hits.any(axis=0)
            cb, nb = covpy_branch(src.lstrip(), inputs)
            rows.append(("lehrbuch", name, n, int(own.sum()), len(own), cb, nb))
    same_n = sum(r[4] == r[6] for r in rows)
    same_c = sum(r[3] == r[5] and r[4] == r[6] for r in rows)
    res = {"n_pairs": len(rows), "same_num_branches": same_n, "identical": same_c,
           "mismatches": [r for r in rows if not (r[3] == r[5] and r[4] == r[6])][:20]}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(DATA, "coveragepy_validation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
