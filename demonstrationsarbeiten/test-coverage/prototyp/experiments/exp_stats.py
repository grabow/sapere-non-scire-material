"""Kennzahlen fuer den Text der Evaluation (data/summary.json)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    progs = [json.loads(l) for l in open(os.path.join(DATA, "programs.jsonl"))]
    s = {}
    s["n_programs"] = len(progs)
    s["conform_all"] = sum(all(p["conform"].values()) for p in progs)
    s["exceptions"] = sum(p["n_exceptions"] for p in progs)
    tg = np.array([p["t_gen"] for p in progs]) * 1000
    te = np.array([p["t_est"] for p in progs]) * 1000
    s["t_gen_ms"] = {"mean": tg.mean(), "p95": np.percentile(tg, 95), "max": tg.max()}
    s["t_est_ms"] = {"median": np.median(te), "mean": te.mean(), "p95": np.percentile(te, 95),
                     "max": te.max()}
    mc = np.array([p["max_cov"] for p in progs])
    s["max_cov_mean"] = mc.mean()
    s["full_coverable"] = float(np.mean(mc == 1))
    nb = np.array([p["n_branches"] for p in progs])
    s["branches"] = {"total": int(nb.sum()), "mean": nb.mean(), "min": int(nb.min()), "max": int(nb.max())}
    loc = np.array([p["metrics"]["loc"] for p in progs])
    s["loc"] = {"mean": loc.mean(), "min": int(loc.min()), "max": int(loc.max())}

    br = pd.read_csv(os.path.join(DATA, "branches.csv"))
    s["branch_feasible_share"] = br["feasible"].mean()
    inf = br[br["feasible"] == 0]
    fea = br[br["feasible"] == 1]
    s["infeasible_pred_zero"] = float(np.mean(inf["p_pred"] < 1e-6))
    s["infeasible_pred_gt_001"] = float(np.mean(inf["p_pred"] > 0.01))
    s["feasible_pred_zero"] = float(np.mean(fea["p_pred"] < 1e-6))
    e = br["p_pred"] - br["p_emp"]
    s["branch_mae"] = float(np.mean(np.abs(e)))
    s["branch_within_001"] = float(np.mean(np.abs(e) <= 0.01))
    s["branch_spearman"] = float(spearmanr(br["p_pred"], br["p_emp"]).statistic)
    f2 = fea[fea["p_emp"] > 0]
    s["branch_log_ratio_median"] = float(np.median(np.abs(np.log10(np.maximum(f2["p_pred"], 1e-9)
                                                                     / f2["p_emp"]))))

    sp = pd.DataFrame([{**{f"s_{k}": v for k, v in p["spec"].items()}, **p["metrics"]} for p in progs])
    s["soft_corr"] = {k: float(pearsonr(sp[f"s_{a}"], sp[b]).statistic)
                      for k, a, b in [("eq", "p_eq", "frac_eq"), ("mod", "p_mod", "frac_mod"),
                                      ("rel2", "p_rel2", "frac_rel2"),
                                      ("compound", "p_compound", "n_boolops"),
                                      ("return", "p_return", "n_return_early")]}
    s["loc_cc_corr"] = float(pearsonr(sp["cc"], sp["loc"]).statistic)

    nbase = json.load(open(os.path.join(DATA, "naive_baseline.json")))
    rates = np.array(nbase["rates"])
    s["naive"] = {"never": float(np.mean(rates == 0)),
                  "median_trials": float(np.median(1 / rates[rates > 0])),
                  "share_gt_1000_trials_or_never": float(np.mean((rates == 0) | (rates < 1e-3))),
                  "t_per_sample_us": nbase["t_per_sample"] * 1e6}
    # Akzeptanzrate in Abhaengigkeit der Tiefe
    sp["rate"] = rates
    s["naive_by_depth"] = {int(d): float(np.mean(g["rate"] == 0)) for d, g in sp.groupby("max_depth")}

    cov = pd.read_csv(os.path.join(DATA, "coverage.csv"))
    piv = cov.pivot_table(index=["id", "budget"], columns="strategy", values="coverage").reset_index()
    dom = pd.DataFrame({"id": [p["id"] for p in progs], "domain": [p["spec"]["domain"] for p in progs],
                        "cc": [p["metrics"]["cc"] for p in progs]})
    piv = piv.merge(dom, on="id")
    s["cov_mean"] = {int(b): {"random": g["random"].mean(), "seeded": g["seeded"].mean()}
                     for b, g in piv.groupby("budget")}
    s["cov_by_domain_100"] = {int(d): {"random": g["random"].mean(), "seeded": g["seeded"].mean()}
                              for d, g in piv[piv["budget"] == 100].groupby("domain")}
    std = cov[(cov["strategy"] == "random")].groupby("budget")["std"].mean()
    s["rep_std"] = {int(k): float(v) for k, v in std.items()}

    oof = pd.read_csv(os.path.join(DATA, "model_oof.csv"))
    d = oof[oof["budget"] == 100].copy()
    d["err"] = d["Analytisch"] - d["target"]
    s["analytic_n100_abs_gt5"] = float(np.mean(np.abs(d["err"]) > 0.05))
    s["analytic_n100_abs_gt10"] = float(np.mean(np.abs(d["err"]) > 0.10))
    # Fehler nach Vorhandensein unerreichbarer, aber als erreichbar geschaetzter Zweige
    bad = inf[inf["p_pred"] > 1e-3].groupby("id").size()
    d["has_bad"] = d["id"].isin(bad.index)
    s["analytic_n100_mae_by_bad"] = {str(k): float(np.mean(np.abs(g["err"])))
                                     for k, g in d.groupby("has_bad")}
    s["share_programs_bad"] = float(d["has_bad"].mean())
    s["analytic_n100_mae_by_cc"] = {f"{lo}-{hi}": float(np.mean(np.abs(d[(d.cc >= lo) & (d.cc <= hi)]["err"])))
                                    for lo, hi in [(2, 8), (9, 16), (17, 25)]}
    s["analytic_n100_mae_by_domain"] = {int(k): float(np.mean(np.abs(g["err"])))
                                        for k, g in d.groupby("domain")}
    json.dump(s, open(os.path.join(DATA, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
