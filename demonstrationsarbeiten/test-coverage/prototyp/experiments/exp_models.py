"""Versuch 5: Vorhersagemodelle fuer die Coverage zufaelliger Tests.

Vergleich: Mittelwert-Baseline, lineare Regression, Random Forest, Gradient
Boosting, analytischer Schaetzer und Hybridmodell (Gradient Boosting mit der
analytischen Schaetzung als zusaetzlichem Merkmal).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES = ["n_params", "log_domain", "loc", "n_stmts", "cc", "max_depth", "n_loops", "n_for",
            "n_while", "n_if", "n_elif", "n_else", "n_boolops", "n_and", "n_or", "n_branches",
            "n_locals", "n_return_early", "n_cond_rel", "n_cond_eq", "n_cond_mod", "n_cond_rel2",
            "frac_eq", "frac_mod", "frac_rel2", "mean_decision_depth", "n_consts"]
X_COLS = FEATURES + ["log_budget"]
SEED = 7


def load() -> pd.DataFrame:
    progs = [json.loads(l) for l in open(os.path.join(DATA, "programs.jsonl"))]
    meta = pd.DataFrame([{**p["metrics"], "id": p["id"], "domain": p["spec"]["domain"],
                          "max_cov": p["max_cov"]} for p in progs])
    meta["log_domain"] = np.log10(meta["domain"])
    cov = pd.read_csv(os.path.join(DATA, "coverage.csv"))
    piv = cov.pivot_table(index=["id", "budget"], columns="strategy", values="coverage").reset_index()
    df = piv.merge(meta, on="id")
    df["log_budget"] = np.log10(df["budget"])
    df = df.rename(columns={"random": "target"})
    return df.sort_values(["id", "budget"]).reset_index(drop=True)


def make_models():
    return {
        "Lineare Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=SEED),
        "Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=31, random_state=SEED),
    }


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    preds = {}
    means = train.groupby("budget")["target"].mean()
    preds["Baseline"] = test["budget"].map(means).to_numpy()
    for name, mk in make_models().items():
        mdl = mk().fit(train[X_COLS], train["target"])
        preds[name] = np.clip(mdl.predict(test[X_COLS]), 0, 1)
    preds["Analytisch"] = test["analytic"].to_numpy()
    hyb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                        random_state=SEED)
    hyb.fit(train[X_COLS + ["analytic"]], train["target"])
    preds["Hybrid"] = np.clip(hyb.predict(test[X_COLS + ["analytic"]]), 0, 1)
    return preds


def scores(y, p) -> dict:
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "R2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else float("nan"),
            "within5": float(np.mean(np.abs(e) <= 0.05)),
            "spearman": float(spearmanr(y, p).statistic), "bias": float(np.mean(e))}


def evaluate(df: pd.DataFrame, preds: dict) -> dict:
    out = {}
    for name, p in preds.items():
        out[name] = {"all": scores(df["target"].to_numpy(), p)}
        for b in sorted(df["budget"].unique()):
            m = (df["budget"] == b).to_numpy()
            out[name][int(b)] = scores(df["target"].to_numpy()[m], p[m])
    return out


def main():
    df = load()
    res = {"n_programs": int(df["id"].nunique()), "n_rows": len(df)}

    # 1) 5-fache Kreuzvalidierung, gruppiert nach Programm
    oof = {k: np.zeros(len(df)) for k in ["Baseline", *make_models(), "Analytisch", "Hybrid"]}
    for tr, te in GroupKFold(n_splits=5).split(df, groups=df["id"]):
        preds = fit_predict(df.iloc[tr], df.iloc[te])
        for k, v in preds.items():
            oof[k][te] = v
    res["cv"] = evaluate(df, oof)
    out = df[["id", "budget", "target", "cc", "domain", "max_depth", "n_loops"]].copy()
    for k, v in oof.items():
        out[k] = v
    out.to_csv(os.path.join(DATA, "model_oof.csv"), index=False)

    # 2) Extrapolation: Komplexitaet und Eingabebereich
    for label, mtr, mte in [("cc", df["cc"] <= 12, df["cc"] >= 13),
                            ("domain", df["domain"] <= 100, df["domain"] == 1000)]:
        tr, te = df[mtr], df[mte]
        res[f"extrapolation_{label}"] = evaluate(te.reset_index(drop=True), fit_predict(tr, te))
        res[f"extrapolation_{label}"]["n_train"] = int(tr["id"].nunique())
        res[f"extrapolation_{label}"]["n_test"] = int(te["id"].nunique())

    # 3) Lernkurve
    rng = np.random.default_rng(SEED)
    ids = df["id"].unique()
    rng.shuffle(ids)
    test_ids, pool = ids[:1000], ids[1000:]
    te = df[df["id"].isin(test_ids)]
    lc = []
    for n in [50, 100, 200, 500, 1000, 2000, len(pool)]:
        tr = df[df["id"].isin(pool[:n])]
        p = fit_predict(tr, te)
        lc.append({"n": int(n), **{k: float(np.mean(np.abs(v - te["target"].to_numpy())))
                                   for k, v in p.items()}})
    res["learning_curve"] = lc

    # 4) Permutations-Wichtigkeit (Gradient Boosting, ohne analytische Schaetzung)
    tr = df[df["id"].isin(pool)]
    gb = make_models()["Gradient Boosting"]().fit(tr[X_COLS], tr["target"])
    pi = permutation_importance(gb, te[X_COLS], te["target"], n_repeats=10, random_state=SEED,
                                scoring="neg_mean_absolute_error", n_jobs=-1)
    res["importance"] = sorted([{"feature": f, "mean": float(m), "std": float(s)}
                                for f, m, s in zip(X_COLS, pi.importances_mean, pi.importances_std)],
                               key=lambda d: -d["mean"])

    # 5) Uebertragung auf Lehrbuchfunktionen
    cs = json.load(open(os.path.join(DATA, "casestudy.json")))
    from ppgen.sampling import BUDGETS  # noqa: E402
    rows = []
    for c in cs:
        for j, b in enumerate(BUDGETS):
            rows.append({**c["metrics"], "log_domain": np.log10(c["domain_equiv"]),
                         "log_budget": np.log10(b), "budget": b, "name": c["name"],
                         "target": c["measured"][j], "analytic": c["analytic"][j]})
    cdf = pd.DataFrame(rows)
    p = fit_predict(df, cdf)
    res["casestudy"] = evaluate(cdf, p)
    cdf_out = cdf[["name", "budget", "target"]].copy()
    for k, v in p.items():
        cdf_out[k] = v
    cdf_out.to_csv(os.path.join(DATA, "casestudy_predictions.csv"), index=False)

    json.dump(res, open(os.path.join(DATA, "model_results.json"), "w"), indent=1)
    for k in oof:
        print(f"{k:20s} MAE={res['cv'][k]['all']['MAE']:.4f} R2={res['cv'][k]['all']['R2']:.3f}"
              f"  N=100: {res['cv'][k][100]['MAE']:.4f}  Fallstudie: {res['casestudy'][k]['all']['MAE']:.4f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    main()
