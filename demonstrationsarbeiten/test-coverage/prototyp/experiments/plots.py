"""Erzeugt alle Diagramme der Arbeit als PDF (thesis/bilder/plots)."""
from __future__ import annotations

import json
import os
import sys

import locale
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.colors import LogNorm               # noqa: E402
from scipy.stats import spearmanr                   # noqa: E402

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "..", "thesis", "bilder", "plots")
os.makedirs(OUT, exist_ok=True)

# Referenzpalette (hell): feste Zuordnung Entitaet -> Farbe
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948"}
INK, INK2, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#8f8e89"
MODEL_COLORS = {"Analytisch": C["blue"], "Hybrid": C["orange"], "Gradient Boosting": C["aqua"],
                "Random Forest": C["yellow"], "Lineare Regression": C["magenta"], "Baseline": GRAY}
MODEL_ORDER = ["Analytisch", "Hybrid", "Gradient Boosting", "Random Forest",
               "Lineare Regression", "Baseline"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
W = 6.1   # Textbreite in Zoll

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "lines.linewidth": 2,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.axisbelow": True, "axes.formatter.use_locale": True,
})
try:                                    # Dezimalkomma in Achsenbeschriftungen
    locale.setlocale(locale.LC_NUMERIC, "de_DE.UTF-8")
except locale.Error:
    pass


def de(x: float, fmt: str = ".1f") -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return format(x, fmt).replace(".", ",")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("geschrieben:", name)


def load_programs():
    return [json.loads(l) for l in open(os.path.join(DATA, "programs.jsonl"))]


def program_frame(progs):
    rows = []
    for p in progs:
        rows.append({"id": p["id"], **p["metrics"], **{f"spec_{k}": v for k, v in p["spec"].items()},
                     "max_cov": p["max_cov"], "t_gen": p["t_gen"], "t_est": p["t_est"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def fig_saturation():
    n = np.logspace(0, 4, 300)
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.3))
    ps = [0.5, 0.1, 0.01, 0.001, 0.0001]
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"]]
    for p, c in zip(ps, cols):
        y = 1 - (1 - p) ** n
        ax.plot(n, y, color=c, label="$p_b = " + format(p, "g").replace(".", "{,}") + "$")
    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Zufallstests $N$")
    ax.set_ylabel("P(Zweig erreicht)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "saettigung.pdf")


def fig_pmf_example():
    """Illustration des analytischen Schaetzers: a, b ~ U[-10, 10], t = a + b, Bedingung t > 5."""
    from ppgen.estimator import PMF
    a = PMF.uniform(-10, 10)
    t = a.add(a)
    fig, axs = plt.subplots(1, 3, figsize=(W, 1.9), sharey=False)
    axs[0].bar(a.values(), a.p, width=0.8, color=C["blue"])
    axs[0].set_title("(a) $a, b \\sim U\\{-10,\\dots,10\\}$")
    tv = t.values()
    axs[1].bar(tv[tv <= 5], t.p[tv <= 5], width=0.8, color=GRAY)
    axs[1].bar(tv[tv > 5], t.p[tv > 5], width=0.8, color=C["orange"])
    pt = t.p[tv > 5].sum()
    axs[1].set_title(f"(b) $t = a + b$,  P$(t > 5) = {de(pt, '.3f')}$")
    # exakte Randverteilung von a unter t > 5
    pc = np.array([1 - a.p_lt(5 - v + 1) for v in a.values()])
    post = a.p * pc / np.sum(a.p * pc)
    postF = a.p * (1 - pc) / np.sum(a.p * (1 - pc))
    axs[2].bar(a.values() - 0.2, post, width=0.4, color=C["orange"], label="wahr")
    axs[2].bar(a.values() + 0.2, postF, width=0.4, color=GRAY, label="falsch")
    axs[2].set_title("(c) Randverteilung von $a$ je Ausgang")
    axs[2].legend(loc="upper left")
    axs[2].set_ylim(0, 0.17)
    for ax in axs:
        ax.set_xlabel("Wert")
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("Wahrscheinlichkeit")
    fig.tight_layout(w_pad=1.0)
    save(fig, "pmf_beispiel.pdf")


def fig_naive(df):
    nb = json.load(open(os.path.join(DATA, "naive_baseline.json")))
    hist = np.zeros((25, 9))
    for k, v in nb["hist_cc_depth"].items():
        cc, d = map(int, k.split(","))
        if 1 <= cc <= 25 and d <= 8:
            hist[cc - 1, d] += v
    prob = hist / nb["n_samples"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(SEQ)
    cmap.set_bad("#f4f3f0")
    m = np.ma.masked_where(prob.T == 0, prob.T)
    im = ax.pcolormesh(np.arange(0.5, 26), np.arange(-0.5, 9), m, cmap="Blues",
                       norm=LogNorm(vmin=1e-6, vmax=prob.max()), edgecolors="none", rasterized=True)
    ax.set_xlabel("zyklomatische Komplexität $V(G)$")
    ax.set_ylabel("Schachtelungstiefe")
    ax.set_title("(a) Naiver Generator: P(V(G), Tiefe)")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax = axs[1]
    rates = np.array(nb["rates"])
    trials = np.where(rates > 0, 1 / np.maximum(rates, 1e-12), np.nan)
    finite = trials[np.isfinite(trials)]
    bins = np.logspace(0, 6, 25)
    ax.hist(finite, bins=bins, color=C["blue"], edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("erwartete Versuche je Treffer")
    ax.set_ylabel("Anzahl Spezifikationen")
    ax.set_title("(b) Generate-and-Test-Aufwand")
    never = np.mean(rates == 0)
    ax.text(0.97, 0.95, f"nie getroffen:\n{de(never * 100)} % der Spez.", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK2)
    fig.tight_layout(w_pad=1.2)
    save(fig, "naiv_vs_konstruktiv.pdf")


def fig_properties(pdf):
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4))
    ax = axs[0]
    groups = [pdf[pdf["cc"] == c]["loc"].to_numpy() for c in range(2, 26)]
    bp = ax.boxplot(groups, positions=range(2, 26), widths=0.6, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "linewidth": 1})
    for b in bp["boxes"]:
        b.set_facecolor("#9ec5f4")
        b.set_edgecolor(C["blue"])
        b.set_linewidth(0.7)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(C["blue"])
        w.set_linewidth(0.7)
    ax.set_xticks([2, 5, 10, 15, 20, 25])
    ax.set_xticklabels([2, 5, 10, 15, 20, 25])
    ax.set_xlabel("vorgegebene zyklomatische Komplexität")
    ax.set_ylabel("Codezeilen (LOC)")
    ax.set_title("(a) Emergente Größe")
    ax = axs[1]
    for col, mcol, c, lab in [("spec_p_eq", "frac_eq", C["orange"], "Gleichheit"),
                              ("spec_p_mod", "frac_mod", C["aqua"], "Modulo"),
                              ("spec_p_rel2", "frac_rel2", C["violet"], "Zwei Variablen")]:
        bins = np.linspace(0, pdf[col].max(), 9)
        idx = np.digitize(pdf[col], bins) - 1
        xs, ys, lo, hi = [], [], [], []
        for k in range(len(bins) - 1):
            v = pdf[mcol][idx == k]
            if len(v) > 20:
                xs.append((bins[k] + bins[k + 1]) / 2)
                ys.append(v.mean())
                lo.append(v.quantile(0.25))
                hi.append(v.quantile(0.75))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(xs, ys, color=c, marker="o", markersize=3.5, label=lab)
    ax.plot([0, 0.5], [0, 0.5], color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Spezifikationsparameter (Wahrscheinlichkeit)")
    ax.set_ylabel("gemessener Anteil Bedingungen")
    ax.set_title("(b) Weiche Eigenschaften")
    ax.legend(loc="upper left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "eigenschaften.pdf")


def coverage_frame():
    cov = pd.read_csv(os.path.join(DATA, "coverage.csv"))
    return cov.pivot_table(index=["id", "budget"], columns="strategy", values="coverage").reset_index()


def fig_curves(pdf, cov):
    m = cov.merge(pdf[["id", "cc", "spec_domain"]], on="id")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.4), sharey=True)
    ax = axs[0]
    for (lo, hi), c in zip([(2, 8), (9, 16), (17, 25)], [C["blue"], C["orange"], C["aqua"]]):
        g = m[(m["cc"] >= lo) & (m["cc"] <= hi)].groupby("budget")["random"]
        med = g.median()
        ax.fill_between(med.index, g.quantile(0.25), g.quantile(0.75), color=c, alpha=0.15, lw=0)
        ax.plot(med.index, med, color=c, label=f"V(G) {lo}–{hi}")
    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Zufallstests $N$")
    ax.set_ylabel("Zweigüberdeckung")
    ax.set_title("(a) nach Komplexität (Zufallstests)")
    ax.legend(loc="lower right")
    ax = axs[1]
    for dom, c in zip([10, 100, 1000], [C["blue"], C["orange"], C["aqua"]]):
        g = m[m["spec_domain"] == dom].groupby("budget")
        ax.plot(g["random"].mean().index, g["random"].mean(), color=c, label=f"±{dom} zufällig")
        ax.plot(g["seeded"].mean().index, g["seeded"].mean(), color=c, linestyle="--",
                linewidth=1.4, label=f"±{dom} mit Seeding")
    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Tests $N$")
    ax.set_title("(b) nach Eingabebereich (Mittelwert)")
    ax.legend(loc="lower right", ncol=2, fontsize=7, columnspacing=0.8, handlelength=1.6)
    ax.set_ylim(0, 1.02)
    fig.tight_layout(w_pad=1.0)
    save(fig, "coverage_kurven.pdf")


LABELS = {"n_params": "Parameter", "log_domain": "log Eingabebereich", "loc": "LOC",
          "n_stmts": "Anweisungen", "cc": "V(G)", "max_depth": "Schachtelungstiefe",
          "n_loops": "Schleifen", "n_for": "for-Schleifen", "n_while": "while-Schleifen",
          "n_if": "if/elif", "n_elif": "elif", "n_else": "else", "n_boolops": "bool. Operatoren",
          "n_and": "and", "n_or": "or", "n_branches": "Zweige", "n_locals": "lokale Variablen",
          "n_return_early": "vorzeitige returns", "n_cond_rel": "Vergleiche mit Konstante",
          "n_cond_eq": "Gleichheitsbed.", "n_cond_mod": "Modulo-Bed.",
          "n_cond_rel2": "Vergleiche zweier Var.", "frac_eq": "Anteil Gleichheit",
          "frac_mod": "Anteil Modulo", "frac_rel2": "Anteil Zwei-Var.",
          "mean_decision_depth": "mittl. Entscheidungstiefe", "n_consts": "Konstanten",
          "log_budget": "log Testbudget N"}


def fig_correlations(pdf, cov):
    m = cov[cov["budget"] == 100].merge(pdf, on="id")
    m["log_domain"] = np.log10(m["spec_domain"])
    feats = ["cc", "max_depth", "loc", "n_branches", "n_loops", "n_params", "log_domain",
             "n_cond_eq", "frac_eq", "n_cond_mod", "n_cond_rel2", "n_boolops", "n_and", "n_or",
             "n_return_early", "mean_decision_depth"]
    rho = [(LABELS[f], spearmanr(m[f], m["random"]).statistic) for f in feats]
    rho.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(W * 0.7, 3.0))
    ys = np.arange(len(rho))
    ax.barh(ys, [r for _, r in rho], color=[C["red"] if r < 0 else C["blue"] for _, r in rho],
            height=0.65)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in rho])
    ax.axvline(0, color=INK2, linewidth=0.7)
    ax.set_xlabel("Spearman-Korrelation mit Coverage ($N = 100$)")
    ax.grid(axis="y", visible=False)
    for y, (_, r) in zip(ys, rho):
        ax.text(r + (0.01 if r >= 0 else -0.01), y, f"{r:+.2f}", va="center",
                ha="left" if r >= 0 else "right", fontsize=7, color=INK2)
    ax.set_xlim(-0.75, 0.45)
    save(fig, "korrelationen.pdf")
    return dict(rho)


def fig_calibration():
    br = pd.read_csv(os.path.join(DATA, "branches.csv"))
    f = br[(br["p_emp"] > 0)]
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    x = np.clip(f["p_pred"], 1e-6, 1)
    y = np.clip(f["p_emp"], 1e-6, 1)
    hb = ax.hexbin(np.log10(x), np.log10(y), gridsize=45, cmap="Blues", bins="log", mincnt=1,
                   linewidths=0)
    ax.plot([-6, 0], [-6, 0], color=C["orange"], linewidth=1)
    ax.set_xlabel("vorhergesagt $\\log_{10} \\hat p_b$")
    ax.set_ylabel("empirisch $\\log_{10} p_b$")
    ax.set_title("(a) Zweigwahrscheinlichkeiten")
    ax.set_xlim(-6.2, 0.1)
    ax.set_ylim(-5, 0.1)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)
    ax = axs[1]
    inf = br[br["feasible"] == 0]["p_pred"]
    fea = br[br["feasible"] == 1]["p_pred"]
    bins = np.logspace(-8, 0, 33)
    ax.hist(np.clip(fea, 1e-8, 1), bins=bins, color=C["blue"], alpha=0.85, label="erreicht",
            edgecolor="white", linewidth=0.4)
    ax.hist(np.clip(inf, 1e-8, 1), bins=bins, color=C["red"], histtype="step",
            label="nie erreicht", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("vorhergesagt $\\hat p_b$ (0 bei $10^{-8}$)")
    ax.set_ylabel("Anzahl Zweige")
    ax.set_title("(b) Erkennung unerreichter Zweige")
    ax.legend(loc="upper center")
    fig.tight_layout(w_pad=1.2)
    save(fig, "kalibrierung.pdf")


def fig_pred_vs_meas():
    oof = pd.read_csv(os.path.join(DATA, "model_oof.csv"))
    d = oof[oof["budget"] == 100]
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.2), sharey=True)
    for ax, name in zip(axs, ["Gradient Boosting", "Analytisch", "Hybrid"]):
        ax.hexbin(d[name], d["target"], gridsize=35, cmap="Blues", bins="log", mincnt=1,
                  extent=(0, 1, 0, 1), linewidths=0)
        ax.plot([0, 1], [0, 1], color=C["orange"], linewidth=1)
        mae = np.mean(np.abs(d[name] - d["target"]))
        ax.set_title(f"{name}")
        ax.text(0.04, 0.93, f"MAE = {de(mae * 100)} PP", transform=ax.transAxes, fontsize=8,
                va="top", color=INK2)
        ax.set_xlabel("vorhergesagte Coverage")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    axs[0].set_ylabel("gemessene Coverage")
    fig.tight_layout(w_pad=0.8)
    save(fig, "vorhersage_vs_messung.pdf")


def fig_mae_budget(res):
    budgets = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    fig, ax = plt.subplots(figsize=(W * 0.72, 2.4))
    for name in MODEL_ORDER:
        ys = [res["cv"][name][str(b)]["MAE"] * 100 for b in budgets]
        ax.plot(budgets, ys, color=MODEL_COLORS[name], marker="o", markersize=3.5,
                linewidth=1.6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_yticklabels(["0,1", "0,3", "1", "3", "10", "30"])
    ax.set_ylim(0.1, 40)
    ax.set_xlabel("Testbudget $N$")
    ax.set_ylabel("MAE [Prozentpunkte, log.]")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, "mae_budget.pdf")


def fig_importance(res):
    imp = res["importance"][:10][::-1]
    fig, ax = plt.subplots(figsize=(2.9, 2.7))
    ys = np.arange(len(imp))
    ax.barh(ys, [d["mean"] * 100 for d in imp], xerr=[d["std"] * 100 for d in imp],
            color=C["aqua"], height=0.65, error_kw={"elinewidth": 0.7, "ecolor": INK2})
    ax.set_yticks(ys)
    ax.set_yticklabels([LABELS.get(d["feature"], d["feature"]) for d in imp])
    ax.set_xlabel("Anstieg MAE [PP]")
    ax.grid(axis="y", visible=False)
    save(fig, "wichtigkeit.pdf")


def fig_learning(res):
    lc = res["learning_curve"]
    ns = [d["n"] for d in lc]
    fig, ax = plt.subplots(figsize=(2.9, 2.7))
    for name in ["Lineare Regression", "Random Forest", "Gradient Boosting", "Hybrid"]:
        ax.plot(ns, [d[name] * 100 for d in lc], color=MODEL_COLORS[name], marker="o",
                markersize=3.5, linewidth=1.6, label=name)
    ax.axhline(np.mean([d["Analytisch"] for d in lc]) * 100, color=MODEL_COLORS["Analytisch"],
               linewidth=1.4, linestyle="--", label="Analytisch")
    ax.set_xscale("log")
    ax.set_xlabel("Anzahl Trainingsprogramme")
    ax.set_ylabel("MAE [PP]")
    ax.set_ylim(0, 22)
    ax.legend(loc="upper center", ncol=2, fontsize=7, columnspacing=0.8, handlelength=1.5)
    save(fig, "lernkurve.pdf")


def fig_generalization(res):
    scen = [("cv", "Kreuzvalidierung"), ("extrapolation_cc", "V(G) ≥ 13\n(Training ≤ 12)"),
            ("extrapolation_domain", "Bereich ±1000\n(Training ≤ ±100)"),
            ("casestudy", "Lehrbuch-\nfunktionen")]
    names = ["Analytisch", "Hybrid", "Gradient Boosting", "Random Forest", "Lineare Regression"]
    fig, ax = plt.subplots(figsize=(W, 2.4))
    x = np.arange(len(scen))
    wdt = 0.15
    for k, name in enumerate(names):
        vals = [res[s][name]["all"]["MAE"] * 100 for s, _ in scen]
        ax.bar(x + (k - 2) * wdt, vals, width=wdt * 0.9, color=MODEL_COLORS[name], label=name)
        if name in ("Analytisch", "Hybrid"):
            for xi, v in zip(x, vals):
                ax.text(xi + (k - 2) * wdt, v + 0.3, de(v), ha="center", va="bottom",
                        fontsize=6.5, color=INK2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in scen])
    ax.set_ylabel("MAE [Prozentpunkte]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
    save(fig, "generalisierung.pdf")


def fig_targeted(cov):
    tg = json.load(open(os.path.join(DATA, "targeted.json")))
    base = cov[cov["budget"] == 100]["random"].to_numpy()
    fig, ax = plt.subplots(figsize=(W * 0.8, 2.5))
    data = [base] + [np.array([p["measured"] for p in t["programs"]]) for t in tg]
    labels = ["ohne\nFilter"] + [f"Ziel\n{de(t['target'])}" for t in tg]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=True,
                    medianprops={"color": INK, "linewidth": 1},
                    flierprops={"markersize": 2.5, "markeredgecolor": GRAY})
    for k, b in enumerate(bp["boxes"]):
        b.set_facecolor("#e4e3df" if k == 0 else "#9ec5f4")
        b.set_edgecolor(GRAY if k == 0 else C["blue"])
    for k, t in enumerate(tg):
        ax.plot([k + 1.65, k + 2.35], [t["target"]] * 2, color=C["orange"], linewidth=1.6)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("gemessene Coverage ($N = 100$)")
    ax.grid(axis="x", visible=False)
    ax.plot([], [], color=C["orange"], linewidth=1.6, label="Zielwert")
    ax.legend(loc="lower right")
    save(fig, "zielsteuerung.pdf")


def main():
    progs = load_programs()
    pdf = program_frame(progs)
    cov = coverage_frame()
    res = json.load(open(os.path.join(DATA, "model_results.json")))
    fig_saturation()
    fig_pmf_example()
    fig_naive(pdf)
    fig_properties(pdf)
    fig_curves(pdf, cov)
    rho = fig_correlations(pdf, cov)
    json.dump(rho, open(os.path.join(DATA, "spearman.json"), "w"), indent=1)
    fig_calibration()
    fig_pred_vs_meas()
    fig_mae_budget(res)
    fig_importance(res)
    fig_learning(res)
    fig_generalization(res)
    fig_targeted(cov)


if __name__ == "__main__":
    main()
