"""Separate figures of the MLIP-CSP fidelity run, one point each: csp/out/<model>/figures/fig1..fig4.png.

fig1: MLIP-CSP E_hull against DFT E_hull, stable phases named; fig2: how the fidelity changes with CSP trials
(rank correlation, and the MLIP rank of each DFT-stable phase); fig3: DFT lookups to find all stable phases and the
first one, by method; fig4: the MLIPs tried, accuracy against Materials Project and speed.
Run: python -m csp.meeting_figures --model fastchgnet-0.3.0 [--budget 2]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from scipy.stats import spearmanr

from csp.common import DEFAULT_MODEL, MODELS, ROOT, model_dir
from csp.summary_figure import EARLIER, GRID, INK, INK2, METHODS, SURFACE, budgets, style

STABLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3, one per DFT-stable phase
STABLE_MARKERS = ["o", "s", "^"]
# relative CSP speed on CPU, from the benchmarks on identical starting structures (MACE-MP-0 = 1)
SPEED = {"mace-medium": 1, "chgnet": 4, "fastchgnet-0.3.0": 6, "fastchgnet": 35}


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {path}")


def fig_scatter(table, path):
    b = budgets(table)[-1]
    col = f"MLIP_b{b}"
    t = table.dropna(subset=[col])
    rho = spearmanr(t.DFT, t[col])[0]
    cap = 300
    fig, ax = plt.subplots(figsize=(8, 7))
    stable = t[t.DFT == 0]
    rest = t[t.DFT != 0]
    ax.scatter(rest.DFT.clip(upper=cap), rest[col].clip(upper=cap), s=26, color="#86b6ef", edgecolor=SURFACE,
               lw=0.6, label=f"other compositions ({len(rest)})")
    for k, (_, r) in enumerate(stable.iterrows()):
        ax.scatter(0, min(r[col], cap), s=150, marker=STABLE_MARKERS[k], color=STABLE_COLORS[k], edgecolor=INK,
                   lw=0.7, zorder=3)
        ax.annotate(f"{r.Formula}: MLIP {r[col]:.0f} meV", (0, min(r[col], cap)), xytext=(14, 0),
                    textcoords="offset points", va="center", fontsize=10, color=INK)
    ax.plot([0, cap], [0, cap], color=INK2, lw=1, ls="--")
    ax.text(cap * 0.62, cap * 0.55, "MLIP = DFT", rotation=40, color=INK2, fontsize=9)
    ax.set_xlim(-8, cap + 5)
    ax.set_ylim(-8, cap + 5)
    ax.set_xlabel("DFT energy above hull (meV/atom) — the truth", color=INK2)
    ax.set_ylabel(f"MLIP-CSP energy above hull after {b} trials (meV/atom)", color=INK2)
    ax.set_title(f"How well does MLIP-CSP predict stability?  Spearman ρ = {rho:.2f}\n"
                 "Stable phases (named) should sit near 0 on both axes", fontsize=12, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    style(ax)
    fig.tight_layout()
    save(fig, path)


def fig_budget(table, path):
    bs = budgets(table)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    rho = [spearmanr(*table[["DFT", f"MLIP_b{b}"]].dropna().to_numpy().T)[0] for b in bs]
    a1.plot(bs, rho, color="#1baf7a", lw=2.5, marker="^", ms=9, mec=SURFACE, label="MLIP-CSP (this run)")
    for (name, r), ls in zip(EARLIER.items(), (":", "--", "-.")):
        a1.axhline(r, color=INK2, lw=1, ls=ls)
        a1.text(bs[-1], r + 0.015, name, ha="right", fontsize=9, color=INK2)
    a1.set_ylim(-0.05, 1)
    a1.set_ylabel("Spearman ρ with DFT E_hull (1 = same ranking)", color=INK2)
    a1.set_title("Ranking agreement with DFT, by CSP trials", fontsize=12, color=INK, loc="left")
    stable = table[table.DFT == 0].Formula.tolist()
    n = len(table)
    for k, f in enumerate(stable):
        ranks = []
        for b in bs:
            col = table[f"MLIP_b{b}"].fillna(np.inf)
            ranks.append(int(col.rank(method="min")[table.Formula == f].iloc[0]))
        a2.plot(bs, ranks, color=STABLE_COLORS[k], marker=STABLE_MARKERS[k], lw=2.5, ms=9, mec=SURFACE, label=f)
        a2.annotate(f"{ranks[-1]}", (bs[-1], ranks[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=10, color=INK)
    a2.set_ylim(n + 5, -5)  # rank 1 at the top
    a2.set_ylabel(f"MLIP rank among {n} compositions (1 = most stable)", color=INK2)
    a2.set_title("Where MLIP-CSP ranks the three DFT-stable phases", fontsize=12, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=10, loc="center right")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(bs, [str(b) for b in bs])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("CSP trials per composition", color=INK2)
        style(ax)
    fig.tight_layout()
    save(fig, path)


def fig_lookups(model, budget, path):
    df = pd.read_csv(ROOT / "play" / "out_mlip" / model / "hull_recovery.csv")
    d = df[df["CSP budget"] == budget].set_index("method")
    methods = [m for m in METHODS if m in d.index]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=False)
    y = np.arange(len(methods))[::-1]
    for ax, col, ci, title in [(a1, "DFT lookups", "±95%", "…to find ALL 3 stable phases"),
                               (a2, "lookups to 1st", None, "…to find the FIRST stable phase")]:
        vals = d.loc[methods, col].to_numpy()
        err = d.loc[methods, ci].to_numpy() if ci else None
        ax.barh(y, vals, xerr=err, height=0.6, color=[METHODS[m][0] for m in methods], edgecolor=SURFACE, lw=2,
                error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
        ends = vals + (np.nan_to_num(err) if err is not None else 0)
        for yi, v, end in zip(y, vals, ends):
            ax.text(end + max(ends) * 0.02, yi, f"{v:.0f}", va="center", fontsize=10, color=INK)
        runs = [int(d.loc[m, "runs"]) for m in methods]
        ax.set_yticks(y, [f"{METHODS[m][3]}\n({r} run{'s' if r != 1 else ''})" for m, r in zip(methods, runs)],
                      fontsize=9.5)
        ax.set_xlabel("DFT lookups (lower is better)", color=INK2)
        ax.set_title(title, fontsize=12, color=INK, loc="left")
        ax.set_xlim(0, max(ends) * 1.15)
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"DFT lookups needed, with MLIP-CSP at {budget} trials per composition "
                 f"({len(pd.read_csv(model_dir(model) / 'LiSnSCl_mlipcsp.csv'))} compositions, 3 stable)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, path)


def fig_models(path):
    rows = []
    for m in MODELS:
        p = model_dir(m) / "refs_mlip_vs_mp.csv"
        if p.exists():
            r = pd.read_csv(p)
            rows.append((m, r[r.formula.str.count("[A-Z]") > 1].error_meV.abs().mean()))
    names = [m for m, _ in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    y = np.arange(len(rows))[::-1]
    chosen = "fastchgnet-0.3.0"
    colors = ["#2a78d6" if m == chosen else "#b7d3f6" for m in names]
    a1.barh(y, [e for _, e in rows], height=0.6, color=colors, edgecolor=SURFACE, lw=2)
    for yi, (_, e) in zip(y, rows):
        a1.text(e + 6, yi, f"{e:.0f} meV/atom", va="center", fontsize=10, color=INK)
    a1.set_xlabel("Formation-energy error vs Materials Project (MAE, lower is better)", color=INK2)
    a1.set_title("Accuracy on 44 known Li-Sn-S-Cl compounds", fontsize=12, color=INK, loc="left")
    a1.set_xlim(0, max(e for _, e in rows) * 1.3)
    a2.barh(y, [SPEED[m] for m in names], height=0.6, color=colors, edgecolor=SURFACE, lw=2)
    for yi, m in zip(y, names):
        a2.text(SPEED[m] + 0.6, yi, f"~{SPEED[m]}×", va="center", fontsize=10, color=INK)
    a2.set_xlabel("CSP trials per hour on CPU, relative to MACE (higher is better)", color=INK2)
    a2.set_title("Speed", fontsize=12, color=INK, loc="left")
    a2.set_xlim(0, max(SPEED.values()) * 1.2)
    for ax in (a1, a2):
        ax.set_yticks(y, names)
        style(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle("MLIPs tried (dark blue = used for the run): same accuracy as CHGNet, ~6× MACE's speed",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--budget", type=int, default=2, help="CSP budget shown in fig3")
    args = ap.parse_args()
    out = model_dir(args.model) / "figures"
    out.mkdir(exist_ok=True)
    table = pd.read_csv(model_dir(args.model) / "LiSnSCl_mlipcsp.csv")
    fig_scatter(table, out / "fig1_fidelity_vs_dft.png")
    fig_budget(table, out / "fig2_fidelity_vs_trials.png")
    if (ROOT / "play" / "out_mlip" / args.model / "hull_recovery.csv").exists():  # the BO replays of PolyBO's play/
        fig_lookups(args.model, args.budget, out / "fig3_dft_lookups.png")
    fig_models(out / "fig4_models.png")


if __name__ == "__main__":
    main()
