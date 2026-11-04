"""How reliable is MLIP-CSP at k trials? Reshuffle the trial order: csp/out/<model>/reshuffle.csv and
csp/out/<model>/figures/fig6_reliability.png.

One CSP run gives one order of trials, so the result at k trials is one draw: a lucky early trial can find a stable
phase at k = 10 that another order would only find at k = 30. Here each composition's trials are reordered at random
(REPS times), the best of the first k is put on the MLIP hull (references as in csp.hull), and we record Spearman vs
DFT and the DFT lookups of "MLIP-screen everything, then DFT the best" to find all DFT-stable phases. That method
runs DFT in MLIP order, so its lookups are the worst MLIP rank among the stable phases.
Run: python -m csp.reshuffle --model fastchgnet-0.3.0 [--reps 200] [--target 50]
"""
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from pymatgen.analysis.phase_diagram import PhaseDiagram
from scipy.stats import spearmanr

from csp.common import DEFAULT_MODEL, MODELS, OUT, model_dir, usable
from csp.hull import entry, reference_entries
from csp.meeting_figures import save
from csp.run_csp import csp_dir
from csp.summary_figure import INK, INK2, SURFACE, style

DFT_ONLY = 106  # DFT lookups of DFT-only PhaseBO to find all 3 stable phases (play/out_mlip/<model>/hull_recovery.csv)


def trial_energies(cells, model):
    """Energy per atom of every trial (inf for unusable ones), trimmed to the trial count all cells share."""
    out = {}
    for cell in cells:
        rows = {}
        for line in (csp_dir(model) / f"{cell}.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["trial"]] = r["energy"] if usable(r) else np.inf
        n = next(t for t in range(len(rows) + 1) if t not in rows)
        out[cell] = np.array([rows[t] for t in range(n)])
    n = min(len(e) for e in out.values())
    return {c: e[:n] for c, e in out.items()}, n


def evaluate(best, comp, refs):
    quats = {c: entry(c, e) for c, e in best.items() if np.isfinite(e)}
    pd_ = PhaseDiagram(refs + list(quats.values()))
    ehull = np.array([max(1000 * pd_.get_e_above_hull(quats[c]), 0) if c in quats else np.inf
                      for c in comp.csp_cell])
    ok = np.isfinite(ehull)
    rho = spearmanr(comp.dft_ehull[ok], ehull[ok])[0]
    ranks = pd.Series(ehull).rank(method="min").to_numpy()
    return rho, int(ranks[comp.dft_ehull.to_numpy() == 0].max())


def plot(df, n, actual, target, path):
    ks = sorted(df.trials.unique())
    g = df.groupby("trials").greedy_lookups
    p = [100 * (g.get_group(k) <= target).mean() for k in ks]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.2))
    a1.plot(ks, p, color="#1baf7a", lw=2, marker="^", ms=9, mec=SURFACE, mew=1.5)
    for k, v in zip(ks, p):
        a1.annotate(f"{v:.0f}%", (k, v), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=10,
                    color=INK)
    a1.set_ylim(0, 105)
    a1.set_ylabel(f"Chance of finding all 3 within {target} DFT lookups (%)", color=INK2)
    a1.set_title("How often MLIP-CSP pays off, by CSP trials", fontsize=12, color=INK, loc="left")
    lo, med, hi = (g.quantile(q).reindex(ks).to_numpy() for q in (0.05, 0.5, 0.95))
    a2.fill_between(ks, lo, hi, color="#1baf7a", alpha=0.18, lw=0)
    a2.plot(ks, med, color="#1baf7a", lw=2, marker="^", ms=9, mec=SURFACE, mew=1.5)
    a2.plot([n], [actual], ls="none", marker="^", ms=9, color="#1baf7a", mec=INK, mew=1)
    a2.annotate(f"all {n} trials: {actual}", (n, actual), xytext=(0, 12), textcoords="offset points", ha="center",
                fontsize=9.5, color=INK)
    a2.text(ks[0], hi[0] + 3, "5–95% of trial orders", fontsize=9.5, color=INK2)
    a2.axhline(DFT_ONLY, color=INK2, lw=1, ls="--")
    a2.text(n, DFT_ONLY + 3, f"DFT-only PhaseBO: {DFT_ONLY}", ha="right", fontsize=9.5, color=INK2)
    a2.axhline(target, color=INK2, lw=1, ls=":")
    a2.set_ylim(0, 175)
    a2.set_ylabel("DFT lookups to find all 3 stable phases (median)", color=INK2)
    a2.set_title("MLIP-screen everything, then DFT the best", fontsize=12, color=INK, loc="left")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ticks = ks + [n]
        ax.set_xticks(ticks, [str(k) for k in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("CSP trials per composition", color=INK2)
        style(ax)
    fig.suptitle(f"Reliability: {df.groupby('trials').size().iloc[0]} random reorderings of the {n} trials "
                 "each composition has", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--target", type=int, default=50, help="DFT lookups counted as a success")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    comp = pd.read_csv(OUT / "compositions.csv")
    energies, n = trial_energies(comp.csp_cell, args.model)
    refs = reference_entries(args.model)
    ks = [k for k in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100) if k < n]
    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.reps):
        perms = {c: rng.permutation(n) for c in comp.csp_cell}
        for k in ks:
            rows.append((k, *evaluate({c: e[perms[c][:k]].min() for c, e in energies.items()}, comp, refs)))
    df = pd.DataFrame(rows, columns=["trials", "rho", "greedy_lookups"])
    df.to_csv(model_dir(args.model) / "reshuffle.csv", index=False)
    rho_n, actual = evaluate({c: e.min() for c, e in energies.items()}, comp, refs)
    for k, d in df.groupby("trials"):
        print(f"{k:3d} trials: rho {d.rho.mean():.2f} [{d.rho.quantile(0.05):.2f}-{d.rho.quantile(0.95):.2f}], "
              f"greedy lookups median {d.greedy_lookups.median():.0f} [{d.greedy_lookups.quantile(0.05):.0f}-"
              f"{d.greedy_lookups.quantile(0.95):.0f}], <= {args.target} in "
              f"{100 * (d.greedy_lookups <= args.target).mean():.0f}%")
    print(f"{n:3d} trials: rho {rho_n:.2f}, greedy lookups {actual}")
    out = model_dir(args.model) / "figures"
    out.mkdir(exist_ok=True)
    plot(df, n, actual, args.target, out / "fig6_reliability.png")


if __name__ == "__main__":
    main()
