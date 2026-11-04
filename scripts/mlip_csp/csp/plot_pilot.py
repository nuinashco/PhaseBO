"""Overview figure of the CSP pilot: csp/out/<model>/pilot_overview.png.  Run: python -m csp.plot_pilot [--model ...]"""
import json

import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Composition

from csp.common import DEFAULT_MODEL, MODELS, dict_to_atoms, model_dir, usable
from csp.report import cell_stats, estimate
from csp.run_csp import csp_dir

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dc", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # reference palette slots 1-4
MARKERS = ["o", "s", "^", "D"]


def trials(cell, model):
    rows = [json.loads(line) for line in (csp_dir(model) / f"{cell}.jsonl").read_text().splitlines() if line.strip()]
    return sorted(rows, key=lambda r: r["trial"])


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ac")
    ax.tick_params(colors=INK2)
    ax.set_facecolor(SURFACE)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    model = ap.parse_args().model
    from ase.visualize.plot import plot_atoms

    stats = cell_stats(model)
    stats = stats[stats.formula.isin(["Li3SnS3Cl", "Li5SnS4Cl", "Li4SnS4", "Li19SnS11Cl"])]
    stats = stats.sort_values("n_atoms").reset_index(drop=True)
    fig = plt.figure(figsize=(15, 9.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 3, left=0.06, right=0.98, top=0.86, bottom=0.08, hspace=0.42, wspace=0.28)
    labels = {r.cell: f"{r.formula} ({r.n_atoms} atoms)" for r in stats.itertuples()}

    # A: best-so-far, relative to the final best
    ax = fig.add_subplot(gs[0, 0])
    for k, cell in enumerate(stats.cell):
        e = np.array([r["energy"] if usable(r) else np.inf for r in trials(cell, model)])
        best = np.minimum.accumulate(e)
        rel = 1000 * (best - best[-1])
        ax.step(np.arange(1, len(e) + 1), np.where(np.isfinite(rel), rel, np.nan), where="post", lw=2,
                color=COLORS[k], marker=MARKERS[k], markevery=4, ms=7, mec=SURFACE, label=labels[cell])
    ax.set_xlabel("CSP trials", color=INK2)
    ax.set_ylabel("Best so far − final best (meV/atom)", color=INK2)
    ax.set_title("A. Best-so-far traces: how fast CSP converges", fontsize=11, color=INK, loc="left")
    ax.set_ylim(bottom=-5)
    style(ax)
    ax.legend(frameon=False, fontsize=9)

    # B: every trial's energy, usable vs discarded; off-scale trials are pinned to the edges
    ax = fig.add_subplot(gs[0, 1])
    lo, hi = -22.0, 0.0
    off = 0
    for k, cell in enumerate(stats.cell):
        rows = [r for r in trials(cell, model) if r["energy"] is not None]
        jitter = {r["trial"]: k + np.random.default_rng(r["trial"]).uniform(-0.25, 0.25) for r in rows}
        ok = [r for r in rows if usable(r)]
        bad = [r for r in rows if not usable(r)]
        off += sum(not lo <= r["energy"] <= hi for r in bad)
        ax.scatter([jitter[r["trial"]] for r in ok], [r["energy"] for r in ok], color=COLORS[k], marker=MARKERS[k],
                   s=40, edgecolor=SURFACE, lw=0.8, zorder=3, label="kept" if k == 0 else None)
        ax.scatter([jitter[r["trial"]] for r in bad], [np.clip(r["energy"], lo, hi) for r in bad], facecolor="none",
                   edgecolor="#e34948", marker="X", s=60, lw=1.2, zorder=3,
                   label="discarded: unconverged or atoms too close" if k == 0 else None)
    ax.set_xticks(range(len(stats)), [labels[c].replace(" (", "\n(") for c in stats.cell], fontsize=8.5)
    ax.set_ylim(lo - 0.5, hi + 0.5)
    ax.text(0.98, 0.04, f"{off} discarded trials far off-scale (pinned to the edges)", transform=ax.transAxes,
            ha="right", fontsize=8.5, color=INK2)
    ax.set_ylabel("Energy per atom (eV); real structures ≈ −4", color=INK2)
    ax.set_title("B. Every trial: MACE 'holes' are filtered out", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="center right")

    # C: time per trial vs cell size, with the full-run estimate
    ax = fig.add_subplot(gs[0, 2])
    k_exp, wall_h, cpu_h = estimate(stats, 50, 15)
    n = np.linspace(5, 45, 100)
    c = np.exp(np.polyfit(np.log(stats.n_atoms), np.log(stats.s_per_trial), 1)[1])
    ax.plot(n, c * n ** k_exp, color=INK2, lw=1.5, ls="--", label=f"fit ∝ atoms^{k_exp:.2f}")
    for k, r in enumerate(stats.itertuples()):
        ax.scatter(r.n_atoms, r.s_per_trial, color=COLORS[k], marker=MARKERS[k], s=70, edgecolor=SURFACE, zorder=3)
    ax.set_xlabel("Atoms in the cell (compositions range 6–44)", color=INK2)
    ax.set_ylabel("Seconds per trial (1 CPU core)", color=INK2)
    ax.set_title("C. Cost: bigger cells are much slower", fontsize=11, color=INK, loc="left")
    ax.text(0.03, 0.95, f"Full run, 50 trials × 165 compositions:\n~{cpu_h:.0f} core-hours ≈ {wall_h:.0f} h on 15 cores"
                        "\n→ planned on the T4 GPU", transform=ax.transAxes, va="top", fontsize=9.5, color=INK)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    # D-E: best predicted structures of the two stable quaternaries
    for col, cell in zip((0, 1), ("Li3Sn1S3Cl1", "Li5Sn1S4Cl1")):
        ax = fig.add_subplot(gs[1, col])
        ok = [r for r in trials(cell, model) if usable(r)]
        best = min(ok, key=lambda r: r["energy"])
        atoms = dict_to_atoms(best["atoms"]).repeat((2, 2, 1))
        plot_atoms(atoms, ax, rotation="10x,20y,0z", radii=0.5)
        ax.set_axis_off()
        ax.set_title(f"{'DE'[col]}. Best MLIP-CSP structure: {Composition(cell).reduced_formula}\n"
                     f"(DFT-stable; trial {best['trial'] + 1}, {best['energy']:.3f} eV/atom, 2×2 cells)",
                     fontsize=11, color=INK, loc="left")
    ax = fig.add_subplot(gs[1, 2])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for i, (el, color) in enumerate([("Li", "#cc80ff"), ("Sn", "#668080"), ("S", "#ffff30"), ("Cl", "#1ff01f")]):
        ax.scatter(0.05, 0.9 - 0.1 * i, s=200, color=color, edgecolor=INK2)
        ax.text(0.12, 0.9 - 0.1 * i, el, va="center", fontsize=11, color=INK)
    ax.text(0.0, 0.42, "Atom colours (ASE / Jmol).\n\nThese are MACE predictions, not yet checked with DFT.\n"
                       "Step 6 relaxes the best structures of 10–20\ncompositions with DFT.",
            va="top", fontsize=10, color=INK2)

    fig.suptitle(f"MLIP-CSP pilot on Li-Sn-S-Cl: {model} + random symmetric structures (pyxtal), 20 trials per cell",
                 x=0.06, y=0.96, ha="left", fontsize=14, color=INK, fontweight="bold")
    path = model_dir(model) / "pilot_overview.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
