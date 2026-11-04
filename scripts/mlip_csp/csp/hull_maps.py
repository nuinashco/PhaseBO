"""Energy-above-hull maps of the Li-Sn-S-Cl field, DFT vs MLIP-CSP: csp/out/<model>/figures/fig5_hull_maps.png.

Each composition sits at (Li/(Li+Sn), S/(S+Cl)), the square PhaseBO plots its phase field on. Colour is the energy
above the hull, interpolated between compositions; the third panel is MLIP minus DFT.
Run: python -m csp.hull_maps --model fastchgnet-0.3.0 [--column MLIP_b3]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pymatgen.core import Composition
from scipy.interpolate import griddata

from csp.common import DEFAULT_MODEL, MODELS, model_dir
from csp.meeting_figures import STABLE_MARKERS, save
from csp.summary_figure import INK, INK2, SURFACE, budgets

CAP = 200  # meV/atom; the colour scale saturates above
# sequential blue ramp, darkest = most stable (0 meV/atom), light = unstable
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6", "#f0efec"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#f0efec", "#e34948"])  # blue < 0 < red


def square(formula):
    c = Composition(formula)
    li, sn, s, cl = (c[el] for el in ("Li", "Sn", "S", "Cl"))
    return li / (li + sn), s / (s + cl)


def panel(ax, xy, values, cmap, norm, title, stable, labels):
    gx, gy = np.mgrid[0:1:200j, 0:1:200j]
    grid = griddata(xy, values, (gx, gy), method="linear")
    mesh = ax.pcolormesh(gx, gy, grid, cmap=cmap, norm=norm, shading="auto")
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color=SURFACE, edgecolor=INK2, lw=0.4, zorder=3)
    for k, i in enumerate(stable):
        ax.scatter(*xy[i], s=170, marker=STABLE_MARKERS[k], color="#eda100", edgecolor=INK, lw=0.9, zorder=4)
        if labels:
            # Li3SnS3Cl and Li5SnS4Cl sit close together: the first label goes below-left
            offset = (-20, -22) if k == 0 else (8, 8)
            ax.annotate(labels[k], xy[i], xytext=offset, textcoords="offset points", fontsize=9.5, color=INK,
                        ha="right" if k == 0 else "left",
                        fontweight="bold",
                        bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("Li / (Li + Sn)", color=INK2)
    ax.set_ylabel("S / (S + Cl)", color=INK2)
    ax.tick_params(colors=INK2)
    ax.set_title(title, fontsize=12, color=INK, loc="left")
    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--column", default=None, help="MLIP_b<b>; default: the largest budget in the table")
    args = ap.parse_args()
    table = pd.read_csv(model_dir(args.model) / "LiSnSCl_mlipcsp.csv")
    col = args.column or f"MLIP_b{budgets(table)[-1]}"
    t = table.dropna(subset=[col]).reset_index(drop=True)
    xy = np.array([square(f) for f in t.Formula])
    stable = list(t.index[t.DFT == 0])
    labels = [t.Formula[i] for i in stable]

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    seq = plt.Normalize(0, CAP)
    m1 = panel(axes[0], xy, t.DFT.clip(upper=CAP).to_numpy(), SEQUENTIAL, seq,
               "A. DFT (the truth)", stable, labels)
    panel(axes[1], xy, t[col].clip(upper=CAP).to_numpy(), SEQUENTIAL, seq,
          f"B. MLIP-CSP after {col[len('MLIP_b'):]} trials", stable, labels)
    diff = (t[col] - t.DFT).to_numpy()
    lim = float(np.nanpercentile(np.abs(diff), 95))
    m3 = panel(axes[2], xy, np.clip(diff, -lim, lim), DIVERGING, TwoSlopeNorm(0, -lim, lim),
               "C. MLIP − DFT (red: MLIP too pessimistic)", stable, None)
    c1 = fig.colorbar(m1, ax=axes[:2], shrink=0.85, pad=0.02)
    c1.set_label(f"Energy above hull (meV/atom; dark = stable, ≥{CAP} saturated)", color=INK2)
    c3 = fig.colorbar(m3, ax=axes[2], shrink=0.85, pad=0.03)
    c3.set_label("MLIP − DFT (meV/atom)", color=INK2)
    for c in (c1, c3):
        c.ax.tick_params(colors=INK2)
    fig.suptitle("Where in Li-Sn-S-Cl the compounds are stable: DFT vs MLIP-CSP "
                 "(white dots = compositions, orange = the 3 DFT-stable phases)",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    out = model_dir(args.model) / "figures"
    out.mkdir(exist_ok=True)
    save(fig, out / "fig5_hull_maps.png")


if __name__ == "__main__":
    main()
