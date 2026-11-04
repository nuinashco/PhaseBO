"""One-page summary of the MLIP-CSP fidelity run: csp/out/<model>/summary.png.

A: MLIP-CSP E_hull against DFT E_hull for every composition (largest CSP budget); B: rank correlation with DFT
against CSP budget, with the earlier NN column and synthetic NNs for scale; C: DFT lookups to recover the hull by
method and CSP budget (play/out_mlip/<model>/hull_recovery.csv); D: the MLIPs tried, accuracy and speed.
Run: python -m csp.summary_figure --model fastchgnet-0.3.0
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullFormatter
from scipy.stats import spearmanr

from csp.common import DEFAULT_MODEL, MODELS, OUT, ROOT, model_dir

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3dc", "#fcfcfb"
METHODS = {  # reference palette slots, line style, marker, label
    "random": ("#2a78d6", ":", "o", "Random DFT"),
    "sf": ("#eb6834", "--", "s", "DFT-only PhaseBO"),
    "greedy": ("#1baf7a", "-.", "^", "MLIP-screen everything, DFT the best"),
    "prescreen": ("#eda100", "-", "D", "MLIP-screen + MF GP"),
    "mfmes": ("#e87ba4", "-", "v", "Adaptive MF-MES"),
}
# rank correlation with DFT of the earlier cheap fidelities (play/hits.py), for scale in panel B
EARLIER = {"real NN column": 0.05, "synthetic NN, 30 meV": 0.75, "synthetic NN, 10 meV": 0.96}


def style(ax):
    ax.grid(color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ac")
    ax.tick_params(colors=INK2)
    ax.set_facecolor(SURFACE)


def budgets(table):
    return sorted(int(c[len("MLIP_b"):]) for c in table.columns if c.startswith("MLIP_b"))


def panel_scatter(ax, table):
    b = budgets(table)[-1]
    t = table.dropna(subset=[f"MLIP_b{b}"])
    rho = spearmanr(t.DFT, t[f"MLIP_b{b}"])[0]
    cap = 300
    stable = t.DFT == 0
    ax.scatter(t.DFT[~stable].clip(upper=cap), t[f"MLIP_b{b}"][~stable].clip(upper=cap), s=22, color="#2a78d6",
               alpha=0.7, edgecolor=SURFACE, lw=0.5, label="compositions")
    ax.scatter(t.DFT[stable], t[f"MLIP_b{b}"][stable].clip(upper=cap), s=110, marker="*", color="#e34948",
               edgecolor=INK, lw=0.6, zorder=3, label="DFT-stable (the targets)")
    ax.plot([0, cap], [0, cap], color=INK2, lw=1, ls="--")
    ax.set_xlim(-5, cap + 5)
    ax.set_ylim(-5, cap + 5)
    ax.set_xlabel("DFT E_hull (meV/atom)", color=INK2)
    ax.set_ylabel(f"MLIP-CSP E_hull after {b} trials (meV/atom)", color=INK2)
    ax.set_title(f"A. The new cheap fidelity vs DFT (Spearman ρ = {rho:.2f})", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    style(ax)


def panel_correlation(ax, table):
    bs = budgets(table)
    rho = [spearmanr(*table[["DFT", f"MLIP_b{b}"]].dropna().to_numpy().T)[0] for b in bs]
    ax.plot(bs, rho, color="#1baf7a", lw=2, marker="^", ms=8, mec=SURFACE, label="MLIP-CSP")
    for (name, r), ls in zip(EARLIER.items(), (":", "--", "-.")):
        ax.axhline(r, color=INK2, lw=1, ls=ls)
        ax.text(bs[-1], r + 0.015, name, ha="right", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks(bs, [str(b) for b in bs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.05, 1)
    ax.set_xlabel("CSP trials per composition", color=INK2)
    ax.set_ylabel("Spearman ρ with DFT E_hull", color=INK2)
    ax.set_title("B. Does more CSP give a better fidelity?", fontsize=11, color=INK, loc="left")
    style(ax)


def panel_recovery(ax, model):
    path = ROOT / "play" / "out_mlip" / model / "hull_recovery.csv"
    if not path.exists():
        ax.text(0.5, 0.5, "retrospective BO not run yet", ha="center", transform=ax.transAxes, color=INK2)
        ax.set_axis_off()
        return
    df = pd.read_csv(path)
    for m, (color, ls, marker, name) in METHODS.items():
        d = df[df.method == m]
        if d.empty:
            continue
        ax.errorbar(d["CSP budget"], d["DFT lookups"], yerr=d["±95%"], color=color, ls=ls, lw=2, marker=marker,
                    ms=8, mec=SURFACE, mew=1.5, capsize=3, label=name)
    ax.set_xscale("log")
    bs = sorted(df["CSP budget"].unique())
    ax.set_xticks(bs, [str(b) for b in bs])
    ax.xaxis.set_minor_formatter(NullFormatter())
    # headroom above the curves for the legend and the first-hit box, so neither hides data
    ax.set_ylim(0, (df["DFT lookups"] + df["±95%"]).max() * 1.55)
    ax.set_xlabel("CSP trials per composition", color=INK2)
    ax.set_ylabel("DFT lookups to find all stable phases", color=INK2)
    ax.set_title("C. DFT lookups needed to recover the hull (lower is better)", fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    # DFT lookups to the first stable phase, at the largest budget each method has
    first = df.sort_values("CSP budget").groupby("method").last()["lookups to 1st"]
    lines = ["DFT lookups to the FIRST stable phase:"]
    lines += [f"  {METHODS[m][3]}: {first[m]:.0f}" for m in METHODS if m in first.index]
    ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=INK, bbox={"facecolor": SURFACE, "edgecolor": GRID})
    style(ax)


def panel_models(ax):
    rows = []
    for m in MODELS:
        path = model_dir(m) / "refs_mlip_vs_mp.csv"
        if not path.exists():
            continue
        r = pd.read_csv(path)
        compounds = r[r.formula.str.count("[A-Z]") > 1]
        rows.append((m, compounds.error_meV.abs().mean()))
    ax.set_axis_off()
    speed = {"mace-medium": "1×", "chgnet": "~4×", "fastchgnet-0.3.0": "~6×", "fastchgnet": "~35×"}
    lines = ["MLIPs tried (Li-Sn-S-Cl references from Materials Project)", "",
             f"{'model':<18}{'error vs MP':>13}{'speed':>9}"]
    lines += [f"{m:<18}{e:>9.0f} meV{speed.get(m, ''):>9}" for m, e in rows]
    lines += ["", "error: formation-energy MAE against each model's", "training target (MP GGA/GGA+U).",
              "speed: CSP trials per hour vs MACE-MP-0, on CPU.", "",
              "FastCHGNet's own 'fast' model is quick but its", "energies are far off; the run uses FastCHGNet",
              "with the CHGNet 0.3.0 weights."]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5, color=INK)
    ax.set_title("D. Models", fontsize=11, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    args = ap.parse_args()
    table = pd.read_csv(model_dir(args.model) / "LiSnSCl_mlipcsp.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor(SURFACE)
    panel_scatter(axes[0, 0], table)
    panel_correlation(axes[0, 1], table)
    panel_recovery(axes[1, 0], args.model)
    panel_models(axes[1, 1])
    n = int(table[f"MLIP_b{budgets(table)[-1]}"].notna().sum())
    fig.suptitle(f"Li-Sn-S-Cl with a real cheap fidelity: {args.model} CSP on {n} compositions",
                 x=0.04, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = model_dir(args.model) / "summary.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
