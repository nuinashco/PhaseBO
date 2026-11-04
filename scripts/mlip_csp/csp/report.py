"""CSP run statistics and the full-run time estimate (used for the pilot).

Per cell: trials, share physical/converged, time per trial, best energy and the trial that found it. The time per
trial is fitted against cell size to extrapolate a full run over compositions.csv. If refs_mlip.csv exists, the best
CSP energy of each cell is compared with the MACE-relaxed Materials Project structure of the same formula.
Run: python -m csp.report --trials 50 --workers 15
"""
import argparse
import json

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from csp.common import DEFAULT_MODEL, MODELS, OUT, model_dir, usable
from csp.run_csp import csp_dir


def cell_stats(model):
    rows = []
    for path in sorted(csp_dir(model).glob("*.jsonl")):
        trials = sorted((json.loads(line) for line in path.read_text().splitlines() if line.strip()),
                        key=lambda r: r["trial"])
        ok = [r for r in trials if usable(r)]
        best = min(ok, key=lambda r: r["energy"]) if ok else None
        comp = Composition(path.stem)
        rows.append({"cell": path.stem, "formula": comp.reduced_formula, "n_atoms": int(comp.num_atoms),
                     "trials": len(trials), "usable": len(ok) / len(trials),
                     "converged": np.mean([r["converged"] for r in trials]),
                     "s_per_trial": np.mean([r["time"] for r in trials]),
                     "best_eV_atom": best["energy"] if best else np.nan,
                     "best_at_trial": best["trial"] + 1 if best else np.nan})
    return pd.DataFrame(rows)


def estimate(stats, trials, workers):
    """Full-run wall time from a power-law fit of time per trial against cell size."""
    comp = pd.read_csv(OUT / "compositions.csv")
    if len(stats) >= 2:
        k, logc = np.polyfit(np.log(stats.n_atoms), np.log(stats.s_per_trial), 1)
    else:
        k, logc = 2.0, np.log(stats.s_per_trial.iloc[0] / stats.n_atoms.iloc[0] ** 2)
    per_trial = np.exp(logc) * comp.n_atoms.to_numpy() ** k
    total = per_trial.sum() * trials
    return k, total / workers / 3600, total / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=50, help="planned trials per composition for the estimate")
    ap.add_argument("--workers", type=int, default=15)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    args = ap.parse_args()

    stats = cell_stats(args.model)
    if stats.empty:
        raise SystemExit("no CSP results in csp/out/csp")
    refs_path = model_dir(args.model) / "refs_mlip.csv"
    if refs_path.exists():
        refs = pd.read_csv(refs_path)
        refs = refs[refs.physical].groupby("formula").energy.min()
        stats["mp_mlip_eV_atom"] = stats.formula.map(refs)
        stats["csp_minus_mp_meV"] = 1000 * (stats.best_eV_atom - stats.mp_mlip_eV_atom)
    pd.set_option("display.width", 200)
    print(stats.round(3).to_string(index=False))
    k, wall_h, cpu_h = estimate(stats, args.trials, args.workers)
    print(f"\ntime per trial ~ n_atoms^{k:.2f}; full run, {args.trials} trials x 165 compositions: "
          f"~{cpu_h:.0f} core-hours, ~{wall_h:.1f} h on {args.workers} CPU workers")


if __name__ == "__main__":
    main()
