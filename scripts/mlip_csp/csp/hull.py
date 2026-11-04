"""Step 4: the MLIP-CSP fidelity. E_hull of every composition after b CSP trials, on a hull of MLIP energies only.

References: Materials Project structures relaxed with the MLIP (refs_mlip.csv), plus CSP for DFT-hull references MP
does not have (refs_missing.csv). Quaternaries: the best usable MLIP-CSP energy after b trials (best-so-far trace).
Every composition's E_hull uses all the others' MLIP results, by design: MLIP-CSP is run once for every composition.

Writes csp/out/<model>/LiSnSCl_mlipcsp.csv with columns Formula, DFT, MLIP_b<b> (meV/atom), ready for
play/mf_play.py --data csp/out/<model>/LiSnSCl_mlipcsp.csv --cheap-column MLIP_b<b>.
Run: python -m csp.hull [--model chgnet]
"""
import argparse
import json

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.stats import spearmanr

from csp.common import DEFAULT_MODEL, MODELS, OUT, model_dir, usable
from csp.run_csp import csp_dir

CHECKPOINTS = (1, 2, 5, 10, 20, 50, 100, 200)


def trace(cell, model):
    """Best-so-far energy per atom (eV) after each trial of a cell; failed, unconverged or unphysical trials count
    as tries but not as results."""
    path = csp_dir(model) / f"{cell}.jsonl"
    if not path.exists():
        return np.array([])
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:  # the line a running CSP job is writing right now
            continue
    rows.sort(key=lambda r: r["trial"])
    # trials can finish out of order; the trace covers only the unbroken run of trials 0, 1, 2, ...
    n = next((i for i, r in enumerate(rows) if r["trial"] != i), len(rows))
    e = np.array([r["energy"] if usable(r) else np.inf for r in rows[:n]])
    return np.minimum.accumulate(e)


def entry(formula, energy_per_atom):
    comp = Composition(formula)
    return ComputedEntry(comp, energy_per_atom * comp.num_atoms)


def reference_entries(model):
    path = model_dir(model) / "refs_mlip.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found: run python -m csp.references --model {model} first")
    refs = pd.read_csv(path)
    refs = refs[refs.physical].sort_values("energy").drop_duplicates("formula")
    entries = [entry(f, e) for f, e in zip(refs.formula, refs.energy)]
    missing = OUT / "refs_missing.csv"
    if missing.exists():
        for cell in pd.read_csv(missing).csp_cell:
            t = trace(cell, model)
            if len(t) and np.isfinite(t[-1]):
                entries.append(entry(cell, t[-1]))
            else:
                print(f"warning: no CSP result for the missing reference {cell}; run csp.run_csp --cells {cell}")
    return entries


def mlip_table(checkpoints, model):
    comp = pd.read_csv(OUT / "compositions.csv")
    traces = {c: trace(c, model) for c in comp.csp_cell}
    refs = reference_entries(model)
    table = pd.DataFrame({"Formula": comp.formula, "DFT": comp.dft_ehull})
    for b in checkpoints:
        have = [c for c in comp.csp_cell if len(traces[c]) >= b and np.isfinite(traces[c][b - 1])]
        if not have:
            continue
        quats = {c: entry(c, traces[c][b - 1]) for c in have}
        pd_ = PhaseDiagram(refs + list(quats.values()))
        ehull = {c: 1000 * pd_.get_e_above_hull(e) for c, e in quats.items()}
        table[f"MLIP_b{b}"] = [round(max(ehull[c], 0.0), 1) if c in ehull else np.nan for c in comp.csp_cell]
    return table


def report(table):
    stable = table[table.DFT == 0].Formula.tolist()
    print(f"DFT-stable quaternaries: {', '.join(stable)}")
    for col in [c for c in table.columns if c.startswith("MLIP_")]:
        t = table.dropna(subset=[col])
        rho = spearmanr(t.DFT, t[col])[0] if len(t) > 2 else np.nan
        ranks = t[col].rank(method="min")
        rank_of = {f: int(ranks[t.Formula == f].iloc[0]) for f in stable if f in set(t.Formula)}
        print(f"  {col:>9}: {len(t):3d} compositions, Spearman vs DFT {rho:+.2f}, "
              f"{int((t[col] == 0).sum())} on the MLIP hull, ranks of the DFT-stable: {rank_of}")
        if len(t) < 10:
            print("    " + ", ".join(f"{f}: MLIP {m:.1f} / DFT {d:.1f}" for f, m, d in zip(t.Formula, t[col], t.DFT)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=list(CHECKPOINTS))
    args = ap.parse_args()
    table = mlip_table(args.checkpoints, args.model)
    path = model_dir(args.model) / "LiSnSCl_mlipcsp.csv"
    table.to_csv(path, index=False)
    report(table)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
