"""Step 6: pick 10-20 compositions for the DFT check and export their best MLIP-CSP structures.

Chosen: the DFT-stable quaternaries; the compositions where MLIP-CSP puts them furthest below their stored DFT E_hull
(where the stored DFT structure may not be the ground state); and a few random controls. Each best structure is
written as CIF and POSCAR to csp/out/dft_check/, with selection.csv.

The stored total energies can only be compared with new DFT under identical settings (code, functional,
pseudopotentials, the 700 eV cutoff). Run: python -m csp.select_dft --column MLIP_b50
"""
import argparse
import json

import numpy as np
import pandas as pd

from csp.common import DEFAULT_MODEL, MODELS, OUT, dict_to_atoms, model_dir, usable
from csp.run_csp import csp_dir


def best_structure(cell, model):
    rows = [json.loads(line) for line in (csp_dir(model) / f"{cell}.jsonl").read_text().splitlines() if line.strip()]
    ok = [r for r in rows if usable(r)]
    best = min(ok, key=lambda r: r["energy"])
    return dict_to_atoms(best["atoms"]), best


def select(table, column, n_below, n_random, seed):
    t = table.dropna(subset=[column]).copy()
    t["mlip_minus_dft"] = t[column] - t.DFT
    picks = [(f, "DFT-stable") for f in t[t.DFT == 0].Formula]
    rest = t[~t.Formula.isin([f for f, _ in picks])]
    picks += [(f, "MLIP below DFT") for f in rest.nsmallest(n_below, "mlip_minus_dft").Formula]
    rest = rest[~rest.Formula.isin([f for f, _ in picks])]
    rng = np.random.default_rng(seed)
    picks += [(f, "random control") for f in rng.choice(rest.Formula, size=min(n_random, len(rest)), replace=False)]
    reason = dict(picks)
    out = t[t.Formula.isin(reason)].copy()
    out["reason"] = out.Formula.map(reason)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--column", default="MLIP_b50", help="CSP budget column of LiSnSCl_mlipcsp.csv")
    ap.add_argument("--n-below", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    args = ap.parse_args()
    from ase.io import write

    table = pd.read_csv(model_dir(args.model) / "LiSnSCl_mlipcsp.csv")
    cells = pd.read_csv(OUT / "compositions.csv").set_index("formula").csp_cell
    chosen = select(table, args.column, args.n_below, args.n_random, args.seed)
    dest = model_dir(args.model) / "dft_check"
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in chosen.iterrows():
        cell = cells[r.Formula]
        atoms, best = best_structure(cell, args.model)
        write(dest / f"{cell}.cif", atoms)
        write(dest / f"{cell}.POSCAR", atoms, format="vasp", direct=True, sort=True)
        rows.append({"formula": r.Formula, "cell": cell, "reason": r.reason, "dft_ehull": r.DFT,
                     "mlip_ehull": r[args.column], "mlip_minus_dft": round(r.mlip_minus_dft, 1),
                     "mlip_eV_atom": round(best["energy"], 5), "csp_trial": best["trial"]})
    sel = pd.DataFrame(rows)
    sel.to_csv(dest / "selection.csv", index=False)
    print(sel.to_string(index=False))
    print(f"\nwrote {len(sel)} structures (CIF + POSCAR) and selection.csv to {dest}")


if __name__ == "__main__":
    main()
