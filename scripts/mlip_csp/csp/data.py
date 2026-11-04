"""Step 1: the compositions to run CSP on, with DFT energies above the hull recomputed from total energies.

Writes csp/out/compositions.csv (one row per reduced quaternary formula) and csp/out/refs_dft.csv (the references).
Run: python -m csp.data
"""
import pandas as pd
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from csp.common import DATA, OUT, REFERENCE_INDEX


def dft_table():
    """All rows of the DFT data: name (stripped), total energy, E_hull recomputed here and as stored (meV/atom)."""
    df = pd.read_csv(DATA)
    df.columns = ["name", "energy", "ehull_stored"]
    df["name"] = df.name.str.strip()  # names start with spaces
    entries = [ComputedEntry(n, e) for n, e in zip(df.name, df.energy)]
    pd_ = PhaseDiagram(entries)
    # round like the stored column; -4e-13 would otherwise not count as stable
    df["ehull"] = [max(round(1000 * pd_.get_e_above_hull(e), 1), 0.0) for e in entries]
    df["reduced"] = [e.composition.reduced_formula for e in entries]
    df["n_atoms"] = [int(e.composition.num_atoms) for e in entries]
    df["reference"] = df.index >= REFERENCE_INDEX
    return df


def compositions(df):
    """One row per reduced quaternary formula: lowest DFT E_hull over its cells, and the smallest cell for CSP."""
    q = df[~df.reference & (df.name.map(lambda n: len(Composition(n).elements)) == 4)]
    rows = []
    for red, g in q.groupby("reduced"):
        smallest = g.loc[g.n_atoms.idxmin()]
        rows.append({"formula": red, "csp_cell": Composition(smallest["name"]).formula.replace(" ", ""),
                     "n_atoms": int(smallest.n_atoms), "dft_ehull": g.ehull.min(),
                     "dft_cells": ";".join(sorted(g.name))})
    return pd.DataFrame(rows).sort_values("n_atoms").reset_index(drop=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = dft_table()
    diff = (df.ehull - df.ehull_stored).abs().max()
    # up to ~0.3: total energies are stored to 1 meV, so the 1-atom S1 cell lands 0.25 meV/atom below S8/S32 and
    # replaces them as the S reference, which the stored column did not do
    print(f"recomputed vs stored E_hull: max |diff| = {diff:.2f} meV/atom over {len(df)} rows")
    comp = compositions(df)
    comp.to_csv(OUT / "compositions.csv", index=False)
    refs = df[df.reference][["name", "reduced", "n_atoms", "energy", "ehull"]]
    refs.to_csv(OUT / "refs_dft.csv", index=False)
    stable = comp[comp.dft_ehull == 0]
    print(f"{len(comp)} quaternary compositions ({comp.n_atoms.min()}-{comp.n_atoms.max()} atoms in the CSP cell), "
          f"{len(stable)} stable: {', '.join(stable.formula)}")
    print(f"{len(refs)} reference rows, {(refs.ehull == 0).sum()} on the hull")
    print(f"wrote {OUT / 'compositions.csv'} and {OUT / 'refs_dft.csv'}")


if __name__ == "__main__":
    main()
