"""Step 2: reference phases from Materials Project, relaxed with the MLIP (the same relaxer as the CSP trials).

Downloads every MP entry of the Li-Sn-S-Cl elements, binaries and ternaries within --max-ehull of MP's own hull
(cached in csp/out/mp_references.jsonl, shared by all models), relaxes it, and writes refs_mlip.csv and
refs_structures.jsonl to csp/out/<model>/. Also lists the references that the DFT data puts on its hull but MP does
not have (csp/out/refs_missing.csv); csp.run_csp --all runs CSP for them.

Needs MP_API_KEY in the environment or in .env.
Run: python -m csp.references [--model chgnet];  python -m csp.references --check [--model ...]
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os

import pandas as pd
from pymatgen.core import Composition

from csp.common import (DEFAULT_MODEL, ELEMENTS, MIN_BOND_RATIO, MODELS, OUT, atoms_to_dict, bond_ratio,
                        dict_to_atoms, load_env, model_dir, pool, relax, worker_calc)

MP_CACHE = OUT / "mp_references.jsonl"


def chemical_systems():
    """Elements, binaries and ternaries; quaternaries are the search space, not references."""
    return ["-".join(sorted(c)) for n in (1, 2, 3) for c in itertools.combinations(ELEMENTS, n)]


def fetch(max_ehull):
    """(material_id, formula, MP E_hull, ASE atoms) of every reference, from the cache if it exists."""
    if MP_CACHE.exists():
        rows = [json.loads(line) for line in MP_CACHE.read_text().splitlines() if line.strip()]
        return [(r["material_id"], r["formula"], r["mp_ehull"], dict_to_atoms(r["atoms"])) for r in rows
                if r["mp_ehull"] <= max_ehull]
    load_env()
    if not os.environ.get("MP_API_KEY"):
        raise SystemExit("MP_API_KEY is not set: put it in .env as MP_API_KEY=...")
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    with MPRester() as mpr:
        docs = mpr.materials.summary.search(chemsys=chemical_systems(), energy_above_hull=(0, max_ehull),
                                            fields=["material_id", "formula_pretty", "structure",
                                                    "energy_above_hull"])
    refs = [(str(d.material_id), d.formula_pretty, d.energy_above_hull, AseAtomsAdaptor.get_atoms(d.structure))
            for d in docs]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(MP_CACHE, "w") as f:
        for material_id, formula, ehull, atoms in refs:
            f.write(json.dumps({"material_id": material_id, "formula": formula, "mp_ehull": ehull,
                                "atoms": atoms_to_dict(atoms)}) + "\n")
    return refs


def relax_reference(task):
    material_id, formula, mp_ehull, atoms = task
    energy, converged, steps, _ = relax(atoms, worker_calc())
    row = {"material_id": material_id, "formula": Composition(formula).reduced_formula, "n_atoms": len(atoms),
           "energy": energy, "mp_ehull": mp_ehull, "converged": converged, "physical": reference_ok(atoms),
           "volume_per_atom": round(atoms.get_volume() / len(atoms), 1), "steps": steps}
    return row, atoms_to_dict(atoms)


def reference_ok(atoms):
    """Bonds only: unlike a CSP cell, an MP structure with a large volume per atom is real (molecular S8 crystals,
    LiS4, SnS slabs reach 90-730 Å^3/atom); the volume check exists to catch CSP cells that blew apart."""
    return bond_ratio(atoms) >= MIN_BOND_RATIO


def missing_references(refs):
    """References on the DFT hull whose reduced formula Materials Project does not have, as CSP targets."""
    dft = pd.read_csv(OUT / "refs_dft.csv")
    stable = dft[dft.ehull == 0].sort_values("n_atoms").drop_duplicates("reduced")
    missing = stable[~stable.reduced.isin(set(refs.formula))]
    return pd.DataFrame({"formula": missing.reduced, "csp_cell": [Composition(n).formula.replace(" ", "")
                                                                  for n in missing.name],
                         "n_atoms": missing.n_atoms})


def formation_energies(energy, formula):
    """Formation energy per atom of each entry, relative to the lowest-energy entry of each element."""
    mu = {f: e for f, e in zip(formula, energy) if len(Composition(f).elements) == 1}
    mu = {el: min(e for f, e in mu.items() if Composition(f).elements[0].symbol == el) for el in ELEMENTS}
    return [e - sum(mu[el.symbol] * Composition(f).get_atomic_fraction(el) for el in Composition(f).elements)
            for e, f in zip(energy, formula)]


def check(model):
    """MLIP vs Materials Project on the references, as formation energies against each side's own elements.
    Each model is compared with its training target: MACE-MP-0 learned MP's uncorrected GGA/GGA+U energies, CHGNet
    the MP2020-corrected ones. The summary endpoint mixes GGA and r2SCAN, whose absolute energies differ, so the
    comparison takes the GGA_GGA+U thermo documents only."""
    load_env()
    from mp_api.client import MPRester

    corrected = "chgnet" in model  # CHGNet and FastCHGNet learned MP2020-corrected energies
    refs = pd.read_csv(model_dir(model) / "refs_mlip.csv")
    with MPRester() as mpr:
        docs = mpr.materials.thermo.search(material_ids=list(refs.material_id), thermo_types=["GGA_GGA+U"],
                                           fields=["material_id", "uncorrected_energy_per_atom", "energy_per_atom"])
    mp_e = {str(d.material_id): d.energy_per_atom if corrected else d.uncorrected_energy_per_atom for d in docs}
    refs["mp_energy"] = refs.material_id.map(mp_e)
    refs = refs.dropna(subset=["mp_energy"])
    refs["mlip_ef"] = formation_energies(refs.energy, refs.formula)
    refs["mp_ef"] = formation_energies(refs.mp_energy, refs.formula)
    refs["error_meV"] = (1000 * (refs.mlip_ef - refs.mp_ef)).round(1)
    compounds = refs[refs.formula.map(lambda f: len(Composition(f).elements)) > 1]
    lowest = compounds.sort_values("mp_energy").drop_duplicates("formula").sort_values("formula")
    print(lowest[["material_id", "formula", "mlip_ef", "mp_ef", "error_meV"]].round(3).to_string(index=False))
    target = "MP2020-corrected" if corrected else "uncorrected"
    print(f"\nformation energy, {model} vs MP (GGA/GGA+U, {target}), {len(compounds)} compound entries: "
          f"MAE {compounds.error_meV.abs().mean():.0f} meV/atom, mean signed {compounds.error_meV.mean():+.0f}")
    refs.to_csv(model_dir(model) / "refs_mlip_vs_mp.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only compare refs_mlip.csv with MP's energies")
    ap.add_argument("--max-ehull", type=float, default=0.1, help="eV/atom above MP's hull")
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()
    if args.check:
        return check(args.model)

    tasks = fetch(args.max_ehull)
    print(f"{len(tasks)} Materials Project entries in {len(chemical_systems())} chemical systems", flush=True)
    rows, structures = [], []
    with pool(args.workers, args.model, args.device, args.dtype) as workers:
        for row, atoms in workers.imap_unordered(relax_reference, tasks):
            rows.append(row)
            structures.append({"material_id": row["material_id"], "atoms": atoms})
            print(f"  {row['material_id']:>12} {row['formula']:>10} {row['energy']:.4f} eV/atom"
                  f"{'' if row['converged'] else ' (not converged)'}{'' if row['physical'] else ' (unphysical)'}",
                  flush=True)
    refs = pd.DataFrame(rows).sort_values(["formula", "energy"])
    out = model_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    refs.to_csv(out / "refs_mlip.csv", index=False)
    with open(out / "refs_structures.jsonl", "w") as f:
        for s in structures:
            f.write(json.dumps(s) + "\n")
    missing = missing_references(refs)
    missing.to_csv(OUT / "refs_missing.csv", index=False)
    print(f"wrote {out / 'refs_mlip.csv'} ({len(refs)} entries, {refs.formula.nunique()} formulas); "
          f"{len(missing)} DFT-hull references not in MP: {', '.join(missing.formula) or 'none'}")


if __name__ == "__main__":
    main()
