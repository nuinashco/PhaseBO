"""Export a finished MLIP-CSP run as raw data of the PhaseBO repository: data/raw/LiSnSCl_<mlip>_csp/.

scripts/process_LiSnSCl_<mlip>_csp.py turns these files into the processed table; nothing else reads out/.

trials.jsonl              every CSP trial, cells in compositions.csv order then the missing references, trials in
                          order: the cell, the trial, energy (eV/atom, null when it failed), converged, steps, physical,
                          usable (the hull's filter, csp.common.usable), wall time (s), start space group, structure
compositions.csv          each composition's CSP cell (its smallest DFT cell), atoms and DFT E_hull (meV/atom)
missing_references.csv    DFT-hull references Materials Project lacks, searched by CSP like the compositions
references.csv            Materials Project references relaxed with the MLIP (energy in eV/atom), physical flag
references_structures.jsonl   their relaxed structures; mp_references.jsonl: the structures as downloaded
run.json                  the MLIP, the relaxation and filter settings, the seed, the trials and package versions
Run: python -m csp.export --model fastchgnet-0.3.0 [--seed 0]
"""
import argparse
import importlib.metadata
import json

import pandas as pd

from csp import common
from csp.common import DEFAULT_MODEL, MODELS, OUT, REPO, model_dir, usable
from csp.run_csp import csp_dir

LABELS = {"fastchgnet-0.3.0": "fastchgnet"}  # the label in the processed table's fidelities


def trials(cells, model):
    rows = []
    for cell in cells:
        path = csp_dir(model) / f"{cell}.jsonl"
        found = sorted((json.loads(line) for line in path.read_text().splitlines() if line.strip()),
                       key=lambda r: r["trial"])
        if [r["trial"] for r in found] != list(range(len(found))):
            raise SystemExit(f"{path}: trials are not 0..{len(found) - 1}; let the CSP run finish first")
        for r in found:
            rows.append({"cell": cell, "trial": r["trial"], "energy": r["energy"], "converged": r["converged"],
                         "steps": r["steps"], "physical": r["physical"], "usable": bool(usable(r)), "time": r["time"],
                         "sg_start": r.get("sg_start"), "error": r.get("error"), "atoms": r.get("atoms")})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--seed", type=int, default=0, help="the --seed the CSP run used")
    args = ap.parse_args()
    label = LABELS.get(args.model, args.model)
    out = REPO / "data" / "raw" / f"LiSnSCl_{label}_csp"
    out.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(OUT / "compositions.csv")
    missing = pd.read_csv(OUT / "refs_missing.csv")
    rows = trials([*comp.csp_cell, *missing.csp_cell], args.model)
    counts = pd.Series([r["cell"] for r in rows]).value_counts()
    if counts.nunique() != 1:
        raise SystemExit(f"cells have different trial counts ({sorted(counts.unique())}); finish the run first")
    with open(out / "trials.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _copy(OUT / "compositions.csv", out / "compositions.csv")
    _copy(OUT / "refs_missing.csv", out / "missing_references.csv")
    _copy(model_dir(args.model) / "refs_mlip.csv", out / "references.csv")
    _copy(model_dir(args.model) / "refs_structures.jsonl", out / "references_structures.jsonl")
    _copy(OUT / "mp_references.jsonl", out / "mp_references.jsonl")
    packages = ("ase", "fastchgnet", "chgnet", "pyxtal", "pymatgen", "torch", "numpy")
    run = {"model": args.model, "label": label, "seed": args.seed, "trials": int(counts.iloc[0]),
           "fmax_ev_per_angstrom": common.FMAX, "max_steps": common.MAX_STEPS,
           "min_bond_ratio": common.MIN_BOND_RATIO, "collapse_ratio": common.COLLAPSE_RATIO,
           "max_volume_per_atom": common.MAX_VOLUME_PER_ATOM,
           "versions": {p: _version(p) for p in packages}}
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(rows)} trials of {len(counts)} cells and the references to {out}")


def _copy(src, dst):
    """Copy a text file with LF line ends, whatever the platform wrote it with."""
    dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
