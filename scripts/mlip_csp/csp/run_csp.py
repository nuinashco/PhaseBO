"""Step 3: MLIP-CSP. Random symmetric structures (pyxtal) relaxed with an MLIP, a fixed number of trials per cell.

Each trial is one line of csp/out/<model>/csp/<cell>.jsonl: trial index, space group, energy per atom (eV), converged,
physical, steps, wall time and the relaxed structure. Runs resume: trials already in the file are skipped, and every
(cell, trial) has its own seed, so a rerun gives the same structures.

Run, e.g.:
  python -m csp.run_csp --formulas Li5SnS4Cl Li3SnS3Cl --trials 20            # by reduced formula
  python -m csp.run_csp --cells Li8Sn2S8 --trials 20                          # any cell, e.g. a known ternary
  python -m csp.run_csp --all --trials 50 --device cuda --workers 3           # every composition, on a GPU
  python -m csp.run_csp --formulas Li3SnS3Cl --model chgnet                    # another MLIP, own output folder
"""
import argparse
import json
import multiprocessing as mp
import time
import zlib

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from csp.common import DEFAULT_MODEL, MODELS, OUT, atoms_to_dict, model_dir, pool, relax, worker_calc


def csp_dir(model):
    return model_dir(model) / "csp"


def random_structure(cell, rng, attempts=60):
    """A random structure of the cell's composition in a random compatible space group (P1 as the last resort)."""
    from pyxtal import pyxtal

    comp = Composition(cell)
    species = [str(el) for el in comp.elements]
    counts = [int(comp[el]) for el in comp.elements]
    for attempt in range(attempts):
        group = int(rng.integers(2, 231)) if attempt < attempts - 1 else 1
        crystal = pyxtal()
        try:
            crystal.from_random(3, group, species, counts, random_state=int(rng.integers(2**31)))
        except Exception:  # composition incompatible with the group's Wyckoff positions
            continue
        if crystal.valid:
            return crystal.to_ase(), group
    raise RuntimeError(f"no random structure for {cell}")


def seed_of(cell, trial, seed):
    return [seed, zlib.crc32(cell.encode()), trial]


def run_trial(task):
    cell, trial, seed = task
    rng = np.random.default_rng(seed_of(cell, trial, seed))
    t0 = time.perf_counter()
    try:
        atoms, group = random_structure(cell, rng)
        energy, converged, steps, ok = relax(atoms, worker_calc())
        result = {"energy": energy, "converged": converged, "physical": ok, "steps": steps, "sg_start": group,
                  "atoms": atoms_to_dict(atoms)}
    except Exception as ex:  # a failed trial still counts against the budget
        result = {"energy": None, "converged": False, "physical": False, "steps": 0, "error": repr(ex)}
    return cell, {"trial": trial, "time": round(time.perf_counter() - t0, 2), **result}


def done_trials(cell, model):
    path = csp_dir(model) / f"{cell}.jsonl"
    if not path.exists():
        return set()
    return {json.loads(line)["trial"] for line in path.read_text().splitlines() if line.strip()}


def targets(args):
    cells = list(args.cells or [])
    if args.formulas or args.all:
        comp = pd.read_csv(OUT / "compositions.csv")
        if not args.all:
            missing = set(args.formulas) - set(comp.formula)
            if missing:
                raise SystemExit(f"not in compositions.csv: {sorted(missing)}")
            comp = comp[comp.formula.isin(args.formulas)]
        cells += list(comp.csp_cell)
        refs = OUT / "refs_missing.csv"  # CSV-stable references that Materials Project does not have
        if args.all and refs.exists():
            cells += list(pd.read_csv(refs).csp_cell)
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formulas", nargs="*", help="reduced formulas from compositions.csv")
    ap.add_argument("--cells", nargs="*", help="cell formulas, e.g. Li8Sn2S8")
    ap.add_argument("--all", action="store_true", help="every composition (and the missing references)")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(mp.cpu_count() - 1, 1))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=MODELS)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cells = targets(args)
    if not cells:
        raise SystemExit("nothing to run: give --formulas, --cells or --all")
    out = csp_dir(args.model)
    out.mkdir(parents=True, exist_ok=True)
    # breadth-first: trial 0 of every cell, then trial 1, ..., so a stopped run still has every composition at the
    # same budget; within a round, largest cells first so the long relaxations do not end up at the tail
    cells.sort(key=lambda c: -Composition(c).num_atoms)
    done = {c: done_trials(c, args.model) for c in cells}
    tasks = [(c, t, args.seed) for t in range(args.trials) for c in cells if t not in done[c]]
    print(f"{args.model}: {len(cells)} cells, {len(tasks)} trials to run on {args.workers} {args.device} worker(s)",
          flush=True)

    files = {c: open(out / f"{c}.jsonl", "a") for c in cells}
    remaining = {c: sum(t[0] == c for t in tasks) for c in cells}
    round_left = {r: sum(t[1] == r for t in tasks) for r in range(args.trials)}
    best = {}
    t0 = time.perf_counter()
    try:
        with pool(args.workers, args.model, args.device, args.dtype) as workers:
            for i, (cell, row) in enumerate(workers.imap_unordered(run_trial, tasks), 1):
                files[cell].write(json.dumps(row) + "\n")
                files[cell].flush()
                if row["energy"] is not None and row["physical"]:
                    best[cell] = min(best.get(cell, np.inf), row["energy"])
                remaining[cell] -= 1
                round_left[row["trial"]] -= 1
                if round_left[row["trial"]] == 0:
                    print(f"[{i}/{len(tasks)} {time.perf_counter() - t0:.0f}s] round {row['trial'] + 1} complete",
                          flush=True)
                if remaining[cell] == 0:
                    print(f"[{i}/{len(tasks)} {time.perf_counter() - t0:.0f}s] {cell}: best {best.get(cell, np.nan):.4f}"
                          f" eV/atom", flush=True)
    finally:
        for f in files.values():
            f.close()


if __name__ == "__main__":
    main()
