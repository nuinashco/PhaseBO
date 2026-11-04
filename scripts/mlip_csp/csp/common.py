"""Shared pieces of the MLIP-CSP pipeline: paths, the .env key, the MLIPs, and the relaxer used for references and
CSP alike."""
import os
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # scripts/mlip_csp
REPO = ROOT.parent.parent  # the PhaseBO repository
DATA = REPO / "data" / "raw" / "LiSnSCl_700eV.csv"  # compositions with DFT total energies (eV); references from row 195
REFERENCE_INDEX = 195
OUT = ROOT / "out"  # gitignored; model-independent tables (compositions.csv, refs_dft.csv) and one folder per MLIP
ELEMENTS = ("Li", "Sn", "S", "Cl")

# MLIPs, all trained on MPtrj (Materials Project GGA/GGA+U relaxations). FastCHGNet
# (github.com/HPC-AI-Team/FastCHGNet): "fastchgnet" is its own "fast" model, whose forces come from a direct head
# rather than the energy gradient; "fastchgnet-0.3.0" runs the CHGNet 0.3.0 weights on its faster code
MODELS = ("mace-small", "mace-medium", "mace-large", "chgnet", "fastchgnet", "fastchgnet-0.3.0")
DEFAULT_MODEL = "mace-medium"


def model_dir(model):
    """Outputs of one MLIP: refs_mlip.csv, refs_structures.jsonl, csp/<cell>.jsonl, LiSnSCl_mlipcsp.csv, ..."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    return OUT / model

# relaxation settings, the same for references and CSP trials so that their energies share one hull
FMAX = 0.05  # eV/Å
MAX_STEPS = 400
# distance / (sum of covalent radii): 0.97-1.07 for the pilot's good structures; MACE-MP has unphysical deep minima
# at short range, e.g. Li-S at 1.28 Å (ratio 0.55, -20 eV/atom) or Li-Li at 2.01 Å (0.78, -8 eV/atom) instead of ~-4
MIN_BOND_RATIO = 0.85
COLLAPSE_RATIO = 0.5  # stop a relaxation early once it falls this far
CHECK_EVERY = 10  # steps
MAX_VOLUME_PER_ATOM = 80.0  # Å^3; larger means the cell blew apart


def load_env():
    """Set variables from the repository's .env (KEY=value lines) that are not already in the environment."""
    path = REPO / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#") and value.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip()


def calculator(model=DEFAULT_MODEL, device="cpu", dtype="float32"):
    """The MLIP as an ASE calculator: MACE-MP-0 of a given size, or CHGNet (dtype applies to MACE only)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model.startswith("mace-"):
            from mace.calculators import mace_mp
            return mace_mp(model=model[len("mace-"):], device=device, default_dtype=dtype)
        if model == "chgnet":
            from chgnet.model.dynamics import CHGNetCalculator
            return CHGNetCalculator(use_device=device)
        if model.startswith("fastchgnet"):
            from fastchgnet.model.dynamics import FastCHGNetCalculator
            weights = model.partition("-")[2] or "fast"
            return FastCHGNetCalculator(model=weights, use_device=device, on_isolated_atoms="ignore")
    raise ValueError(f"unknown model {model!r}; choose from {MODELS}")


_WORKER_CALC = None


def init_worker(model, device, dtype):
    """Pool initializer: one single-threaded calculator per worker process."""
    import torch
    torch.set_num_threads(1)
    global _WORKER_CALC
    _WORKER_CALC = calculator(model, device, dtype)


def worker_calc():
    return _WORKER_CALC


def pool(workers, model, device, dtype):
    """A process pool of MLIP workers; the model is loaded (downloaded) first so that workers do not race to write it."""
    import multiprocessing as mp
    calculator(model, "cpu", dtype)
    return mp.get_context("spawn").Pool(workers, init_worker, (model, device, dtype))


class Collapsed(Exception):
    pass


def relax(atoms, calc, fmax=FMAX, steps=MAX_STEPS):
    """Relax positions and cell in place. Returns energy per atom (eV), converged, steps taken, and whether the
    result is physical (no collapsed atoms, no exploded cell). A relaxation heading into a collapse is stopped early
    and reported as unphysical."""
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms.calc = calc
    opt = FIRE(FrechetCellFilter(atoms), logfile=None)

    def watchdog():
        if bond_ratio(atoms) < COLLAPSE_RATIO:
            raise Collapsed

    opt.attach(watchdog, interval=CHECK_EVERY)
    try:
        converged = bool(opt.run(fmax=fmax, steps=steps))
    except Collapsed:
        return float(atoms.get_potential_energy()) / len(atoms), False, opt.nsteps, False
    energy = float(atoms.get_potential_energy()) / len(atoms)
    return energy, converged, opt.nsteps, bool(physical(atoms))


def bond_ratio(atoms):
    """Smallest interatomic distance relative to the sum of the two covalent radii."""
    from ase.data import covalent_radii

    if len(atoms) < 2:
        return np.inf
    d = atoms.get_all_distances(mic=True)
    r = covalent_radii[atoms.numbers]
    ratio = d / (r[:, None] + r[None, :])
    return float(ratio[~np.eye(len(atoms), dtype=bool)].min())


def physical(atoms):
    return bool(atoms.get_volume() / len(atoms) <= MAX_VOLUME_PER_ATOM and bond_ratio(atoms) >= MIN_BOND_RATIO)


def usable(row):
    """Whether a stored trial counts: physical under the current thresholds (rechecked from the saved structure,
    since the flag in older files used looser ones), and either converged or stopped by the step limit. A relaxation
    that ran out of steps is a real structure with a slightly high energy; one stopped by the collapse watchdog
    (fewer steps than the limit, not converged) is not. The pilot's -8 eV/atom MACE artefact fails the bond check."""
    if row.get("energy") is None or "atoms" not in row:
        return False
    finished = row.get("converged", False) or row.get("steps", 0) >= MAX_STEPS
    return finished and physical(dict_to_atoms(row["atoms"]))


def atoms_to_dict(atoms):
    return {"numbers": atoms.numbers.tolist(), "cell": atoms.cell.array.round(6).tolist(),
            "positions": atoms.positions.round(6).tolist()}


def dict_to_atoms(d):
    from ase import Atoms
    return Atoms(numbers=d["numbers"], cell=d["cell"], positions=d["positions"], pbc=True)
