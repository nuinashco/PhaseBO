"""Write fake Li-Sn-S-Cl phase fields with three frameworks, calibrated to the real one.

Run it again only when the real table or the recipe below changes:

    uv run scripts/make_fake_fields.py

For each size and seed it writes data/processed/fake_LiSnSCl_<size>_<seed>.csv with its
sidecar, and configs/system/fake_LiSnSCl_<size>_<seed>.yaml naming it.

The recipe, in formation energy per atom: every charge-balanced composition with at most
<size> atoms per cell gets the real field's trend (an isotropic Matern 5/2 GP fitted to
the real formation energies over all four atomic fractions) plus an independent
roughness the size of that fit's noise. Compositions of fewer than four elements are the
references, one structure each. Every four-element composition sits in each framework,
which adds its own smooth offset (a Matern 5/2 draw over the fractions).
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import rootutils
from hydra import compose, initialize_config_dir
from pymatgen.core import Composition, Element

from nuikit.errors import cli_boundary
from nuikit.seeding import seed_everything

from phasebo import paths
from phasebo.config import DataConfig
from phasebo.data import store, table
from phasebo.field import candidates

rootutils.setup_root(__file__, indicator=".project-root")

REAL = paths.ROOT / "data/processed/LiSnSCl.csv"
SIZES = (12, 16, 20, 24, 32)  # Most atoms per cell: about 20 to 500 compositions.
SEEDS = (0, 1, 2)
FRAMEWORKS = ("cubic", "hexagonal", "wurtzite")
SPREAD_MEV = 50.0  # Std of each framework's offset, per atom.
# Lengthscale of the offsets, in atomic fractions: at 0.15 the top framework wins
# 40-65% of compositions, in regions; at 0.3 one framework wins nearly everywhere.
SCALE = 0.15
# Keeps the Cholesky factor of a Matern covariance finite; far below any energy here.
JITTER = 1e-8


@cli_boundary
def main() -> None:
    # Torch drives the trend fit's starting values and any retries.
    seed_everything(0)
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        system = DataConfig.from_hydra(compose("data", ["system=LiSnSCl"])).system
    charges = {symbol: element.charge for symbol, element in system.elements.items()}
    elements = list(charges)
    trend, rough_mev = fit_trend(store.load(REAL).rows, elements)
    for size in SIZES:
        formulas = list_compositions(charges, size)
        fractions = to_fractions(formulas, elements)
        base_mev = trend(fractions)
        for seed in SEEDS:
            rng = np.random.default_rng([size, seed])
            rows = make_rows(
                formulas, fractions, base_mev=base_mev, rough_mev=rough_mev, rng=rng
            )
            cleaned, _ = table.clean(rows, elements=elements, source=str(REAL))
            name = f"fake_LiSnSCl_{size}_{seed}"
            out = paths.ROOT / "data/processed" / f"{name}.csv"
            store.write(out, cleaned, raw=[REAL], script=Path(__file__))
            write_config(name, size=size, seed=seed)
            pairs = int(np.count_nonzero(~cleaned["reference"].to_numpy()))
            print(f"wrote {name}: {pairs // len(FRAMEWORKS)} compositions")


def fit_trend(
    rows: pd.DataFrame, elements: list[str]
) -> tuple[Callable[[np.ndarray], np.ndarray], float]:
    """Return the real field's formation-energy trend (meV/atom) and its noise."""
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    cells = [Composition(f) for f in rows["formula"]]
    per_atom_ev = [
        e / c.num_atoms for c, e in zip(cells, rows["total_energy_ev"], strict=True)
    ]
    pure_ev: dict[str, float] = {}
    for cell, energy in zip(cells, per_atom_ev, strict=True):
        if len(cell.elements) == 1:
            symbol = str(cell.elements[0])
            pure_ev[symbol] = min(pure_ev.get(symbol, np.inf), energy)
    lowest_mev: dict[str, float] = {}
    for point, cell, energy in zip(rows["point"], cells, per_atom_ev, strict=True):
        mixed = sum(
            cell.get_atomic_fraction(el) * pure_ev[str(el)] for el in cell.elements
        )
        lowest_mev[point] = min(lowest_mev.get(point, np.inf), 1000 * (energy - mixed))
    x = torch.as_tensor(to_fractions(list(lowest_mev), elements))
    y = torch.tensor(list(lowest_mev.values()), dtype=torch.float64)[:, None]
    model = SingleTaskGP(x, y, covar_module=ScaleKernel(MaternKernel(nu=2.5)))
    with warnings.catch_warnings():
        # `linear_operator`, a gpytorch dependency, warns on import under 3.14.
        warnings.filterwarnings("ignore", category=FutureWarning)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    model.eval()
    scale = cast("Standardize", model.outcome_transform).stdvs
    with torch.no_grad():
        likelihood: Any = model.likelihood
        noise_mev = float(likelihood.noise.sqrt() * scale)

    def trend(fractions: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            mean_mev = model.posterior(torch.as_tensor(fractions)).mean.numpy().ravel()
        # Pure elements are the zero of formation energy.
        return np.where((fractions > 0).sum(axis=1) == 1, 0.0, mean_mev)

    return trend, noise_mev


def list_compositions(charges: dict[str, int], size: int) -> list[str]:
    """Return every charge-balanced reduced composition up to `size` atoms per cell."""
    ranges = dict.fromkeys(charges, (0, size))
    cells = candidates.generate(charges, ranges, max_atoms=size)
    reduced = {Composition(cell).reduced_composition.formula for cell in cells}
    return [*charges, *sorted(f for f in reduced if len(Composition(f).elements) > 1)]


def to_fractions(formulas: list[str], elements: list[str]) -> np.ndarray:
    return np.array(
        [
            [Composition(f).get_atomic_fraction(Element(e)) for e in elements]
            for f in formulas
        ]
    )


def make_rows(
    formulas: list[str],
    fractions: np.ndarray,
    *,
    base_mev: np.ndarray,
    rough_mev: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return one fake field's rows: references, then each quaternary per framework."""
    counts = (fractions > 0).sum(axis=1)
    energy_mev = base_mev + np.where(
        counts == 1, 0.0, rough_mev * rng.normal(size=len(formulas))
    )
    quaternary = np.flatnonzero(counts == len(fractions[0]))
    factor = factor_matern(fractions[quaternary])
    offsets_mev = SPREAD_MEV * np.column_stack(
        [factor @ rng.normal(size=len(quaternary)) for _ in FRAMEWORKS]
    )
    names, energies_mev, frameworks = [], [], []
    for i in np.flatnonzero(counts < len(fractions[0])):
        names.append(formulas[i])
        energies_mev.append(energy_mev[i])
        frameworks.append(None)
    for row, i in enumerate(quaternary):
        for column, framework in enumerate(FRAMEWORKS):
            names.append(formulas[i])
            energies_mev.append(energy_mev[i] + offsets_mev[row, column])
            frameworks.append(framework)
    # Formation energies with every element at zero: a cell's total is atoms times it.
    totals_ev = [
        Composition(f).num_atoms * e / 1000
        for f, e in zip(names, energies_mev, strict=True)
    ]
    return table.make_rows(
        names,
        fidelity="dft",
        reference=[f is None for f in frameworks],
        total_energy_ev=totals_ev,
        framework=frameworks,
    )


def factor_matern(fractions: np.ndarray) -> np.ndarray:
    """Return a Cholesky factor of a unit Matern 5/2 correlation over `fractions`."""
    import torch
    from gpytorch.kernels import MaternKernel

    kernel = MaternKernel(nu=2.5).double()
    kernel.lengthscale = torch.tensor(SCALE, dtype=torch.float64)
    with torch.no_grad():
        correlation = kernel(torch.as_tensor(fractions)).to_dense().numpy()
    return np.linalg.cholesky(correlation + JITTER * np.eye(len(fractions)))


def write_config(name: str, *, size: int, seed: int) -> None:
    text = (
        "# Generated by scripts/make_fake_fields.py; rerun it, do not edit this file.\n"
        "# A fake Li-Sn-S-Cl table: compositions of up to "
        f"{size} atoms per cell, three frameworks, seed {seed}.\n"
        "defaults:\n  - LiSnSCl\n  - _self_\n"
        f"processed: data/processed/{name}.csv  # from scripts/make_fake_fields.py\n"
    )
    (paths.CONFIGS_DIR / "system" / f"{name}.yaml").write_text(text)


if __name__ == "__main__":
    main()
