"""Noisy source: fake cheap energies, the modelled ones with errors of a chosen size."""

from typing import Literal

import numpy as np
import pandas as pd
from attrs import define, field
from pydantic import Field, NonNegativeFloat, PositiveFloat
from pymatgen.core import Composition

from nuikit.errors import ConfigError, DataError

from phasebo.benchmark.cheap.base import CheapConfig, CheapSource, CheapSourceFactory
from phasebo.bo.surrogate.base import MATERN_NU, quiet_gpytorch
from phasebo.field import hull
from phasebo.field.points import Points

# Keeps the Cholesky factor of a Matern covariance finite; far below any error size.
_JITTER = 1e-8


class NoisyCheapConfig(CheapConfig):
    """Noisy source settings: a random and a smooth error, mixed, and a bias."""

    random_mev: NonNegativeFloat = Field(
        default=25.0,
        description="std in meV/atom, independent per cell; the errors go on total "
        "energies, then the hull is rebuilt",
    )
    smooth_mev: NonNegativeFloat = Field(
        default=0.0, description="std in meV/atom, shared by nearby points"
    )
    lengthscale: PositiveFloat = Field(
        default=0.2,
        description="of the smooth error's Matern kernel, over all elements' atomic "
        "fractions: moving x between two elements is a distance x*sqrt(2)",
    )
    kernel: Literal["matern12", "matern32", "matern52"] = Field(
        default="matern52",
        description="matern12|matern32|matern52; of the smooth error",
    )
    bias_mev: dict[str, float] = Field(
        default_factory=dict,
        description="framework: meV/atom on each of its cells' total energy",
    )


@CheapSourceFactory.register("noisy")
@define
class NoisyCheapSource(CheapSource[NoisyCheapConfig]):
    """Errors added to each cell's total energy, then the hull rebuilt; drawn per game.

    Both errors are drawn whatever their size, so settings of one seed share them. A
    composition's frameworks share its smooth error.
    """

    _factors: dict[tuple[tuple[int, ...], bytes], np.ndarray] = field(
        factory=dict, init=False, repr=False, eq=False
    )

    def check(self, rows: pd.DataFrame, points: Points, *, fidelity: str) -> None:
        unknown = sorted(set(self.config.bias_mev) - set(points.framework_names))
        if unknown:
            known = ", ".join(points.framework_names) or "none"
            raise ConfigError(
                f"benchmark.cheap.bias_mev names {unknown[0]!r}, not a framework of "
                f"this field ({known}); name only those"
            )
        cells = _select_cells(rows, points, fidelity=fidelity)
        bare = sorted(set(points.names) - set(cells["point"]))
        if bare:
            # Commands build points from these rows, so only a caller's bug gets here.
            msg = f"{len(bare)} points, such as {bare[0]}, have no {fidelity!r} rows"
            raise ValueError(msg)
        if cells["total_energy_ev"].isna().to_numpy().any():
            raise DataError(
                f"the {fidelity!r} rows give energies above the hull, and "
                "benchmark/cheap=noisy needs total energies; choose a phase field "
                "whose table has them, such as system=LiSnSCl"
            )
        if self.config.smooth_mev > 0:
            # Built here, in the command's process, so every worker gets it with `self`.
            self._factor_correlation(points.coords[_first_of_each(points)])

    def draw(
        self,
        rows: pd.DataFrame,
        points: Points,
        *,
        fidelity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cells = _select_cells(rows, points, fidelity=fidelity)
        frameworks = [None] * len(cells)
        if "framework" in cells:
            frameworks = [None if pd.isna(f) else f for f in cells["framework"]]
        at = points.find(cells["point"].tolist(), frameworks)
        first = _first_of_each(points)
        composition = {points.names[i]: c for c, i in enumerate(first)}
        formulas = cells["formula"].tolist()
        random_error_mev = self.config.random_mev * rng.normal(size=len(cells))
        normals = rng.normal(size=len(first))
        bias_mev = np.array(
            [self.config.bias_mev.get(f, 0.0) if f else 0.0 for f in frameworks]
        )
        error_mev = random_error_mev + bias_mev
        if self.config.smooth_mev > 0:
            shared = self._factor_correlation(points.coords[first]) @ normals
            of = np.array([composition[name] for name in cells["point"]])
            error_mev = error_mev + self.config.smooth_mev * shared[of]
        atoms = np.array([Composition(f).num_atoms for f in formulas])
        totals_ev = cells["total_energy_ev"].to_numpy(dtype=float)
        above_mev = hull.measure(
            formulas, (totals_ev + error_mev / 1000 * atoms).tolist()
        )
        energies_mev = np.full(len(points.formulas), np.nan)
        np.fmin.at(energies_mev, at, above_mev)
        return energies_mev

    def _factor_correlation(self, coords: np.ndarray) -> np.ndarray:
        """Return the smooth error's Cholesky factor over `coords`, built once."""
        key = (coords.shape, coords.tobytes())
        if key not in self._factors:
            self._factors[key] = self._build_factor(coords)
        return self._factors[key]

    def _build_factor(self, coords: np.ndarray) -> np.ndarray:
        """Return a Cholesky factor of the smooth error's correlation over `coords`."""
        import torch

        with quiet_gpytorch():
            from gpytorch.kernels import MaternKernel

        # Coordinates leave out the first element; with its fraction back, moving
        # atoms between any two elements covers the same distance.
        fractions = np.column_stack([1 - coords.sum(axis=1), coords])
        kernel = MaternKernel(nu=MATERN_NU[self.config.kernel]).double()
        kernel.lengthscale = torch.tensor(self.config.lengthscale, dtype=torch.float64)
        with torch.no_grad():
            correlation = kernel(torch.as_tensor(fractions)).to_dense().numpy()
        return np.linalg.cholesky(correlation + _JITTER * np.eye(len(coords)))


def _first_of_each(points: Points) -> np.ndarray:
    """Return each composition's first point, in field order."""
    # Field order, so a field without frameworks draws as before.
    _, first = np.unique(np.asarray(points.names), return_index=True)
    return np.sort(first)


def _select_cells(rows: pd.DataFrame, points: Points, *, fidelity: str) -> pd.DataFrame:
    """Return the rows at `fidelity` of `points`' points, every cell of each."""
    inside = rows["point"].isin(points.names) & (rows["fidelity"] == fidelity)
    return rows.loc[inside].reset_index(drop=True)
