"""Noisy source: fake cheap energies, the modelled ones with errors of a chosen size."""

import numpy as np
import pandas as pd
from attrs import define
from pydantic import Field, NonNegativeFloat, PositiveFloat
from pymatgen.core import Composition

from nuikit.errors import DataError

from phasebo.benchmark.cheap.base import Cheap, CheapConfig, CheapFactory
from phasebo.bo.surrogate.base import quiet_gpytorch
from phasebo.field import hull
from phasebo.field.points import Points

# Keeps the Cholesky factor of a Matern covariance finite; far below any error size.
_JITTER = 1e-8


class NoisyCheapConfig(CheapConfig):
    """Noisy source settings: a random and a smooth error, mixed."""

    random_mev: NonNegativeFloat = Field(
        default=25.0,
        description="std in meV/atom, independent per cell; both errors go on total "
        "energies, then the hull is rebuilt",
    )
    smooth_mev: NonNegativeFloat = Field(
        default=0.0, description="std in meV/atom, shared by nearby points"
    )
    lengthscale: PositiveFloat = Field(
        default=0.2, description="of the smooth error's Matern 5/2, in atomic fractions"
    )


@CheapFactory.register("noisy")
@define
class NoisyCheap(Cheap[NoisyCheapConfig]):
    """Errors added to each cell's total energy, then the hull rebuilt; drawn per game.

    Both errors are drawn whatever their size, so settings of one seed share them.
    """

    def check(self, rows: pd.DataFrame, points: Points, *, fidelity: str) -> None:
        totals_ev = _select_cells(rows, points, fidelity=fidelity)["total_energy_ev"]
        if totals_ev.isna().to_numpy().any():
            raise DataError(
                f"the {fidelity!r} rows give energies above the hull, and "
                "benchmark/cheap=noisy needs total energies; choose a phase field "
                "whose table has them, such as system=LiSnSCl"
            )

    def draw(
        self,
        rows: pd.DataFrame,
        points: Points,
        *,
        fidelity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        cells = _select_cells(rows, points, fidelity=fidelity)
        index = {name: i for i, name in enumerate(points.names)}
        at = np.array([index[name] for name in cells["point"]])
        formulas = cells["formula"].tolist()
        random_mev = self.config.random_mev * rng.normal(size=len(cells))
        shared = self._factor_correlation(points.coords) @ rng.normal(
            size=len(points.formulas)
        )
        error_mev = random_mev + self.config.smooth_mev * shared[at]
        atoms = np.array([Composition(f).num_atoms for f in formulas])
        totals_ev = cells["total_energy_ev"].to_numpy(dtype=float)
        above_mev = hull.measure(
            formulas, (totals_ev + error_mev / 1000 * atoms).tolist()
        )
        energies_mev = np.full(len(points.formulas), np.inf)
        np.minimum.at(energies_mev, at, above_mev)
        return energies_mev

    def _factor_correlation(self, coords: np.ndarray) -> np.ndarray:
        """Return a Cholesky factor of the smooth error's correlation over `coords`."""
        import torch

        with quiet_gpytorch():
            from gpytorch.kernels import MaternKernel

        kernel = MaternKernel(nu=2.5).double()
        kernel.lengthscale = torch.tensor(self.config.lengthscale)
        with torch.no_grad():
            correlation = kernel(torch.as_tensor(coords)).to_dense().numpy()
        return np.linalg.cholesky(correlation + _JITTER * np.eye(len(coords)))


def _select_cells(rows: pd.DataFrame, points: Points, *, fidelity: str) -> pd.DataFrame:
    """Return the rows at `fidelity` of `points`' points, every cell of each."""
    inside = rows["point"].isin(points.names) & (rows["fidelity"] == fidelity)
    return rows.loc[inside].reset_index(drop=True)
