"""Points of a phase field: one entry per reduced composition, with its coordinates."""

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from attrs import field, frozen
from pymatgen.core import Composition, Element

from nuikit.errors import DataError

from phasebo.field import hull

# Distances to the hull from `pymatgen` carry rounding noise of about 1e-13 meV/atom.
_ON_HULL_MEV = 1e-6


@frozen
class Points:
    """The points of a phase field: each one's cell, coordinates, energy and role.

    A point is a reduced composition; it keeps its lowest-energy cell. Coordinates
    are the atomic fractions of every element but the first, in `pymatgen` element
    order, which makes them its phase diagram coordinates.
    """

    elements: tuple[str, ...]
    formulas: tuple[str, ...]
    coords: np.ndarray = field(eq=False)
    energies_mev: np.ndarray = field(eq=False)  # Above the hull, per atom.
    references: np.ndarray = field(eq=False)

    @property
    def pool(self) -> np.ndarray:
        """Return the indices of the explored points, those BO may pick."""
        return np.flatnonzero(~self.references)

    @property
    def targets(self) -> np.ndarray:
        """Return the indices of the explored points on the hull."""
        return np.flatnonzero(~self.references & (self.energies_mev <= _ON_HULL_MEV))

    def locate(self, formulas: Sequence[str]) -> np.ndarray:
        """Return the coordinates of `formulas` in this phase field's basis."""
        return _project(formulas, self.elements)

    @classmethod
    def from_rows(
        cls,
        rows: pd.DataFrame,
        *,
        fidelity: str,
        elements: Collection[str],
        excluded: Collection[str] = (),
    ) -> Points:
        """Return the points of one fidelity's rows of a processed table.

        `excluded` formulas drop every row at their points before the hull is built.
        """
        found = sorted(set(rows["fidelity"]))
        rows = rows.loc[rows["fidelity"] == fidelity].reset_index(drop=True)
        if rows.empty:
            raise DataError(
                f"no rows of fidelity {fidelity!r}; set fidelity to one of "
                f"{', '.join(found)}"
            )
        left_out = sorted(
            {Composition(f).reduced_composition.formula for f in excluded}
        )
        rows = rows.loc[~rows["point"].isin(left_out)].reset_index(drop=True)
        energies_mev = _read_energies(rows)
        groups = pd.DataFrame(
            {
                "point": rows["point"],
                "energy": energies_mev,
                "reference": rows["reference"],
            }
        ).groupby("point", sort=False)
        # `idxmin` keeps the first of tied cells, in table order.
        lowest = groups["energy"].idxmin().to_numpy()
        order = tuple(str(el) for el in sorted(Element(symbol) for symbol in elements))
        formulas = tuple(rows["formula"].to_numpy()[lowest])
        return cls(
            elements=order,
            formulas=formulas,
            coords=_project(formulas, order),
            energies_mev=energies_mev[lowest],
            references=groups["reference"].any().to_numpy(dtype=bool),
        )


def _project(formulas: Sequence[str], elements: Sequence[str]) -> np.ndarray:
    """Return each formula's atomic fractions of every element but the first."""
    basis = [Element(symbol) for symbol in elements[1:]]
    return np.array(
        [[Composition(f).get_atomic_fraction(el) for el in basis] for f in formulas]
    ).reshape(len(formulas), len(basis))


def _read_energies(rows: pd.DataFrame) -> np.ndarray:
    totals_ev = rows["total_energy_ev"].to_numpy(dtype=float)
    given_mev = rows["e_above_hull_mev"].to_numpy(dtype=float)
    if not np.isnan(totals_ev).any():
        return hull.measure(rows["formula"].tolist(), totals_ev.tolist())
    if not np.isnan(given_mev).any():
        return given_mev
    raise DataError(
        "rows of one fidelity mix total energies and energies above the hull; "
        "fix the script that writes the processed table"
    )
