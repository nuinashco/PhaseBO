"""Table source: another fidelity of the processed table, such as a network's."""

import numpy as np
import pandas as pd
from attrs import define
from pydantic import Field

from nuikit.errors import DataError

from phasebo.benchmark.cheap.base import CheapConfig, CheapSource, CheapSourceFactory
from phasebo.field.points import Points


class TableCheapConfig(CheapConfig):
    """Table source settings."""

    fidelity: str = Field(
        default="nn", description="a value of the processed table's fidelity column"
    )


@CheapSourceFactory.register("table")
@define
class TableCheapSource(CheapSource[TableCheapConfig]):
    """Each point's lowest energy at the configured fidelity, the same in every game."""

    def check(self, rows: pd.DataFrame, points: Points, *, fidelity: str) -> None:
        found = set(rows.loc[rows["point"].isin(points.names), "fidelity"])
        if self.config.fidelity not in found:
            raise DataError(
                f"the table has no {self.config.fidelity!r} rows at this field's "
                "points; set benchmark.cheap.fidelity to one of "
                f"{', '.join(sorted(found))}"
            )
        energies_mev = self._read_energies(rows, points)
        missing = [points.names[i] for i in points.pool if np.isnan(energies_mev[i])]
        if missing:
            raise DataError(
                f"fidelity {self.config.fidelity!r} has no energy for {len(missing)} "
                f"explored points, such as {missing[0]}; choose a "
                "benchmark.cheap.fidelity that covers them"
            )

    def draw(
        self,
        rows: pd.DataFrame,
        points: Points,
        *,
        fidelity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return self._read_energies(rows, points)

    def _read_energies(self, rows: pd.DataFrame, points: Points) -> np.ndarray:
        cheap = Points.from_rows(
            rows.loc[rows["point"].isin(points.names)],
            fidelity=self.config.fidelity,
            elements=points.elements,
        )
        at = points.find(cheap.names, cheap.frameworks)
        energies_mev = np.full(len(points.formulas), np.nan)
        energies_mev[at[at >= 0]] = cheap.energies_mev[at >= 0]
        return energies_mev
