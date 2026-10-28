"""Table source: another fidelity of the processed table, such as a network's."""

import numpy as np
import pandas as pd
from attrs import define
from pydantic import Field

from nuikit.errors import DataError

from phasebo.benchmark.cheap.base import Cheap, CheapConfig, CheapFactory
from phasebo.field.points import Points


class TableCheapConfig(CheapConfig):
    """Table source settings."""

    fidelity: str = Field(
        default="nn", description="a value of the processed table's fidelity column"
    )


@CheapFactory.register("table")
@define
class TableCheap(Cheap[TableCheapConfig]):
    """Each point's lowest energy at the configured fidelity, the same in every game."""

    def check(self, rows: pd.DataFrame, points: Points, *, fidelity: str) -> None:
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
        index = {name: i for i, name in enumerate(points.names)}
        cheap = Points.from_rows(
            rows.loc[rows["point"].isin(index)],
            fidelity=self.config.fidelity,
            elements=points.elements,
        )
        energies_mev = np.full(len(index), np.nan)
        energies_mev[[index[name] for name in cheap.names]] = cheap.energies_mev
        return energies_mev
