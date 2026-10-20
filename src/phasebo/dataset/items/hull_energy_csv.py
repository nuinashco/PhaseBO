"""CSV of formulas and energies above the convex hull (meV/atom), by fidelity."""

import pandas as pd
from attrs import define
from pydantic import Field

from nuikit.errors import DataError

from phasebo.dataset.base import Dataset, DatasetConfig, DatasetFactory


class HullEnergyCSVConfig(DatasetConfig):
    """Formula in the first column, then one column per fidelity."""

    fidelities: dict[str, str] = Field(description="column: fidelity")


@DatasetFactory.register("hull_energy_csv")
@define
class HullEnergyCSVDataset(Dataset[HullEnergyCSVConfig]):
    """Reader of energies above the hull, one row per composition and fidelity."""

    def read(self) -> pd.DataFrame:
        raw = self._read_raw()
        if missing := sorted(set(self.config.fidelities) - set(raw.columns)):
            columns = ", ".join(map(str, raw.columns[1:]))
            raise DataError(
                f"no column {', '.join(missing)} in {self.config.raw}; "
                f"key system.dataset.fidelities by its columns: {columns}"
            )
        formulas = raw.iloc[:, 0].str.strip()
        reference = self._flag_references(formulas)
        return pd.concat(
            [
                pd.DataFrame(
                    {
                        "formula": formulas,
                        "fidelity": fidelity,
                        "reference": reference,
                        "total_energy_ev": float("nan"),
                        "e_above_hull_mev": raw[column].astype(float),
                    }
                )
                for column, fidelity in self.config.fidelities.items()
            ],
            ignore_index=True,
        )
