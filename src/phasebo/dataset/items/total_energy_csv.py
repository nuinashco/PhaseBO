"""CSV of formulas and total energies (eV per cell), as DFT codes report them."""

import pandas as pd
from attrs import define

from phasebo.dataset.base import Dataset, DatasetConfig, DatasetFactory


class TotalEnergyCSVConfig(DatasetConfig):
    """Formula in the first column, total energy in the second; later ones ignored."""

    fidelity: str = "dft"


@DatasetFactory.register("total_energy_csv")
@define
class TotalEnergyCSVDataset(Dataset[TotalEnergyCSVConfig]):
    """Reader of a formula, total energy CSV."""

    def read(self) -> pd.DataFrame:
        raw = self._read_raw()
        formulas = raw.iloc[:, 0].str.strip()
        return pd.DataFrame(
            {
                "formula": formulas,
                "fidelity": self.config.fidelity,
                "reference": self._flag_references(formulas),
                "total_energy_ev": raw.iloc[:, 1].astype(float),
                "e_above_hull_mev": float("nan"),
            }
        )
