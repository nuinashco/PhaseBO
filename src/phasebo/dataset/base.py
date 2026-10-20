"""Datasets: raw files of computed compositions, one reader per file format."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
from attrs import define
from pydantic import Field, NonNegativeInt
from pymatgen.core import Composition

from nuikit.config import Schema
from nuikit.errors import DataError
from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo import paths

COLUMNS = ("formula", "fidelity", "reference", "total_energy_ev", "e_above_hull_mev")


class ReferencesConfig(Schema):
    """Rows of a raw file that list the reference compositions."""

    file: Path | None = Field(default=None, description="null: the dataset's own file")
    start_row: NonNegativeInt = Field(description="counted from 0, header excluded")


class DatasetConfig(KindConfig):
    """Raw file of computed compositions."""

    raw: Path = Field(description="relative to the repository root")
    references: ReferencesConfig


@define
class Dataset[C: DatasetConfig](Configurable[C], ABC):
    """Reader of one raw format into rows of COLUMNS, one per composition and fidelity.

    A composition listed in the reference rows is a reference wherever it appears.
    Each row carries one energy: total (eV per cell) or above the hull (meV/atom).
    """

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Return the raw file's rows with the references flagged."""

    def list_files(self) -> list[Path]:
        """Return the raw files this dataset reads, references included."""
        refs = self.config.references.file
        named = [self.config.raw] if refs is None else [self.config.raw, refs]
        return [paths.ROOT / path for path in named]

    def _read_raw(self) -> pd.DataFrame:
        return read_csv(paths.ROOT / self.config.raw, key="system.dataset.raw")

    def _flag_references(self, formulas: pd.Series) -> pd.Series:
        refs = self.config.references
        if refs.file is None:
            rows = self._read_raw()
        else:
            rows = read_csv(
                paths.ROOT / refs.file, key="system.dataset.references.file"
            )
        known = [Composition(f) for f in rows.iloc[refs.start_row :, 0].str.strip()]
        if not known:
            raise DataError(
                f"no reference rows from row {refs.start_row}; "
                "set system.dataset.references.start_row"
            )
        return formulas.map(lambda formula: Composition(formula) in known)


class DatasetFactory(ConfigurableFactory[Dataset[Any]]):
    """Registry of dataset readers by `kind`."""

    config_group = "dataset"


def read_csv(path: Path, *, key: str) -> pd.DataFrame:
    """Read a raw CSV with exact floats; fail on a missing file, naming config `key`."""
    if not path.is_file():
        raise DataError(f"no raw data at {path}; fix {key}")
    # The default fast parser may round the last bit, which would make
    # processed tables differ between platforms.
    return pd.read_csv(path, skipinitialspace=True, float_precision="round_trip")
