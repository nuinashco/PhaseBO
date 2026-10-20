"""Processed phase field table: cleaning, validation and its CSV text."""

import io
from collections.abc import Collection

import pandas as pd
from pymatgen.core import Composition

from nuikit.errors import DataError

_COLUMNS = (
    "formula",
    "point",
    "fidelity",
    "reference",
    "total_energy_ev",
    "e_above_hull_mev",
)


def clean(rows: pd.DataFrame, *, elements: Collection[str]) -> tuple[pd.DataFrame, int]:
    """Return the rows made only of `elements`, with their point; count the others."""
    compositions = [_parse(formula) for formula in rows["formula"]]
    keep = [{el.symbol for el in c.elements} <= set(elements) for c in compositions]
    table = rows.loc[keep].reset_index(drop=True)
    # A point is the reduced composition: cells of one formula share it.
    table["point"] = [
        c.reduced_composition.formula
        for c, kept in zip(compositions, keep, strict=True)
        if kept
    ]
    table = table.loc[:, list(_COLUMNS)]
    _validate(table)
    return table, len(rows) - len(table)


def render(table: pd.DataFrame) -> str:
    """Return the CSV text of a processed table."""
    return table.to_csv(index=False, lineterminator="\n")


def parse(text: str, *, source: str) -> pd.DataFrame:
    """Return the validated table that `render` wrote; `source` names it in errors."""
    table = pd.read_csv(
        io.StringIO(text),
        float_precision="round_trip",
        dtype={"formula": str, "point": str, "fidelity": str, "reference": bool},
    )
    if tuple(table.columns) != _COLUMNS:
        raise DataError(f"{source} is not a processed table; run `make data`")
    _validate(table)
    return table


def _parse(formula: str) -> Composition:
    try:
        return Composition(formula)
    except ValueError as e:
        raise DataError(f"cannot read formula {formula!r}; fix the raw file") from e


def _validate(table: pd.DataFrame) -> None:
    energies = table[["total_energy_ev", "e_above_hull_mev"]].notna().sum(axis=1)
    if bad := int((energies != 1).sum()):
        raise DataError(
            f"{bad} rows need exactly one energy, total or above the hull; "
            "check the raw file"
        )
    if not table["reference"].to_numpy(dtype=bool).any():
        raise DataError(
            "no reference compositions in the phase field; "
            "set system.dataset.references"
        )
