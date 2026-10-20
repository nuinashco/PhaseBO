"""Processed phase field table: its schema, and building, cleaning and CSV text."""

import io
from collections.abc import Collection, Iterable, Sequence

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from pandera.typing import DataFrame, Series
from pymatgen.core import Composition

from nuikit.errors import DataError


class ProcessedTable(pa.DataFrameModel):
    """One row per composition and fidelity, carrying exactly one energy."""

    formula: Series[str]
    point: Series[str]  # The reduced composition: cells of one formula share it.
    fidelity: Series[str]
    reference: Series[bool]
    total_energy_ev: Series[float] = pa.Field(nullable=True)  # Per cell.
    e_above_hull_mev: Series[float] = pa.Field(nullable=True)  # Per atom.

    class Config(pa.DataFrameModel.Config):
        strict = True
        ordered = True
        coerce = True

    @pa.dataframe_check
    def one_energy_per_row(cls, df: pd.DataFrame) -> pd.Series:
        return df[["total_energy_ev", "e_above_hull_mev"]].notna().sum(axis=1) == 1

    @pa.dataframe_check
    def has_references(cls, df: pd.DataFrame) -> bool:
        return bool(df["reference"].any())


def make_rows(
    formulas: Sequence[str],
    *,
    fidelity: str,
    reference: Sequence[bool],
    total_energy_ev: Sequence[float] | None = None,
    e_above_hull_mev: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return one source's rows in the shape `clean` takes.

    Give exactly one energy: total (eV per cell) or above the hull (meV/atom).
    """
    if (total_energy_ev is None) == (e_above_hull_mev is None):
        msg = "give total_energy_ev or e_above_hull_mev, not both or neither"
        raise ValueError(msg)
    nan = [float("nan")] * len(formulas)
    return pd.DataFrame(
        {
            "formula": [f.strip() for f in formulas],
            "fidelity": fidelity,
            "reference": list(reference),
            "total_energy_ev": nan if total_energy_ev is None else total_energy_ev,
            "e_above_hull_mev": nan if e_above_hull_mev is None else e_above_hull_mev,
        }
    )


def flag_references(formulas: Iterable[str], references: Iterable[str]) -> list[bool]:
    """Return whether each formula's composition is one of the references."""
    known = [Composition(f) for f in references]
    return [Composition(f) in known for f in formulas]


def clean(
    rows: pd.DataFrame, *, elements: Collection[str], source: str
) -> tuple[DataFrame[ProcessedTable], int]:
    """Return the rows made only of `elements`, with their point; count the others.

    `source` names the raw file the rows came from, in errors.
    """
    compositions = [
        _parse_formula(formula, source=source) for formula in rows["formula"]
    ]
    keep = [{el.symbol for el in c.elements} <= set(elements) for c in compositions]
    table = rows.loc[keep].reset_index(drop=True)
    table.insert(
        1,
        "point",
        [
            c.reduced_composition.formula
            for c, k in zip(compositions, keep, strict=True)
            if k
        ],
    )
    checked = validate(
        table,
        source=f"the table built from {source}",
        remedy="fix the script that reads it",
    )
    return checked, len(rows) - len(table)


def render(table: DataFrame[ProcessedTable]) -> str:
    """Return the CSV text of a processed table."""
    return table.to_csv(index=False, lineterminator="\n")


def parse(text: str, *, source: str) -> DataFrame[ProcessedTable]:
    """Return the validated table that `render` wrote; `source` names it in errors."""
    frame = pd.read_csv(io.StringIO(text), float_precision="round_trip")
    return validate(frame, source=source, remedy="rerun its script")


def validate(
    frame: pd.DataFrame, *, source: str, remedy: str
) -> DataFrame[ProcessedTable]:
    """Return `frame` coerced to ProcessedTable; fail naming every broken rule."""
    try:
        return ProcessedTable.validate(frame, lazy=True)
    except SchemaErrors as e:
        raise DataError(
            f"{source} is not a processed table ({_describe(e)}); {remedy}"
        ) from e


def _describe(error: SchemaErrors) -> str:
    cases = error.failure_cases
    rows = cases.groupby("check", sort=True)["index"].nunique()
    return ", ".join(
        f"{check}: {n} rows" if n else str(check) for check, n in rows.items()
    )


def _parse_formula(formula: str, *, source: str) -> Composition:
    try:
        return Composition(formula)
    except ValueError as e:
        raise DataError(f"cannot read formula {formula!r} in {source}; fix it") from e
