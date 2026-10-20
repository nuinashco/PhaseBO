"""Processed tables keep the phase field's rows, each with its point, and validate."""

import pandas as pd
import pytest

from nuikit.errors import DataError

from phasebo.dataset import table
from phasebo.dataset.base import COLUMNS

NAN = float("nan")


def raw(*rows: tuple[object, ...]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(COLUMNS))


FIELD = raw(
    ("Li2 S1", "dft", True, -11.9, NAN),
    ("Li2 O1", "dft", True, -14.3, NAN),
    ("Li4 S2", "dft", False, -23.0, NAN),
)


def test_clean_drops_rows_outside_the_phase_field():
    rows, dropped = table.clean(FIELD, elements=["Li", "S"])

    assert (rows["formula"].tolist(), dropped) == (["Li2 S1", "Li4 S2"], 1)


def test_clean_gives_cells_of_one_formula_one_point():
    rows, _ = table.clean(FIELD, elements=["Li", "S"])

    assert rows["point"].tolist() == ["Li2 S1", "Li2 S1"]


def test_clean_rejects_a_row_without_an_energy():
    rows = raw(("Li2 S1", "dft", True, NAN, NAN))

    with pytest.raises(DataError, match="exactly one energy"):
        table.clean(rows, elements=["Li", "S"])


def test_clean_rejects_a_field_without_references():
    rows = raw(("Li2 S1", "dft", False, -11.9, NAN))

    with pytest.raises(DataError, match="no reference compositions"):
        table.clean(rows, elements=["Li", "S"])


def test_clean_drops_an_unknown_element_as_off_field():
    rows = raw(("Li2 S1", "dft", True, -11.9, NAN), ("Li2 Qq1", "dft", True, -5.0, NAN))

    assert table.clean(rows, elements=["Li", "S"])[1] == 1


def test_clean_rejects_an_unreadable_formula():
    rows = raw(("Li2 S1)", "dft", True, -11.9, NAN))

    with pytest.raises(DataError, match="cannot read formula"):
        table.clean(rows, elements=["Li", "S"])


def test_parse_returns_what_render_wrote():
    rows, _ = table.clean(FIELD, elements=["Li", "S"])

    assert table.parse(table.render(rows), source="t.csv").equals(rows)


def test_parse_rejects_text_that_is_not_a_processed_table():
    with pytest.raises(DataError, match=r"t\.csv is not a processed table"):
        table.parse("formula,energy\nLi2 S1,-11.9\n", source="t.csv")
