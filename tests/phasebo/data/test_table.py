"""Processed tables keep the phase field's rows, each with its point, and validate."""

import pytest

from nuikit.errors import DataError

from phasebo.data import table


def rows(formulas, reference, energies):
    return table.make_rows(
        formulas, fidelity="dft", reference=reference, total_energy_ev=energies
    )


FIELD = rows(["Li2 S1", "Li2 O1", "Li4 S2"], [True, True, False], [-11.9, -14.3, -23.0])


def test_make_rows_wants_exactly_one_energy():
    with pytest.raises(ValueError, match="not both or neither"):
        table.make_rows(["Li1"], fidelity="dft", reference=[True])


def test_flag_references_matches_compositions_not_strings():
    flags = table.flag_references(["S1 Li2", "Li4 S2"], ["Li2 S1"])

    assert flags == [True, False]


def test_clean_drops_rows_outside_the_phase_field():
    cleaned, dropped = table.clean(FIELD, elements=["Li", "S"])

    assert (cleaned["formula"].tolist(), dropped) == (["Li2 S1", "Li4 S2"], 1)


def test_clean_gives_cells_of_one_formula_one_point():
    cleaned, _ = table.clean(FIELD, elements=["Li", "S"])

    assert cleaned["point"].tolist() == ["Li2 S1", "Li2 S1"]


def test_clean_rejects_a_field_without_references():
    with pytest.raises(DataError, match="has_references"):
        table.clean(rows(["Li2 S1"], [False], [-11.9]), elements=["Li", "S"])


def test_clean_drops_an_unknown_element_as_off_field():
    field = rows(["Li2 S1", "Li2 Qq1"], [True, True], [-11.9, -5.0])

    assert table.clean(field, elements=["Li", "S"])[1] == 1


def test_clean_rejects_an_unreadable_formula():
    with pytest.raises(DataError, match="cannot read formula"):
        table.clean(rows(["Li2 S1)"], [True], [-11.9]), elements=["Li", "S"])


def test_parse_returns_what_render_wrote():
    cleaned, _ = table.clean(FIELD, elements=["Li", "S"])

    assert table.parse(table.render(cleaned), source="t.csv").equals(cleaned)


def test_parse_names_every_broken_rule():
    cleaned, _ = table.clean(FIELD, elements=["Li", "S"])
    broken = cleaned.assign(reference=False, e_above_hull_mev=1.0)

    with pytest.raises(DataError) as error:
        table.parse(broken.to_csv(index=False), source="t.csv")

    assert "has_references, one_energy_per_row: 2 rows" in str(error.value)


def test_parse_rejects_text_that_is_not_a_processed_table():
    with pytest.raises(DataError, match=r"t\.csv is not a processed table"):
        table.parse("formula,energy\nLi2 S1,-11.9\n", source="t.csv")
