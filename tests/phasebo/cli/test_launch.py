"""Commands read their inputs and fail fast on any missing file."""

import pytest

from nuikit.errors import DataError

from phasebo.cli import _launch


def test_read_field_rejects_a_missing_file(system, tmp_path):
    missing = system.model_copy(update={"inputfile": tmp_path / "missing.csv"})

    with pytest.raises(DataError, match=r"set system\.inputfile"):
        _launch.read_field(missing)


def test_read_field_rejects_references_past_the_last_row(system):
    past = system.model_copy(update={"reference_index": 10_000})

    with pytest.raises(DataError, match=r"set system\.reference_index"):
        _launch.read_field(past)


def test_read_field_splits_off_the_references(system, li_sn_s_cl):
    compositions, references, _ = li_sn_s_cl

    rows, refs = _launch.read_field(system)

    assert (rows[:, 0].tolist(), refs[:, 0].tolist()) == (
        compositions[:, 0].tolist(),
        references[:, 0].tolist(),
    )
    # pandas' default CSV float parser may round the last bit, depending on platform.
    energies = compositions[:, 1].astype(float)
    assert rows[:, 1].astype(float) == pytest.approx(energies)


def test_read_formulas_returns_none_for_a_null_path():
    assert _launch.read_formulas(None, key="system.excludefile") is None


def test_read_formulas_rejects_a_missing_file(tmp_path):
    with pytest.raises(DataError, match=r"fix system\.excludefile or set it to null"):
        _launch.read_formulas(tmp_path / "missing.csv", key="system.excludefile")


def test_read_formulas_returns_the_first_column(tmp_path):
    (tmp_path / "f.csv").write_text("formula,note\nLi2 S1,a\nLi1 Cl1,b\n")

    formulas = _launch.read_formulas(tmp_path / "f.csv", key="system.excludefile")

    assert formulas == ["Li2 S1", "Li1 Cl1"]
