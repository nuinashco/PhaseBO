"""Commands read their inputs and fail fast on any missing file."""

import pytest

from nuikit.errors import DataError

from phasebo.cli import _launch
from phasebo.config import ElementConfig


def test_read_field_rejects_a_missing_file(system, tmp_path):
    data = system.data.model_copy(update={"file": tmp_path / "missing.csv"})

    with pytest.raises(DataError, match=r"set system\.data\.file"):
        _launch.read_field(system.model_copy(update={"data": data}))


def test_read_field_rejects_references_past_the_last_row(system):
    data = system.data.model_copy(update={"reference_index": 10_000})

    with pytest.raises(DataError, match=r"set system\.data\.reference_index"):
        _launch.read_field(system.model_copy(update={"data": data}))


def test_read_field_splits_off_the_references(system, li_sn_s_cl):
    compositions, references, _ = li_sn_s_cl

    rows, refs = _launch.read_field(system)

    assert (rows[:, 0].tolist(), refs[:, 0].tolist()) == (
        compositions[:, 0].tolist(),
        references[:, 0].tolist(),
    )
    # `pandas` may round the last bit of a CSV float, depending on the platform.
    energies = compositions[:, 1].astype(float)
    assert rows[:, 1].astype(float) == pytest.approx(energies)


def test_charges_keep_the_config_order(system):
    assert list(_launch.charges(system).items()) == [
        ("Li", 1),
        ("Sn", 4),
        ("S", -2),
        ("Cl", -1),
    ]


def test_atom_limits_make_inclusive_ranges_half_open(system):
    elements = {
        "Li": ElementConfig(charge=1, atoms=(1, 3)),
        "Cl": ElementConfig(charge=-1),
    }

    limits = _launch.atom_limits(system.model_copy(update={"elements": elements}))

    assert limits == {"Li": (1, 4), "Cl": (0, system.candidates.max_atoms + 1)}


def test_read_formulas_returns_none_for_a_null_path():
    assert _launch.read_formulas(None, key="system.excludefile") is None


def test_read_formulas_rejects_a_missing_file(tmp_path):
    with pytest.raises(DataError, match=r"fix system\.excludefile or set it to null"):
        _launch.read_formulas(tmp_path / "missing.csv", key="system.excludefile")


def test_read_formulas_returns_the_first_column(tmp_path):
    (tmp_path / "f.csv").write_text("formula,note\nLi2 S1,a\nLi1 Cl1,b\n")

    formulas = _launch.read_formulas(tmp_path / "f.csv", key="system.excludefile")

    assert formulas == ["Li2 S1", "Li1 Cl1"]
