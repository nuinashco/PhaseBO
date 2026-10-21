"""Commands read the processed table and their inputs, and fail fast on a gap."""

import pytest
from pymatgen.core import Composition

from nuikit.errors import DataError

from phasebo.bo.acquisition.items.qlogei import QLogEIConfig
from phasebo.bo.surrogate.items.gp import GPConfig
from phasebo.cli import _launch
from phasebo.config import BOConfig, ElementConfig, SuggestConfig


def test_load_points_drops_the_excluded_points(system, tmp_path):
    (tmp_path / "exclude.csv").write_text("formula\nLi6 S2 Cl2\n")
    excluding = system.model_copy(
        update={
            "candidates": system.candidates.model_copy(
                update={"exclude": tmp_path / "exclude.csv"}
            )
        }
    )
    cfg = SuggestConfig(
        system=excluding,
        bo=BOConfig(
            surrogate=GPConfig(kind="gp"), acquisition=QLogEIConfig(kind="qlogei")
        ),
    )

    _, points, excluded = _launch.load_points(cfg)

    assert (excluded, "Li3 S1 Cl1" in points.formulas) == (["Li6 S2 Cl2"], False)


def test_map_charges_keeps_the_config_order(system):
    assert list(_launch.map_charges(system).items()) == [
        ("Li", 1),
        ("Sn", 4),
        ("S", -2),
        ("Cl", -1),
    ]


def test_generate_formulas_skips_computed_and_excluded_points(system):
    rows = _launch.load_table(system).rows
    point = Composition("Li1 Sn1 S1 Cl3").reduced_composition

    every = _launch.generate_formulas(system, rows, excluded=None)
    kept = _launch.generate_formulas(system, rows, excluded=["Li1 Sn1 S1 Cl3"])

    computed = {Composition(f).reduced_composition for f in rows["formula"]}
    points = {Composition(f).reduced_composition for f in kept}
    assert point in {Composition(f).reduced_composition for f in every}
    assert (point in points, points & computed) == (False, set())


def test_select_candidates_prefers_the_listed_formulas(system, candidates):
    rows = _launch.load_table(system).rows

    selected = _launch.select_candidates(system, rows, listed=candidates, excluded=None)

    assert selected == candidates


def test_select_candidates_refuses_an_empty_list(system):
    empty = system.model_copy(
        update={
            "elements": {
                symbol: ElementConfig(charge=element.charge, atoms=(0, 0))
                for symbol, element in system.elements.items()
            }
        }
    )

    with pytest.raises(DataError, match="no candidates left"):
        _launch.select_candidates(
            empty, _launch.load_table(empty).rows, listed=None, excluded=None
        )


def test_read_formulas_returns_none_for_a_null_path():
    assert _launch.read_formulas(None, key="system.candidates.exclude") is None


def test_read_formulas_rejects_a_missing_file(tmp_path):
    with pytest.raises(DataError, match=r"fix system\.candidates\.exclude or set it"):
        _launch.read_formulas(tmp_path / "missing.csv", key="system.candidates.exclude")


def test_read_formulas_returns_the_first_column(tmp_path):
    (tmp_path / "f.csv").write_text("formula,note\nLi2 S1,a\nLi1 Cl1,b\n")

    formulas = _launch.read_formulas(tmp_path / "f.csv", key="system.candidates.file")

    assert formulas == ["Li2 S1", "Li1 Cl1"]
