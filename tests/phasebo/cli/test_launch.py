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


def test_generate_formulas_skips_explored_points(system):
    rows = _launch.load_table(system).rows
    computed = list(rows["formula"])

    every = _launch.generate_formulas(system, explored=computed)
    kept = _launch.generate_formulas(system, explored=[*computed, "Li1 Sn1 S1 Cl3"])

    points = {Composition(f).reduced_composition for f in every}
    assert points & {Composition(f).reduced_composition for f in computed} == set()
    assert ("Li1 Sn1 S1 Cl3" in every, "Li1 Sn1 S1 Cl3" in kept) == (True, False)


def test_select_candidates_prefers_the_listed_formulas(system, candidates):
    explored = list(_launch.load_table(system).rows["formula"])

    selected = _launch.select_candidates(system, explored=explored, listed=candidates)

    assert selected == candidates


def test_select_candidates_drops_listed_formulas_at_explored_points(system):
    listed = ["Li6 Sn1 S5", "Li1 Sn1 S1 Cl3"]

    selected = _launch.select_candidates(
        system, explored=["Li12 Sn2 S10"], listed=listed
    )

    assert selected == ["Li1 Sn1 S1 Cl3"]


def test_select_candidates_refuses_an_empty_list(system):
    empty = system.model_copy(
        update={
            "elements": {
                symbol: ElementConfig(charge=element.charge, atoms=(0, 0))
                for symbol, element in system.elements.items()
            }
        }
    )

    with pytest.raises(DataError, match=r"widen system\.elements atoms"):
        _launch.select_candidates(empty, explored=[], listed=None)


def test_select_candidates_refuses_listed_formulas_that_are_all_explored(system):
    with pytest.raises(DataError, match=r"list others in system\.candidates\.file"):
        _launch.select_candidates(
            system, explored=["Li3 S1 Cl1"], listed=["Li3 S1 Cl1", "Li6 S2 Cl2"]
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


def test_family_tags_name_each_choice_and_setting():
    tags = _launch.family_tags(
        surrogate=GPConfig(kind="gp"), acquisition=QLogEIConfig(kind="qlogei")
    )

    assert tags == [
        "surrogate=gp",
        "surrogate.kernel=matern52",
        "surrogate.noise_mev=0.0",
        "acquisition=qlogei",
    ]


def test_field_name_is_the_processed_tables(system):
    assert _launch.field_name(system) == "processed"
