"""Commands read the processed table and their inputs, and fail fast on a gap."""

import hashlib
import json

import pytest
from pymatgen.core import Composition

from nuikit.errors import DataError

from phasebo.cli import _launch
from phasebo.config import ElementConfig


def test_read_table_points_to_make_data(system, tmp_path):
    missing = system.model_copy(update={"processed": tmp_path / "missing.csv"})

    with pytest.raises(DataError, match="run `make data`"):
        _launch.read_table(missing)


def test_fingerprint_hashes_the_processed_and_raw_files(system, tmp_path):
    _launch.write_data(tmp_path, _launch.fingerprint(system))

    recorded = json.loads((tmp_path / "data.json").read_text())
    digest = hashlib.sha256(system.processed.read_bytes()).hexdigest()
    assert (recorded["processed"]["sha256"], [f["path"] for f in recorded["raw"]]) == (
        digest,
        [str(system.dataset.raw)],
    )


def test_fingerprint_rejects_a_missing_raw_file(system):
    system.dataset.raw.unlink()

    with pytest.raises(DataError, match=r"fix system\.dataset\.raw"):
        _launch.fingerprint(system)


def test_to_legacy_inputs_split_off_the_references(system, li_sn_s_cl):
    compositions, references, _ = li_sn_s_cl

    field, refs = _launch.to_legacy_inputs(_launch.read_table(system))

    assert (field[:, 0].tolist(), refs[:, 0].tolist()) == (
        compositions[:, 0].tolist(),
        references[:, 0].tolist(),
    )
    assert field[:, 1].astype(float) == pytest.approx(compositions[:, 1].astype(float))


def test_to_legacy_inputs_need_total_energies(system):
    rows = _launch.read_table(system).assign(
        total_energy_ev=float("nan"), e_above_hull_mev=0.0
    )

    with pytest.raises(DataError, match="needs total energies"):
        _launch.to_legacy_inputs(rows)


def test_map_charges_keeps_the_config_order(system):
    assert list(_launch.map_charges(system).items()) == [
        ("Li", 1),
        ("Sn", 4),
        ("S", -2),
        ("Cl", -1),
    ]


def test_generate_formulas_skips_computed_and_excluded_points(system):
    rows = _launch.read_table(system)
    point = Composition("Li1 Sn1 S1 Cl3").reduced_composition

    every = _launch.generate_formulas(system, rows, excluded=None)
    kept = _launch.generate_formulas(system, rows, excluded=["Li1 Sn1 S1 Cl3"])

    computed = {Composition(f).reduced_composition for f in rows["formula"]}
    points = {Composition(f).reduced_composition for f in kept}
    assert point in {Composition(f).reduced_composition for f in every}
    assert (point in points, points & computed) == (False, set())


def test_select_candidates_prefers_the_listed_formulas(system, candidates):
    rows = _launch.read_table(system)

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
            empty, _launch.read_table(empty), listed=None, excluded=None
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
