"""An energy-above-hull CSV gives one row per composition and fidelity."""

import pytest

from nuikit.errors import DataError

from phasebo.dataset.base import DatasetFactory


def hull_energy(tmp_path, fidelities):
    (tmp_path / "raw.csv").write_text(
        "Formula,DFT,NNmodel\nLi2 S2,5.0,9.5\nLi1,0.0,0.0\n"
    )
    (tmp_path / "refs.csv").write_text("formula,energy\nLi2 S2,-20.0\nLi1,-1.9\n")
    return DatasetFactory.from_config(
        {
            "kind": "hull_energy_csv",
            "raw": tmp_path / "raw.csv",
            "fidelities": fidelities,
            "references": {"file": tmp_path / "refs.csv", "start_row": 1},
        }
    )


def test_hull_energy_csv_gives_one_row_per_fidelity(tmp_path):
    rows = hull_energy(tmp_path, {"DFT": "dft", "NNmodel": "nn"}).read()

    assert (
        rows["fidelity"].tolist(),
        rows["e_above_hull_mev"].tolist(),
        bool(rows["total_energy_ev"].isna().to_numpy().all()),
    ) == (["dft", "dft", "nn", "nn"], [5.0, 0.0, 9.5, 0.0], True)


def test_hull_energy_csv_takes_references_from_another_file(tmp_path):
    rows = hull_energy(tmp_path, {"DFT": "dft"}).read()

    assert rows["reference"].tolist() == [False, True]


def test_hull_energy_csv_names_a_missing_column(tmp_path):
    with pytest.raises(DataError, match="no column NN"):
        hull_energy(tmp_path, {"NN": "nn"}).read()
