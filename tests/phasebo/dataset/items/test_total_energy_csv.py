"""A total-energy CSV gives one row per composition, as DFT codes list them."""

from phasebo.dataset.base import DatasetFactory

RAW = (
    "Composition, Total E eV, meV/atom above CH\n"
    "  Li2 S2, -20.0, 5.0\n"
    "  Li1, -1.9, 0.0\n"
    "  S1, -4.1, 0.0\n"
)


def test_total_energy_csv_flags_rows_from_the_start_row_as_references(tmp_path):
    (tmp_path / "raw.csv").write_text(RAW)
    dataset = DatasetFactory.from_config(
        {
            "kind": "total_energy_csv",
            "raw": tmp_path / "raw.csv",
            "references": {"start_row": 1},
        }
    )

    rows = dataset.read()

    assert (
        rows["formula"].tolist(),
        rows["reference"].tolist(),
        rows["total_energy_ev"].tolist(),
        bool(rows["e_above_hull_mev"].isna().to_numpy().all()),
    ) == (["Li2 S2", "Li1", "S1"], [False, True, True], [-20.0, -1.9, -4.1], True)
