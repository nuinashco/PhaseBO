"""Datasets find their raw files and reference rows, or say what is missing."""

import pytest

from nuikit.errors import DataError

from phasebo.dataset.base import DatasetFactory


def total_energy(raw, *, start_row, references=None):
    return DatasetFactory.from_config(
        {
            "kind": "total_energy_csv",
            "raw": raw,
            "references": {"file": references, "start_row": start_row},
        }
    )


def test_dataset_reports_a_missing_raw_file(tmp_path):
    with pytest.raises(DataError, match="no raw data"):
        total_energy(tmp_path / "missing.csv", start_row=0).read()


def test_dataset_rejects_a_start_row_past_the_last_row(tmp_path):
    (tmp_path / "raw.csv").write_text("formula,energy\nLi1,-1.9\n")

    with pytest.raises(DataError, match=r"references\.start_row"):
        total_energy(tmp_path / "raw.csv", start_row=5).read()


def test_dataset_lists_its_raw_and_reference_files(tmp_path):
    dataset = total_energy(
        tmp_path / "a.csv", start_row=0, references=tmp_path / "b.csv"
    )

    assert dataset.list_files() == [tmp_path / "a.csv", tmp_path / "b.csv"]
