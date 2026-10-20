"""A processed table loads only while it and its raw files match its sidecar."""

import pytest

from nuikit.errors import DataError

from phasebo import processed


def test_load_returns_the_rows_and_their_sidecar(system):
    loaded = processed.load(system.processed)

    raw = [r.path.name for r in loaded.sidecar.raw]
    assert (len(loaded.rows), raw, loaded.sidecar.script.name) == (
        22,
        ["raw.csv"],
        "process.py",
    )


def test_find_problems_catches_a_malformed_sidecar(system):
    system.processed.with_suffix(".json").write_text("{}")

    assert "is malformed" in processed.find_problems(system.processed)[0]


def test_find_problems_is_empty_for_a_fresh_table(system):
    assert processed.find_problems(system.processed) == []


def test_find_problems_names_the_script_when_a_raw_file_changed(system):
    raw = system.processed.parent / "raw.csv"
    raw.write_text(raw.read_text() + "Li1,-2.0\n")

    assert processed.find_problems(system.processed) == [
        "raw file "
        f"{raw} changed; rerun `uv run {system.processed.parent / 'process.py'}`"
    ]


def test_find_problems_catches_a_table_edited_by_hand(system):
    system.processed.write_text(system.processed.read_text() + "\n")

    assert (
        "edited since its script wrote it"
        in processed.find_problems(system.processed)[0]
    )


def test_find_problems_catches_a_missing_sidecar(system):
    system.processed.with_suffix(".json").unlink()

    assert "no sidecar processed.json" in processed.find_problems(system.processed)[0]


def test_find_problems_catches_a_missing_table(tmp_path):
    assert "missing" in processed.find_problems(tmp_path / "absent.csv")[0]


def test_load_refuses_an_untrustworthy_table(system):
    (system.processed.parent / "raw.csv").unlink()

    with pytest.raises(DataError, match=r"raw file .* is missing"):
        processed.load(system.processed)
