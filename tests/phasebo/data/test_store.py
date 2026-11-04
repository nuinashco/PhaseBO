"""A processed table loads only while it and its raw files match its sidecar."""

import pytest

from nuikit.errors import DataError

from phasebo.data import store


def test_load_returns_the_rows_and_their_sidecar(system):
    loaded = store.load(system.processed)

    raw = [r.path.name for r in loaded.sidecar.raw]
    script = loaded.sidecar.script
    assert (len(loaded.rows), raw, script and script.path.name) == (
        22,
        ["raw.csv"],
        "process.py",
    )


def test_a_computed_table_names_the_command_that_rebuilds_it(system):
    rows = store.load(system.processed).rows
    raw = system.processed.parent / "raw.csv"
    path = system.processed.with_name("computed.csv")
    store.write(path, rows, raw=[raw], command="phasebo-csp run_name=r")
    raw.write_text(raw.read_text() + "Li1,-2.0\n")

    assert store.find_problems(path) == [
        f"{raw} changed since the table was written; rerun `phasebo-csp run_name=r`"
    ]


def test_a_table_and_its_sidecar_are_written_alike_on_every_platform(system, tmp_path):
    rows = store.load(system.processed).rows
    raw = tmp_path / "nested" / "raw.csv"
    raw.parent.mkdir()
    raw.write_text("Li1,-1.0\n")
    path = tmp_path / "table.csv"
    store.write(path, rows, raw=[raw], command="phasebo-csp run_name=r")

    sidecar = path.with_suffix(".json").read_bytes()
    assert (b"\r\n" in path.read_bytes(), b"\r\n" in sidecar, b"\\\\" in sidecar) == (
        False,
        False,
        False,
    )


def test_find_problems_catches_a_malformed_sidecar(system):
    system.processed.with_suffix(".json").write_text("{}")

    assert "is malformed" in store.find_problems(system.processed)[0]


def test_find_problems_is_empty_for_a_fresh_table(system):
    assert store.find_problems(system.processed) == []


def test_find_problems_names_the_script_when_a_raw_file_changed(system):
    raw = system.processed.parent / "raw.csv"
    raw.write_text(raw.read_text() + "Li1,-2.0\n")
    script = system.processed.parent / "process.py"

    assert store.find_problems(system.processed) == [
        f"{raw} changed since the table was written; rerun `uv run {script}`"
    ]


def test_find_problems_catches_a_script_edited_without_a_rerun(system):
    script = system.processed.parent / "process.py"
    script.write_text(script.read_text() + "FIRST_REFERENCE_ROW = 0\n")

    assert f"{script} changed since" in store.find_problems(system.processed)[0]


def test_find_problems_catches_a_table_edited_by_hand(system):
    system.processed.write_text(system.processed.read_text() + "\n")

    assert "edited since it was written" in store.find_problems(system.processed)[0]


def test_find_problems_catches_a_missing_sidecar(system):
    system.processed.with_suffix(".json").unlink()

    assert "no sidecar processed.json" in store.find_problems(system.processed)[0]


def test_find_problems_catches_a_missing_table(tmp_path):
    assert "missing" in store.find_problems(tmp_path / "absent.csv")[0]


def test_load_refuses_an_untrustworthy_table(system):
    (system.processed.parent / "raw.csv").unlink()

    with pytest.raises(DataError, match=r"raw\.csv is missing"):
        store.load(system.processed)
