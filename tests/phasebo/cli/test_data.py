"""`phasebo-data` passes fresh tables and fails naming what to rerun."""

from click.testing import CliRunner

from phasebo.cli import data


def test_data_passes_fresh_tables(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])

    result = CliRunner().invoke(data.main, [])

    assert (result.exit_code, f"ok {system.processed}" in result.output) == (0, True)


def test_data_fails_naming_the_script_to_rerun(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])
    raw = system.processed.parent / "raw.csv"
    raw.write_text(raw.read_text() + "Li1,-2.0\n")

    result = CliRunner().invoke(data.main, [])

    assert (result.exit_code, "process.py" in result.output) == (3, True)
