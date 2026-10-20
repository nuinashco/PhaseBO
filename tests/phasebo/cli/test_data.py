"""`phasebo-data` passes fresh tables and fails on stale ones."""

from click.testing import CliRunner

from phasebo.cli import data


def test_data_passes_fresh_tables(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])

    result = CliRunner().invoke(data.main, [])

    assert (result.exit_code, f"ok {system.processed}" in result.output) == (0, True)


def test_data_fails_on_a_stale_table(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])
    raw = system.processed.parent / "raw.csv"
    raw.write_text(raw.read_text() + "Li1,-2.0\n")

    result = CliRunner().invoke(data.main, [])

    assert result.exit_code == 3
