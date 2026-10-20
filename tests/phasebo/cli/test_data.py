"""`phasebo-data` finds stale processed tables, rewrites them, or fails on --check."""

from click.testing import CliRunner

from phasebo.cli import data


def test_find_stale_lists_a_missing_table(system):
    system.processed.unlink()

    assert [s.processed for s, _, _ in data.find_stale([system])] == [system.processed]


def test_find_stale_lists_an_edited_table(system):
    system.processed.write_text("edited\n", encoding="utf-8")

    assert [s.processed for s, _, _ in data.find_stale([system])] == [system.processed]


def test_find_stale_skips_a_fresh_table(system):
    assert data.find_stale([system]) == []


def test_check_fails_naming_the_stale_tables(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])
    system.processed.unlink()

    result = CliRunner().invoke(data.main, ["--check"])

    assert (result.exit_code, system.processed.is_file()) == (3, False)
    assert str(system.processed) in result.output


def test_main_rewrites_a_stale_table(system, monkeypatch):
    monkeypatch.setattr(data, "compose_systems", lambda: [system])
    fresh = system.processed.read_text(encoding="utf-8")
    system.processed.write_text("edited\n", encoding="utf-8")

    result = CliRunner().invoke(data.main, [])

    assert (result.exit_code, system.processed.read_text(encoding="utf-8")) == (
        0,
        fresh,
    )
