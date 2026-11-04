"""`phasebo-csp` computes each job once and writes a table the benchmark reads."""

import json

import pytest

from nuikit.errors import ConfigError, DataError

from phasebo.cli import csp
from phasebo.config import CSPConfig, PredictionConfig, ReferencesConfig
from phasebo.csp.mlip.items.emt import EMTConfig
from phasebo.csp.search.items.basin_hop import BasinHopConfig
from phasebo.csp.search.items.relax import RelaxConfig
from phasebo.data import store
from phasebo.field.points import Points

# Short relaxes keep each run under a second; the searches' own tests relax fully.
STEPS = {"steps": 20, "release_steps": 20}


def config(
    references_dir,
    *,
    epochs=2,
    formulas=("Al4N4",),
    frameworks=("wurtzite",),
    publish=None,
):
    return CSPConfig(
        run_name="r",
        formulas=list(formulas),
        csp=PredictionConfig(
            mlip=EMTConfig(kind="emt"),
            search={
                "relax": RelaxConfig(kind="relax", **STEPS),
                "basin_hop": BasinHopConfig(kind="basin_hop", epochs=epochs, **STEPS),
            },
            frameworks=list(frameworks),
            references=ReferencesConfig(directory=references_dir),
        ),
        publish=publish,
    )


def read_lines(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [{k: v for k, v in row.items() if k != "elapsed_s"} for row in rows]


def test_csp_writes_a_table_whose_fidelities_the_benchmark_reads(
    references_dir, tmp_path
):
    cfg = config(references_dir, frameworks=("wurtzite", "cubic"))

    run_dir = csp.run(cfg, runs_dir=tmp_path / "runs")

    rows = store.load(run_dir / "table.csv").rows
    points = Points.from_rows(rows, fidelity="emt_bh2", elements=["Al", "N"])
    assert sorted(set(rows["fidelity"])) == ["emt_bh2", "emt_relax"]
    assert (points.framework_names, int(points.references.sum())) == (
        ("cubic", "wurtzite"),
        2,
    )


def test_csp_resumes_by_computing_only_the_missing_jobs(references_dir, tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = csp.run(config(references_dir), runs_dir=runs_dir)
    first = read_lines(run_dir / "results.jsonl")
    (run_dir / "results" / "Al4N4_wurtzite_emt_bh2.json").unlink()

    csp.run(config(references_dir), runs_dir=runs_dir)

    assert read_lines(run_dir / "results.jsonl") == first
    assert len((run_dir / "resumes.jsonl").read_text().splitlines()) == 1


def test_csp_refuses_to_resume_a_run_with_other_settings(references_dir, tmp_path):
    csp.run(config(references_dir), runs_dir=tmp_path / "runs")

    with pytest.raises(ConfigError, match=r"\(csp\.search\.basin_hop\.epochs\)"):
        csp.run(config(references_dir, epochs=3), runs_dir=tmp_path / "runs")


def test_csp_without_references_writes_results_but_no_table(tmp_path):
    run_dir = csp.run(config(None), runs_dir=tmp_path / "runs")

    assert len(read_lines(run_dir / "results.jsonl")) == 2
    assert not (run_dir / "table.csv").exists()


def test_csp_refuses_references_that_miss_an_element(references_dir, tmp_path):
    (references_dir / "N2.cif").unlink()

    with pytest.raises(DataError, match="no reference of N alone"):
        csp.run(config(references_dir), runs_dir=tmp_path / "runs")


def test_csp_refuses_a_run_without_formulas(references_dir, tmp_path):
    with pytest.raises(ConfigError, match="no formulas"):
        csp.run(config(references_dir, formulas=()), runs_dir=tmp_path / "runs")


def test_publish_writes_the_table_with_copies_of_what_it_pins(references_dir, tmp_path):
    processed = tmp_path / "processed"

    csp.run(
        config(references_dir, publish="AlN_emt"),
        runs_dir=tmp_path / "runs",
        processed_dir=processed,
    )

    loaded = store.load(processed / "AlN_emt.csv")
    assert {pin.path.parent for pin in loaded.sidecar.raw} == {processed / "AlN_emt"}
    assert loaded.sidecar.command == (
        "phasebo-csp run_name=r publish=AlN_emt with the settings in "
        f"{processed / 'AlN_emt' / 'config.yaml'}"
    )
