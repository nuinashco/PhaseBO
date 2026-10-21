"""`phasebo-compare` reads benchmark runs and compares them with the first."""

import json

import pytest

from nuikit.errors import ConfigError

from phasebo.benchmark.metric.items.discovery import DiscoveryConfig
from phasebo.benchmark.start.items.random import RandomStartConfig
from phasebo.bo.acquisition.items.random import RandomAcquisitionConfig
from phasebo.bo.surrogate.items.gp import GPConfig
from phasebo.cli import benchmark, compare
from phasebo.config import BenchmarkConfig, BOConfig, CompareConfig, ProtocolConfig


def benchmark_run(system, runs_dir, *, name, seed):
    cfg = BenchmarkConfig(
        run_name=name,
        seed=seed,
        system=system,
        bo=BOConfig(
            surrogate=GPConfig(kind="gp"),
            acquisition=RandomAcquisitionConfig(kind="random"),
            batch_size=2,
        ),
        benchmark=ProtocolConfig(
            start=RandomStartConfig(kind="random", size=2),
            repeats=6,
            max_iter=2,
            workers=1,
            metric={"discovery": DiscoveryConfig(kind="discovery")},
        ),
    )
    return benchmark.run(cfg, runs_dir=runs_dir).name


def test_compare_writes_each_runs_contrast_with_the_baseline(system, tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    first = benchmark_run(system, runs_dir, name="a", seed=1)
    second = benchmark_run(system, runs_dir, name="b", seed=1)
    capsys.readouterr()

    run_dir = compare.run(
        CompareConfig(run_name="c", runs=[first, second]), runs_dir=runs_dir
    )

    report = json.loads((run_dir / "comparison.json").read_text())
    assert (report["baseline"], [row["run"] for row in report["rows"]]) == ("a", ["b"])
    assert capsys.readouterr().out.startswith("baseline: a\nb\n  discovery/found")


def test_compare_refuses_unpaired_runs_before_creating_its_run(system, tmp_path):
    runs_dir = tmp_path / "runs"
    first = benchmark_run(system, runs_dir, name="a", seed=1)
    second = benchmark_run(system, runs_dir, name="b", seed=2)

    with pytest.raises(ConfigError, match="starts from other sets"):
        compare.run(
            CompareConfig(run_name="c", runs=[first, second]), runs_dir=runs_dir
        )
    assert not (runs_dir / "c").exists()


def test_compare_needs_two_runs_or_more(tmp_path):
    with pytest.raises(ConfigError, match="needs two runs or more"):
        compare.run(CompareConfig(runs=["only"]), runs_dir=tmp_path / "runs")
