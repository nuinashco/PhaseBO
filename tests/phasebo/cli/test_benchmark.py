"""`phasebo-benchmark` replays BO over the computed points and writes its path."""

import pandas as pd
import pytest

from nuikit.errors import ConfigError

from phasebo.bo.acquisition.items.qlogei import QLogEIConfig
from phasebo.bo.start.items.random import RandomStartConfig
from phasebo.bo.surrogate.items.gp import GPConfig
from phasebo.cli import benchmark
from phasebo.config import BenchmarkBOConfig, BenchmarkConfig


def test_benchmark_writes_the_path_and_its_plots(system, tmp_path, capsys):
    cfg = BenchmarkConfig(
        run_name="r",
        system=system,
        bo=BenchmarkBOConfig(
            surrogate=GPConfig(kind="gp"),
            acquisition=QLogEIConfig(kind="qlogei"),
            start=RandomStartConfig(kind="random", size=2),
            batch_size=2,
            max_iter=2,
        ),
    )

    run_dir = benchmark.run(cfg, runs_dir=tmp_path / "runs")

    path = pd.read_csv(run_dir / "bo_path.csv")
    assert path["batch"].tolist() == [0, 0, 1, 1, 2, 2]
    assert "targets found in 6 evaluations" in capsys.readouterr().out
    assert {"convergence.png", "convex_hull.png"} <= {p.name for p in run_dir.iterdir()}


def test_benchmark_refuses_an_oversized_start_before_creating_the_run(system, tmp_path):
    cfg = BenchmarkConfig(
        system=system,
        bo=BenchmarkBOConfig(
            surrogate=GPConfig(kind="gp"),
            acquisition=QLogEIConfig(kind="qlogei"),
            start=RandomStartConfig(kind="random", size=100),
        ),
    )

    with pytest.raises(ConfigError, match=r"bo\.start\.size"):
        benchmark.run(cfg, runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
