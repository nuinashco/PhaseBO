"""`phasebo-benchmark` replays BO from many starts and scores it against random."""

import json

import pandas as pd
import pytest

from nuikit.errors import ConfigError

from phasebo.benchmark.cheap.items.noisy import NoisyCheapConfig
from phasebo.benchmark.metric.items.hazard_ratio import HazardRatioConfig
from phasebo.benchmark.metric.items.steps_to_hit import StepsToHitConfig
from phasebo.benchmark.start.items.random import RandomStartConfig
from phasebo.bo.acquisition.items.qlogei import QLogEIConfig
from phasebo.bo.acquisition.items.screen import ScreenConfig
from phasebo.bo.surrogate.items.gp import GPConfig
from phasebo.bo.surrogate.items.multitask import MultiTaskConfig
from phasebo.cli import benchmark
from phasebo.config import BenchmarkConfig, BOConfig, ProtocolConfig

BO = BOConfig(
    surrogate=GPConfig(kind="gp"), acquisition=QLogEIConfig(kind="qlogei"), batch_size=2
)


def config(system, *, size=2, bo=BO, cheap=None):
    return BenchmarkConfig(
        run_name="r",
        system=system,
        bo=bo,
        benchmark=ProtocolConfig(
            cheap=cheap,
            start=RandomStartConfig(kind="random", size=size),
            repeats=2,
            max_iter=2,
            workers=1,
            metric={
                "steps_to_hit": StepsToHitConfig(kind="steps_to_hit"),
                "hazard_ratio": HazardRatioConfig(kind="hazard_ratio"),
            },
        ),
    )


def test_benchmark_writes_every_evaluation_and_the_chosen_metrics(
    system, tmp_path, capsys
):
    run_dir = benchmark.run(config(system), runs_dir=tmp_path / "runs")

    paths = pd.read_parquet(run_dir / "paths.parquet")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    batches = paths.groupby("repeat")["batch"].apply(list).tolist()
    assert batches == [[0, 0, 1, 1, 2, 2]] * 2
    assert list(metrics) == ["evaluations", "steps_to_hit", "hazard_ratio"]
    assert "first hit after" in capsys.readouterr().out


def test_benchmark_refuses_an_oversized_start_before_creating_the_run(system, tmp_path):
    with pytest.raises(ConfigError, match=r"benchmark\.start\.size"):
        benchmark.run(config(system, size=100), runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_benchmark_with_a_noisy_source_runs_the_screen(system, tmp_path, capsys):
    bo = BO.model_copy(update={"acquisition": ScreenConfig(kind="screen")})
    cheap = NoisyCheapConfig(kind="noisy", random_mev=0.0)

    benchmark.run(config(system, bo=bo, cheap=cheap), runs_dir=tmp_path / "runs")

    # Exact cheap energies put a target in every first batch.
    assert "first hit after 2.0 evaluations" in capsys.readouterr().out


def test_benchmark_with_a_cheap_source_writes_each_games_cheap_energies(
    system, points, tmp_path
):
    cheap = NoisyCheapConfig(kind="noisy")

    run_dir = benchmark.run(config(system, cheap=cheap), runs_dir=tmp_path / "runs")

    drawn = pd.read_parquet(run_dir / "cheap.parquet")
    assert len(drawn) == 2 * len(points.pool)
    assert list(drawn.columns) == ["repeat", "formula", "cheap_mev", "e_above_hull_mev"]


def test_benchmark_refuses_a_multi_fidelity_model_without_a_cheap_source(
    system, tmp_path
):
    bo = BO.model_copy(update={"surrogate": MultiTaskConfig(kind="multitask")})

    with pytest.raises(ConfigError, match="set benchmark/cheap=table"):
        benchmark.run(config(system, bo=bo), runs_dir=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()
