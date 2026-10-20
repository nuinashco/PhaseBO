"""`phasebo-benchmark` replays BO over the computed compositions, reproducibly."""

import pandas as pd

from phasebo.cli import benchmark
from phasebo.config import BenchmarkConfig, BOConfig


def replay(system, runs_dir, *, seed, run_name):
    cfg = BenchmarkConfig(
        run_name=run_name,
        seed=seed,
        system=system,
        bo=BOConfig(batch_size=2),
        n_starts=2,
        max_iter=2,
    )
    return pd.read_csv(benchmark.run(cfg, runs_dir=runs_dir) / "bo_path.csv")


def test_benchmark_writes_the_starts_then_every_batch(system, tmp_path):
    path = replay(system, tmp_path, seed=0, run_name="r")

    assert path["seed"].tolist() == [True] * 2 + [False] * 4


def test_benchmark_path_depends_only_on_the_seed(system, tmp_path):
    first = replay(system, tmp_path, seed=0, run_name="a")
    again = replay(system, tmp_path, seed=0, run_name="b")
    other = replay(system, tmp_path, seed=1, run_name="c")

    assert first.equals(again)
    assert not first.equals(other)
