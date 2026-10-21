"""Comparisons pair runs game by game, and refuse runs that are not paired."""

import numpy as np
import pytest

from nuikit.errors import ConfigError

from phasebo.benchmark import comparison
from phasebo.benchmark.metric.base import MetricFactory
from phasebo.benchmark.replay import Replays


def run_compare(base, runs):
    for name, replays in runs.items():
        comparison.check_paired(base, replays, name=name)
    return comparison.compare(
        base,
        runs,
        metrics={"discovery": MetricFactory.from_config({"kind": "discovery"})},
        level=0.9,
        resamples=500,
        rng=np.random.default_rng(0),
    )


def test_compare_keys_each_runs_contrasts_by_metric_and_outcome(made):
    contrasts = run_compare(made([{}, {}]), {"bo": made([{0: 1}, {}])})

    assert list(contrasts) == ["bo"]
    assert list(contrasts["bo"]) == ["discovery/found"]


def test_compare_refuses_runs_from_other_starting_sets(made):
    base = made([{}, {}])
    # Every repeat starts from one point, which no set of the baseline does.
    starts = base.paths["point"].where(
        base.paths["batch"] > 0, base.paths["point"].max()
    )
    other = Replays(
        points=base.points,
        batch_size=2,
        max_iter=3,
        paths=base.paths.assign(point=starts),
    )

    with pytest.raises(ConfigError, match="starts from other sets"):
        run_compare(base, {"other": other})


def test_compare_refuses_runs_of_another_budget(made):
    base = made([{}])
    shorter = Replays(points=base.points, batch_size=2, max_iter=2, paths=base.paths)

    with pytest.raises(ConfigError, match="picks 4 times, the baseline 6"):
        run_compare(base, {"shorter": shorter})
