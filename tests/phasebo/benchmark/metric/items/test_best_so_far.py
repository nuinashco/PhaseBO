"""Best so far tracks the lowest energy known, from the starting set on."""

import numpy as np

from phasebo.benchmark.metric.base import MetricFactory


def measure(replays):
    return MetricFactory.from_config({"kind": "best_so_far"}).measure(replays)


def test_best_so_far_starts_at_the_start_and_drops_with_a_target(made):
    curves = measure(made([{0: 2}, {}])).curves
    mean = np.array(curves["mean"], dtype=float)

    assert mean[0] == curves["mean/random"][0]
    assert np.all(np.diff(mean) <= 0)
    assert mean[2] < mean[1]


def test_best_so_far_of_random_picking_converges_on_random(random_replays):
    values = measure(random_replays).values

    gap = abs(values["at_budget/mean"] - values["at_budget/random"])
    assert gap < 4 * values["at_budget/se"]


def contrast(base, run):
    return MetricFactory.from_config({"kind": "best_so_far"}).contrast(
        base, run, level=0.9, resamples=500, rng=np.random.default_rng(0)
    )


def test_best_so_far_contrast_is_lower_when_the_run_finds_a_target(made):
    at_budget = contrast(made([{}, {}]), made([{0: 3}, {0: 3}]))["at_budget"]

    assert (at_budget.estimate < 0, at_budget.better) == (True, 1.0)
