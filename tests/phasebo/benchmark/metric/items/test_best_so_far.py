"""Best so far tracks the lowest energy known, from the starting set on."""

import numpy as np

from phasebo.benchmark.metric.base import MetricFactory


def measure(replays):
    return MetricFactory.from_config({"kind": "best_so_far"}).measure(replays)


def test_best_so_far_starts_at_the_start_and_drops_with_a_target(made):
    curves = measure(made([{0: 2}, {}])).curves

    assert curves["mean"][0] == curves["random"][0]
    assert np.all(np.diff(curves["mean"]) <= 0)
    assert curves["mean"][2] < curves["mean"][1]


def test_best_so_far_of_random_picking_converges_on_random(random_replays):
    values = measure(random_replays).values

    gap = abs(values["at_budget/mean"] - values["at_budget/random"])
    assert gap < 4 * values["at_budget/se"]
