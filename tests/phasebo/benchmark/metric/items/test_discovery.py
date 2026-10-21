"""Discovery curves give the share of repeats that found a target, batch by batch."""

import numpy as np
import pytest

from phasebo.benchmark.metric.base import MetricFactory


def measure(replays):
    return MetricFactory.from_config({"kind": "discovery"}).measure(replays)


def test_discovery_curves_follow_the_hits_and_random(made):
    replays = made([{0: 1, 1: 3}, {1: 2}])
    first, second = replays.target_names

    curves = measure(replays).curves

    assert curves[first] == [0.0, 0.5, 0.5, 0.5]
    assert curves[second] == [0.0, 0.0, 0.5, 1.0]
    assert curves["first_hit"] == [0.0, 0.5, 1.0, 1.0]
    assert curves[f"{first}/random"] == pytest.approx([0, 2 / 11, 4 / 11, 6 / 11])


def test_discovery_of_random_picking_converges_on_random(random_replays):
    curves = measure(random_replays).curves

    found = np.array(curves["first_hit"], dtype=float)
    gap = np.abs(found - np.array(curves["first_hit/random"], dtype=float))
    assert gap.max() < 0.05
