"""The hazard ratio is 1 for random picking and grows with earlier hits."""

from phasebo.benchmark.metric.base import MetricFactory
from phasebo.benchmark.replay import Replays


def measure(replays):
    return MetricFactory.from_config({"kind": "hazard_ratio"}).measure(replays)


def test_hazard_ratio_is_zero_without_hits(made):
    assert measure(made([{}, {}])).values == {"theta": 0.0}


def test_hazard_ratio_is_zero_when_random_could_not_miss(made, points):
    replays = made([{0: 1}])
    certain = Replays(
        points=points, batch_size=len(points.pool) - 2, max_iter=1, paths=replays.paths
    )

    assert measure(certain).values == {"theta": 0.0}


def test_hazard_ratio_grows_when_targets_are_hit_earlier(made):
    early = measure(made([{0: 1}, {}] * 10)).values["theta"]
    late = measure(made([{0: 3}, {}] * 10)).values["theta"]

    assert early > late


def test_hazard_ratio_of_random_picking_is_one(random_replays):
    values = measure(random_replays).values

    assert values["low"] < 1 < values["high"]
