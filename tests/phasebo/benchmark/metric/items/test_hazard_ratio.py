"""The hazard ratio is 1 for random picking and grows with earlier hits."""

import numpy as np
import pytest

from phasebo.benchmark.metric.base import MetricFactory
from phasebo.benchmark.replay import Replays


def measure(replays):
    return MetricFactory.from_config({"kind": "hazard_ratio"}).measure(replays)


def test_hazard_ratio_is_zero_without_hits(made):
    assert measure(made([{}, {}])).values == {"theta": 0.0, "theta/random": 1.0}


def test_hazard_ratio_is_zero_when_random_could_not_miss(made, points):
    replays = made([{0: 1}])
    certain = Replays(
        points=points, batch_size=len(points.pool) - 2, max_iter=1, paths=replays.paths
    )

    assert measure(certain).values == {"theta": 0.0, "theta/random": 1.0}


def test_hazard_ratio_grows_when_targets_are_hit_earlier(made):
    early = measure(made([{0: 1}, {}] * 10)).values["theta"]
    late = measure(made([{0: 3}, {}] * 10)).values["theta"]

    assert early > late


def test_hazard_ratio_of_random_picking_is_one(random_replays):
    values = measure(random_replays).values

    assert values["low"] < 1 < values["high"]


def test_hazard_per_batch_matches_random_for_random_picking(random_replays):
    curves = measure(random_replays).curves

    assert curves["per_batch"][0] is None
    assert curves["per_batch"][1:] == pytest.approx([1.0] * 3, abs=0.25)
    assert curves["hit_chance/random"][1:] == pytest.approx([2 / 11, 2 / 9, 2 / 7])


def test_hazard_per_batch_is_undefined_once_every_target_is_found(made):
    score = measure(made([{0: 1, 1: 1}] * 3))
    curves = score.curves

    # Batch 1 found every target, a certain hit; later batches had none to find.
    assert curves["hit_chance"][1:] == [1.0, None, None]
    assert curves["per_batch"][1:] == [None, None, None]
    assert "theta" not in score.values


def contrast(base, run):
    return MetricFactory.from_config({"kind": "hazard_ratio"}).contrast(
        base, run, level=0.9, resamples=500, rng=np.random.default_rng(0)
    )


def test_hazard_contrast_favours_the_run_that_hits_earlier(made):
    late, early = made([{0: 3}, {}] * 10), made([{0: 1}, {}] * 10)

    ratio = contrast(late, early)["theta"]

    assert (ratio.scale, ratio.estimate > 1, ratio.better) == ("ratio", True, None)


def test_hazard_contrast_needs_hits_in_both_runs(made):
    assert contrast(made([{}, {}]), made([{0: 1}, {}])) == {}
