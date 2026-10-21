"""Random picking's exact odds match counting every possible draw."""

import itertools

import numpy as np
import pytest

from phasebo.benchmark import baseline


def test_survival_is_the_share_of_draws_that_miss_every_target():
    draws = list(itertools.combinations(range(7), 3))
    missed = sum(not {0, 1} & set(draw) for draw in draws)

    assert baseline.survival(7, 3, targets=2) == pytest.approx(missed / len(draws))


def test_hazard_is_the_batch_share_of_the_points_left_and_at_most_one():
    assert baseline.hazard(10, 4, batch=2) == pytest.approx(2 / 6)
    assert baseline.hazard(10, 8, batch=4) == 1.0


def test_expected_min_averages_the_lowest_of_every_draw():
    values = np.array([5.0, 1.0, 3.0, 8.0, 2.0])
    lowest = [min(4.0, *draw) for draw in itertools.combinations(values, 2)]

    assert baseline.expected_min(values, 2, floor_mev=4.0) == pytest.approx(
        np.mean(lowest)
    )


def test_expected_min_without_draws_is_the_floor():
    assert baseline.expected_min(np.array([1.0]), 0, floor_mev=4.0) == 4.0
