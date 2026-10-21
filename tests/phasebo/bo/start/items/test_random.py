"""Random starting sets: explored points off the hull, never more than exist."""

import numpy as np
import pytest

from nuikit.errors import ConfigError

from phasebo.bo.start.base import StartFactory


def test_random_start_never_takes_a_reference_or_a_target(points):
    start = StartFactory.from_config({"kind": "random", "size": 5})

    picked = start.pick(points, rng=np.random.default_rng(0))

    assert len(set(picked)) == 5
    assert set(picked) <= set(points.pool) - set(points.targets)


def test_random_start_refuses_more_points_than_are_off_the_hull(points):
    start = StartFactory.from_config({"kind": "random", "size": 100})

    with pytest.raises(ConfigError, match=r"lower bo\.start\.size"):
        start.check(points)
