"""Replays of the synthetic field: built by hand, or by random picking."""

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

from phasebo.benchmark import replay
from phasebo.benchmark.replay import Replays
from phasebo.benchmark.start.base import StartFactory
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory
from phasebo.field.points import Points

type ReplaysMaker = Callable[[Sequence[Mapping[int, int]]], Replays]


@pytest.fixture
def made(points: Points) -> ReplaysMaker:
    """Return a maker of replays, batches of 2 for 3 batches, that hit as told.

    Each mapping takes a target's position in `points.targets` to the batch that
    picks it; every other pick, and the 2-point start, is a point off the hull.
    """

    def make(hits: Sequence[Mapping[int, int]]) -> Replays:
        filler = [int(i) for i in np.setdiff1d(points.pool, points.targets)]
        rows = []
        for repeat, plan in enumerate(hits):
            spare = iter(filler)
            rows += [(repeat, 0, next(spare)) for _ in range(2)]
            for batch in range(1, 4):
                picked = [int(points.targets[t]) for t, b in plan.items() if b == batch]
                picked += [next(spare) for _ in range(2 - len(picked))]
                rows += [(repeat, batch, point) for point in picked]
        paths = pd.DataFrame(rows, columns=["repeat", "batch", "point"])
        return Replays(points=points, batch_size=2, max_iter=3, paths=paths)

    return make


@pytest.fixture
def random_replays(points: Points) -> Replays:
    """Return 2000 replays of random picking: their metrics converge on random's."""
    return replay.replay_many(
        points,
        start=StartFactory.from_config({"kind": "random", "size": 2}),
        surrogate=SurrogateFactory.from_config({"kind": "gp"}),
        acquisition=AcquisitionFactory.from_config({"kind": "random"}),
        batch_size=2,
        max_iter=3,
        repeats=2000,
        seed=0,
        workers=1,
    )
