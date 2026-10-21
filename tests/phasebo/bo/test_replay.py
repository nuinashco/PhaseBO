"""A replay evaluates its starts, then batches of unevaluated explored points."""

import numpy as np

from phasebo.bo import replay
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.start.base import StartFactory


def run(points, surrogate, *, acquisition=None, max_iter, seed=0):
    return replay.replay(
        points,
        start=StartFactory.from_config({"kind": "random", "size": 2}),
        surrogate=surrogate,
        acquisition=acquisition or AcquisitionFactory.from_config({"kind": "random"}),
        batch_size=2,
        max_iter=max_iter,
        rng=np.random.default_rng(seed),
    )


def test_replay_evaluates_the_starts_then_each_batch_without_a_fit(
    points, recording_surrogate
):
    path = run(points, recording_surrogate, max_iter=3)

    assert path["batch"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert path["formula"].is_unique
    assert recording_surrogate.sizes == []


def test_replay_fits_a_model_based_rule_to_every_point_evaluated(
    points, recording_surrogate, first_choices
):
    run(points, recording_surrogate, acquisition=first_choices, max_iter=3)

    assert recording_surrogate.sizes == [2, 4, 6]


def test_replay_never_picks_a_reference(points, recording_surrogate):
    path = run(points, recording_surrogate, max_iter=50)

    references = {points.formulas[i] for i in np.flatnonzero(points.references)}
    assert not references & set(path["formula"])


def test_replay_stops_when_every_explored_point_is_evaluated(
    points, recording_surrogate
):
    path = run(points, recording_surrogate, max_iter=50)

    assert len(path) == len(points.pool)


def test_replay_marks_the_targets_it_finds(points, recording_surrogate):
    path = run(points, recording_surrogate, max_iter=50)

    found = set(path.loc[path["target"], "formula"])
    assert found == {points.formulas[i] for i in points.targets}


def test_replay_repeats_with_its_seed(points, recording_surrogate):
    first = run(points, recording_surrogate, max_iter=3, seed=1)

    assert first.equals(run(points, recording_surrogate, max_iter=3, seed=1))
