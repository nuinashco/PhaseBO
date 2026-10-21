"""Replays evaluate a starting set, then batches; repeats are paired and repeatable."""

import numpy as np
import pytest

from nuikit.errors import DataError

from phasebo.benchmark import replay
from phasebo.benchmark.start.base import StartFactory
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory

START = {"kind": "random", "size": 2}


def run_one(points, surrogate, *, acquisition=None, max_iter=3):
    starting = StartFactory.from_config(START).pick(
        points, rng=np.random.default_rng(0)
    )
    batches = replay.replay(
        points,
        starting,
        surrogate=surrogate,
        acquisition=acquisition or AcquisitionFactory.from_config({"kind": "random"}),
        batch_size=2,
        max_iter=max_iter,
        rng=np.random.default_rng(0),
    )
    return starting, batches


def run_many(points, *, acquisition=None, surrogate=None, seed=0, workers=1):
    return replay.replay_many(
        points,
        start=StartFactory.from_config(START),
        surrogate=surrogate or SurrogateFactory.from_config({"kind": "gp"}),
        acquisition=acquisition or AcquisitionFactory.from_config({"kind": "random"}),
        batch_size=2,
        max_iter=3,
        repeats=4,
        seed=seed,
        workers=workers,
    )


def test_replay_picks_new_explored_points_without_a_fit(points, recording_surrogate):
    starting, batches = run_one(points, recording_surrogate)

    picked = [point for batch in batches for point in batch]
    assert [len(batch) for batch in batches] == [2, 2, 2]
    assert len(set(picked) | set(starting)) == 8
    assert set(picked) <= set(points.pool.tolist())
    assert recording_surrogate.sizes == []


def test_replay_fits_a_model_based_rule_to_every_point_evaluated(
    points, recording_surrogate, first_choices
):
    run_one(points, recording_surrogate, acquisition=first_choices)

    assert recording_surrogate.sizes == [2, 4, 6]


def test_replay_stops_when_every_explored_point_is_evaluated(
    points, recording_surrogate
):
    _, batches = run_one(points, recording_surrogate, max_iter=50)

    assert sum(len(batch) for batch in batches) == len(points.pool) - 2


def test_replay_many_repeats_with_its_seed(points):
    assert run_many(points, seed=1).paths.equals(run_many(points, seed=1).paths)


def test_replay_many_starts_every_setting_from_the_same_sets(
    points, recording_surrogate, first_choices
):
    random = run_many(points).paths
    model = run_many(
        points, acquisition=first_choices, surrogate=recording_surrogate
    ).paths

    starts = random["batch"] == 0
    assert random.loc[starts].equals(model.loc[starts])
    assert not random.equals(model)


def test_replay_many_does_not_depend_on_the_workers(points):
    assert run_many(points, workers=1).paths.equals(run_many(points, workers=2).paths)


def test_hit_batches_give_the_batch_that_picked_each_target(made):
    replays = made([{0: 1, 1: 3}, {1: 2}])

    assert replays.hit_batches().tolist() == [[1.0, 3.0], [np.inf, 2.0]]


def test_table_lists_every_evaluation_with_its_energy_and_role(made, points):
    table = made([{0: 1}]).table()

    found = table.loc[table["formula"] == points.formulas[points.targets[0]]]
    assert len(table) == 8
    assert found[["batch", "target"]].values.tolist() == [[1, True]]


def test_from_table_rebuilds_the_replays_a_table_holds(made, points):
    replays = made([{0: 1, 1: 3}, {1: 2}])

    rebuilt = replay.Replays.from_table(
        points, replays.table(), batch_size=2, max_iter=3
    )

    assert rebuilt.paths.equals(replays.paths)


def test_from_table_refuses_formulas_of_another_field(made, points):
    table = made([{}]).table().assign(formula="Na1 Cl1")

    with pytest.raises(DataError, match="not points of this field"):
        replay.Replays.from_table(points, table, batch_size=2, max_iter=3)
