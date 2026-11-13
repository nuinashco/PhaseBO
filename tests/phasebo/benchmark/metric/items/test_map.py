"""The map scores believed framework winners and believed stable compositions."""

import numpy as np
import pandas as pd
import pytest

from phasebo.benchmark.metric.base import MetricFactory
from phasebo.benchmark.replay import Replays
from phasebo.field.points import Points

# Compositions the field's cubic framework wins.
CUBIC_WINNERS = {"Li3 S1 Cl1", "Li4 S1 Cl2", "Li5 S2 Cl1"}


def believing(points, means, stds=None):
    """Return replays of one game whose beliefs, at every step, are `means`."""
    steps = 2
    stds = np.zeros_like(means) if stds is None else stds
    one = np.column_stack([means, stds]).astype(np.float32)
    beliefs = np.stack([one] * steps)[None]
    paths = pd.DataFrame({"repeat": [0], "batch": [0], "point": [int(points.pool[0])]})
    return Replays(
        points=points, batch_size=2, max_iter=1, paths=paths, beliefs=beliefs
    )


def dropping_wurtzite(rows, points, compositions):
    """Return the points of `rows` with `compositions` left in cubic alone."""
    dropped = rows["point"].isin(compositions) & (rows["framework"] == "wurtzite")
    return Points.from_rows(
        rows[~dropped].reset_index(drop=True),
        fidelity="dft",
        elements=points.elements,
    )


def measure(replays, **settings):
    return MetricFactory.from_config({"kind": "map", **settings}).measure(replays)


def test_beliefs_equal_to_the_truth_make_a_perfect_map(framework_points):
    truth = framework_points.energies_mev[framework_points.pool]

    values = measure(believing(framework_points, truth)).values

    assert (values["winner/mean"], values["stable_f1/mean"]) == (1.0, 1.0)


def test_believing_the_wrong_framework_lowest_misses_that_winner(framework_points):
    pool = framework_points.pool
    means = framework_points.energies_mev[pool].copy()
    winner = int(np.flatnonzero(pool == framework_points.winners[0])[0])
    means[winner] += 1000.0

    values = measure(believing(framework_points, means)).values

    compositions = len(framework_points.winners)
    assert values["winner/mean"] == pytest.approx(1 - 1 / compositions)


def test_even_odds_of_lying_within_the_tolerance_count_as_stable(framework_points):
    pool = framework_points.pool
    means = np.full(len(pool), 500.0)
    stds = np.zeros(len(pool))
    target = int(np.flatnonzero(pool == framework_points.targets[0])[0])
    means[target], stds[target] = 10.0, 30.0

    believed = believing(framework_points, means, stds)

    values = measure(believed, tolerance_mev=10.0).values

    lowest = framework_points.energies_mev[framework_points.winners]
    stable = int((lowest <= 10.0).sum())
    assert values["stable_f1/mean"] == pytest.approx(2 / (stable + 1))


def test_replays_without_beliefs_have_no_map(made):
    assert measure(made([{}])).values == {}


def test_map_contrast_pairs_winner_accuracy_game_by_game(framework_points):
    truth = framework_points.energies_mev[framework_points.pool]
    flat = np.zeros_like(truth)
    metric = MetricFactory.from_config({"kind": "map"})

    contrasts = metric.contrast(
        believing(framework_points, flat),
        believing(framework_points, truth),
        level=0.95,
        resamples=100,
        rng=np.random.default_rng(0),
    )

    assert contrasts["winner"].estimate > 0


def test_compositions_in_one_framework_are_left_out_of_the_winner_score(
    framework_rows, framework_points
):
    points = dropping_wurtzite(framework_rows, framework_points, CUBIC_WINNERS)
    means = points.energies_mev[points.pool].copy()
    contested = 0
    for winner in points.winners:
        position = int(np.flatnonzero(points.pool == winner)[0])
        if points.names.count(points.names[winner]) > 1:
            means[position] += 1000.0
            contested += 1

    values = measure(believing(points, means)).values

    assert contested > 0
    assert values["winner/mean"] == 0.0


def test_winner_bar_is_the_most_common_winners_share(framework_rows, framework_points):
    points = dropping_wurtzite(framework_rows, framework_points, CUBIC_WINNERS)
    truth = points.energies_mev[points.pool]

    score = measure(believing(points, truth))

    # Ten compositions keep both frameworks; wurtzite wins seven of them.
    assert score.values["winner/random"] == pytest.approx(7 / 10)
    assert "always wurtzite 70%" in score.headline


def test_a_field_without_two_frameworks_anywhere_has_no_winner_score(
    framework_rows, framework_points
):
    points = dropping_wurtzite(
        framework_rows, framework_points, set(framework_points.names)
    )
    truth = points.energies_mev[points.pool]

    values = measure(believing(points, truth)).values

    assert "winner/mean" not in values
    assert "stable_f1/mean" in values
