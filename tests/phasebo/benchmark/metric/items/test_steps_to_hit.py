"""Steps to hit count the evaluations to each target, a miss as the budget."""

import pytest

from phasebo.benchmark.metric.base import MetricFactory


def measure(replays):
    return MetricFactory.from_config({"kind": "steps_to_hit"}).measure(replays)


def test_steps_to_hit_charge_the_whole_batch_and_a_miss_the_budget(made):
    replays = made([{0: 1, 1: 3}, {1: 2}])
    first, second = replays.target_names

    values = measure(replays).values

    assert (values[f"{first}/mean"], values[f"{second}/mean"]) == (4.0, 5.0)
    assert (values["first_hit/mean"], values["mean_over_targets/mean"]) == (3.0, 4.5)


def test_steps_to_hit_compare_with_random_pickings_exact_mean(made):
    replays = made([{0: 1}])
    first, _ = replays.target_names

    values = measure(replays).values

    # 11 points are left after the start; x picks miss one with chance (11 - x) / 11.
    assert values[f"{first}/random"] == pytest.approx(2 * (1 + 9 / 11 + 7 / 11))


def test_steps_to_hit_of_random_picking_converge_on_random(random_replays):
    values = measure(random_replays).values

    for name in ("first_hit", "mean_over_targets"):
        gap = abs(values[f"{name}/mean"] - values[f"{name}/random"])
        assert gap < 4 * values[f"{name}/se"]
