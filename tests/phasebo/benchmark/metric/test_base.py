"""Scores turn into metrics.json and into W&B's summary and rows."""

import numpy as np
import pytest

from phasebo.benchmark.metric.base import (
    MetricFactory,
    Score,
    pair,
    summarise,
    to_json,
    to_wandb,
)

SCORES = {"a": Score(headline="a line", values={"x": 1.0}, curves={"c": [0.0, 1.0]})}


def test_summarise_gives_the_mean_its_standard_error_and_random():
    values = summarise("t", np.array([2.0, 4.0]), random=3.5)

    assert values == {"t/mean": 3.0, "t/random": 3.5, "t/se": pytest.approx(1.0)}


def test_summarise_leaves_out_the_standard_error_of_one_sample():
    assert "t/se" not in summarise("t", np.array([2.0]), random=3.5)


def test_to_json_keeps_each_score_under_its_kind():
    assert to_json(SCORES, [0, 4]) == {
        "evaluations": [0, 4],
        "a": {"headline": "a line", "values": {"x": 1.0}, "curves": {"c": [0.0, 1.0]}},
    }


def test_to_wandb_summarises_values_and_logs_one_row_per_evaluation():
    summary, rows = to_wandb(SCORES, [0, 4])

    assert (summary, rows) == ({"a/x": 1.0}, [{"a/c": 0.0}, {"a/c": 1.0}])


def test_to_wandb_leaves_random_pickings_exact_values_out():
    scores = {
        "a": Score(
            headline="",
            values={"x": 1.0, "x/random": 0.5},
            curves={"c": [1.0], "c/random": [0.5]},
        )
    }

    assert to_wandb(scores, [0]) == ({"a/x": 1.0}, [{"a/c": 1.0}])


def test_to_wandb_leaves_undefined_curve_points_out():
    scores = {"a": Score(headline="", values={}, curves={"c": [None, 2.0]})}

    assert to_wandb(scores, [0, 4])[1] == [{}, {"a/c": 2.0}]


def test_every_metric_says_how_to_read_it():
    kinds = MetricFactory.list_registered()

    assert all(MetricFactory.get_class(kind).reading for kind in kinds)


def test_pair_bootstraps_the_mean_difference_and_counts_who_won():
    contrast = pair(
        np.array([3.0, 3.0, 3.0, 3.0]),
        np.array([1.0, 2.0, 3.0, 5.0]),
        lower_is_better=True,
        level=0.9,
        resamples=2000,
        rng=np.random.default_rng(0),
    )

    assert (contrast.estimate, contrast.better, contrast.worse) == (-0.25, 0.5, 0.25)
    assert contrast.low < contrast.estimate < contrast.high


def test_pair_of_a_constant_difference_has_no_spread():
    contrast = pair(
        np.zeros(3),
        np.ones(3),
        lower_is_better=False,
        level=0.95,
        resamples=100,
        rng=np.random.default_rng(0),
    )

    assert (contrast.low, contrast.estimate, contrast.high) == (1.0, 1.0, 1.0)
