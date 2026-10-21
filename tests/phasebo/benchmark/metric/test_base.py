"""Scores turn into metrics.json and into W&B's summary and rows."""

import numpy as np
import pytest

from phasebo.benchmark.metric.base import Score, summarise, to_json, to_wandb

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


def test_to_wandb_leaves_undefined_curve_points_out():
    scores = {"a": Score(headline="", values={}, curves={"c": [None, 2.0]})}

    assert to_wandb(scores, [0, 4])[1] == [{}, {"a/c": 2.0}]
