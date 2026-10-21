"""Metrics: what replays achieved, next to random picking's exact value."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np
from attrs import define, field, frozen

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo.benchmark.replay import Replays


class MetricConfig(KindConfig):
    """Settings of a benchmark metric."""


@frozen
class Score:
    """What a metric measured: a line to print, named values, and curves.

    Each curve has one value per entry of `Replays.evaluations`, None where it is
    undefined, and a curve `x` comes with `x/random`, random picking's exact value.
    """

    headline: str
    values: dict[str, float]
    curves: dict[str, list[float | None]] = field(factory=dict)


@define
class Metric[C: MetricConfig](Configurable[C], ABC):
    """Measure of how well replays found the targets, or low energies."""

    @abstractmethod
    def measure(self, replays: Replays) -> Score:
        """Return this metric over every repeat of `replays`."""


class MetricFactory(ConfigurableFactory[Metric[Any]]):
    """Registry of benchmark metrics by `kind`; a run computes several."""

    config_group = "benchmark/metric"
    multiple = True


def summarise(name: str, samples: np.ndarray, *, random: float) -> dict[str, float]:
    """Return the mean of `samples`, its standard error if any, and random's value."""
    values = {f"{name}/mean": float(samples.mean()), f"{name}/random": random}
    if len(samples) > 1:
        values[f"{name}/se"] = float(samples.std(ddof=1) / np.sqrt(len(samples)))
    return values


def to_json(scores: Mapping[str, Score], evaluations: list[int]) -> dict[str, Any]:
    """Return metrics.json: the evaluation counts curves follow, then each score."""
    return {
        "evaluations": evaluations,
        **{
            kind: {
                "headline": score.headline,
                "values": score.values,
                "curves": score.curves,
            }
            for kind, score in scores.items()
        },
    }


def to_wandb(
    scores: Mapping[str, Score], evaluations: list[int]
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Return W&B's summary values and one row of curve points per evaluation count."""
    summary = {
        f"{kind}/{name}": value
        for kind, score in scores.items()
        for name, value in score.values.items()
    }
    rows = [
        {
            f"{kind}/{name}": value
            for kind, score in scores.items()
            for name, curve in score.curves.items()
            if (value := curve[step]) is not None
        }
        for step in range(len(evaluations))
    ]
    return summary, rows
