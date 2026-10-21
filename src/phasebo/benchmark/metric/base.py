"""Metrics: what replays achieved, next to random picking's exact value."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

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


@frozen
class Contrast:
    """How a run differs from a baseline run on one outcome, over paired repeats."""

    scale: Literal["difference", "ratio"]  # Run minus baseline, or run over baseline.
    estimate: float
    low: float
    high: float
    # Shares of repeats in which the run did better and worse; None for a pooled ratio.
    better: float | None = None
    worse: float | None = None


@define
class Metric[C: MetricConfig](Configurable[C], ABC):
    """Measure of how well replays found the targets, or low energies."""

    # Plain words on the metric's curves; the W&B view prints them above its panels.
    reading: ClassVar[str]

    @abstractmethod
    def measure(self, replays: Replays) -> Score:
        """Return this metric over every repeat of `replays`."""

    @abstractmethod
    def contrast(
        self,
        base: Replays,
        run: Replays,
        *,
        level: float,
        resamples: int,
        rng: np.random.Generator,
    ) -> dict[str, Contrast]:
        """Return how `run` differs from `base`, whose repeat r starts as its own."""


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


def pair(
    base: np.ndarray,
    run: np.ndarray,
    *,
    lower_is_better: bool,
    level: float,
    resamples: int,
    rng: np.random.Generator,
) -> Contrast:
    """Return the mean of `run - base` over paired repeats; bootstrap its interval."""
    from scipy import stats

    difference = run - base
    estimate = float(difference.mean())
    if np.all(difference == difference[0]):
        # A constant difference has no spread to resample.
        low = high = estimate
    else:
        interval = stats.bootstrap(
            (difference,),
            np.mean,
            n_resamples=resamples,
            confidence_level=level,
            rng=rng,
        ).confidence_interval
        low, high = float(interval.low), float(interval.high)
    ahead, behind = difference < 0, difference > 0
    if not lower_is_better:
        ahead, behind = behind, ahead
    return Contrast(
        scale="difference",
        estimate=estimate,
        low=low,
        high=high,
        better=float(ahead.mean()),
        worse=float(behind.mean()),
    )


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
    # Random picking's exact values stay in metrics.json; in W&B a run of the random
    # batch rule is the comparison.
    summary = {
        f"{kind}/{name}": value
        for kind, score in scores.items()
        for name, value in score.values.items()
        if not name.endswith("/random")
    }
    rows = [
        {
            f"{kind}/{name}": value
            for kind, score in scores.items()
            for name, curve in score.curves.items()
            if not name.endswith("/random") and (value := curve[step]) is not None
        }
        for step in range(len(evaluations))
    ]
    return summary, rows
