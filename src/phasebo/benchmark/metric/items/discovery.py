"""Discovery curves: the share of repeats that found each target, batch by batch."""

import numpy as np
from attrs import define

from phasebo.benchmark import baseline
from phasebo.benchmark.metric.base import (
    Contrast,
    Metric,
    MetricConfig,
    MetricFactory,
    Score,
    pair,
)
from phasebo.benchmark.replay import Replays


class DiscoveryConfig(MetricConfig):
    """Discovery-curve settings."""


@MetricFactory.register("discovery")
@define
class DiscoveryMetric(Metric[DiscoveryConfig]):
    """Share of repeats that found each target, and any target, after each batch."""

    reading = (
        "The share of games that have found each target (one panel per target), or "
        "any target (`first_hit`), after each number of picks. Higher and earlier is "
        "better: a setting beats random picking where its line is above the `random` "
        "run's."
    )

    def measure(self, replays: Replays) -> Score:
        names = replays.target_names
        if not names:
            return Score(headline="discovery: no target to find", values={})
        evaluations = replays.evaluations
        steps = replays.hit_batches() * replays.batch_size
        pool = replays.pool
        one = [1 - baseline.survival(pool, int(x)) for x in evaluations]
        curves: dict[str, list[float | None]] = {}
        for name, column in zip(names, steps.T, strict=True):
            curves[name] = _found(column, evaluations)
            curves[f"{name}/random"] = list(one)
        curves["first_hit"] = _found(steps.min(axis=1), evaluations)
        curves["first_hit/random"] = [
            1 - baseline.survival(pool, int(x), targets=len(names)) for x in evaluations
        ]
        found = _count_found(replays)
        random = len(names) * one[-1]
        headline = (
            f"{found.mean():.2f} of {len(names)} targets found within "
            f"{replays.budget} evaluations on average (random {random:.2f})"
        )
        values: dict[str, float] = {
            "found/mean": float(found.mean()),
            "found/random": random,
        }
        return Score(headline=headline, values=values, curves=curves)

    def contrast(
        self,
        base: Replays,
        run: Replays,
        *,
        level: float,
        resamples: int,
        rng: np.random.Generator,
    ) -> dict[str, Contrast]:
        if not base.target_names:
            return {}
        found = pair(
            _count_found(base),
            _count_found(run),
            lower_is_better=False,
            level=level,
            resamples=resamples,
            rng=rng,
        )
        return {"found": found}


def _count_found(replays: Replays) -> np.ndarray:
    """Return the targets each repeat found within the budget."""
    steps = replays.hit_batches() * replays.batch_size
    return (steps <= replays.budget).sum(axis=1).astype(float)


def _found(steps: np.ndarray, evaluations: np.ndarray) -> list[float | None]:
    return [float((steps <= x).mean()) for x in evaluations]
