"""Steps to hit: evaluations until each target is picked, capped at the budget."""

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
    summarise,
)
from phasebo.benchmark.replay import Replays


class StepsToHitConfig(MetricConfig):
    """Steps-to-hit settings."""


@MetricFactory.register("steps_to_hit")
@define
class StepsToHitMetric(Metric[StepsToHitConfig]):
    """Mean evaluations until a target is picked; a miss counts as the budget.

    Reported per target, for the first of them and averaged over them; lower is
    better. A hit in batch b costs every evaluation up to that batch's end.
    """

    reading = (
        "Picks until the first target, and until each one, averaged over the games; a "
        "game that never finds it counts as the whole budget. Lower is better. It is "
        "blunt while most games find nothing, since they all count the same."
    )

    def measure(self, replays: Replays) -> Score:
        names = replays.target_names
        if not names:
            return Score(headline="steps to hit: no target to find", values={})
        budget = replays.budget
        steps = _cap_steps(replays)
        one = _random(replays, targets=1)
        first = _random(replays, targets=len(names))
        values: dict[str, float] = {}
        for name, column in zip(names, steps.T, strict=True):
            values |= summarise(name, column, random=one)
        values |= summarise("first_hit", steps.min(axis=1), random=first)
        values |= summarise("mean_over_targets", steps.mean(axis=1), random=one)
        headline = (
            f"first hit after {steps.min(axis=1).mean():.1f} evaluations on average "
            f"(random {first:.1f}; a miss counts as {budget})"
        )
        return Score(headline=headline, values=values)

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
        first_hit = pair(
            _cap_steps(base).min(axis=1),
            _cap_steps(run).min(axis=1),
            lower_is_better=True,
            level=level,
            resamples=resamples,
            rng=rng,
        )
        return {"first_hit": first_hit}


def _cap_steps(replays: Replays) -> np.ndarray:
    """Return evaluations to each target per repeat; a miss counts as the budget."""
    return np.minimum(replays.hit_batches() * replays.batch_size, replays.budget)


def _random(replays: Replays, *, targets: int) -> float:
    """Return random picking's capped mean evaluations to the first of `targets`."""
    q = replays.batch_size
    return float(
        sum(
            q * baseline.survival(replays.pool, b * q, targets=targets)
            for b in range(replays.max_iter)
        )
    )
