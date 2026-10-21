"""Hazard ratio: BO's chance per batch of picking a target, over random's."""

import numpy as np
from attrs import define
from pydantic import Field

from phasebo.benchmark import baseline
from phasebo.benchmark.metric.base import (
    Contrast,
    Metric,
    MetricConfig,
    MetricFactory,
    Score,
)
from phasebo.benchmark.replay import Replays


class HazardRatioConfig(MetricConfig):
    """Hazard-ratio settings."""

    level: float = Field(default=0.95, gt=0, lt=1, description="of the interval")


@MetricFactory.register("hazard_ratio")
@define
class HazardRatioMetric(Metric[HazardRatioConfig]):
    """Theta of a grouped proportional-hazards fit against random (Prentice, Gloeckler).

    Each batch in which a target is still unfound is a trial: random's chance h0 of
    finding it is exact, and BO's is 1 - (1 - h0) ** theta. Batches random could not
    miss carry no information and are left out. The interval treats the repeat as
    the unit, since a repeat's targets share its starting set.

    Curves follow the run batch by batch: `hit_chance`, the share of unfound targets
    each batch picked, and `per_batch`, the theta of that batch alone, which shows
    whether the advantage holds early, late or throughout.
    """

    reading = (
        "`hit_chance`: of the targets not found yet, the share each batch picked. "
        "`per_batch`: the same as a multiple of random picking's chance, so 1 is "
        "random and 2 twice as good. Read the trend, not single batches: one batch "
        "swings by chance, as the `random` run's own line shows."
    )

    def measure(self, replays: Replays) -> Score:
        hits = replays.hit_batches()
        curves = _measure_per_batch(replays, hits)
        repeat, hit, h0 = _trials(replays, hits)
        if not hit.any():
            return Score(
                headline="hazard ratio: no target found in a batch random could miss",
                values={"theta": 0.0, "theta/random": 1.0},
                curves=curves,
            )
        if hit.all():
            return Score(
                headline="hazard ratio: every target found as soon as it could be; "
                "theta is unbounded",
                values={"theta/random": 1.0},
                curves=curves,
            )
        import statsmodels.api as sm

        model = sm.GLM(
            hit,
            np.ones((len(hit), 1)),
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            offset=np.log(-np.log1p(-h0)),
        )
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": repeat})
        theta = float(np.exp(fit.params[0]))
        low, high = np.exp(fit.conf_int(alpha=1 - self.config.level)[0])
        headline = (
            f"{theta:.2f} times random's chance per batch of picking a target "
            f"({self.config.level:.0%} interval {low:.2f} to {high:.2f})"
        )
        values: dict[str, float] = {
            "theta": theta,
            "theta/random": 1.0,
            "low": float(low),
            "high": float(high),
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
        """Return the run's hazard as a multiple of the baseline's, from one joint fit.

        Both runs' trials share one model with an indicator for the run's; a repeat's
        trials in both runs form one cluster, since they share a starting set.
        """
        trials = [_trials(replays, replays.hit_batches()) for replays in (base, run)]
        hits = [hit for _, hit, _ in trials]
        if not all(0 < hit.sum() < len(hit) for hit in hits):
            return {}
        import statsmodels.api as sm

        repeat, hit, h0 = (np.concatenate(parts) for parts in zip(*trials, strict=True))
        is_run = np.repeat([0.0, 1.0], [len(hits[0]), len(hits[1])])
        model = sm.GLM(
            hit,
            np.column_stack([np.ones(len(hit)), is_run]),
            family=sm.families.Binomial(link=sm.families.links.CLogLog()),
            offset=np.log(-np.log1p(-h0)),
        )
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": repeat})
        low, high = np.exp(fit.conf_int(alpha=1 - level)[1])
        ratio = Contrast(
            scale="ratio",
            estimate=float(np.exp(fit.params[1])),
            low=float(low),
            high=float(high),
        )
        return {"theta": ratio}


def _measure_per_batch(
    replays: Replays, hits: np.ndarray
) -> dict[str, list[float | None]]:
    """Return each batch's hit chance, random's, and the theta they imply."""
    q = replays.batch_size
    curves: dict[str, list[float | None]] = {
        "hit_chance": [None],
        "hit_chance/random": [None],
        "per_batch": [None],
        "per_batch/random": [None],
    }
    for batch in range(1, replays.max_iter + 1):
        searching = int((hits >= batch).sum())
        chance = float((hits == batch).sum() / searching) if searching else None
        h0 = baseline.hazard(replays.pool, (batch - 1) * q, batch=q)
        theta = None
        if chance is not None and chance < 1 and h0 < 1:
            # One batch alone solves 1 - (1 - h0) ** theta = chance in closed form.
            theta = float(np.log1p(-chance) / np.log1p(-h0))
        curves["hit_chance"].append(chance)
        curves["hit_chance/random"].append(h0)
        curves["per_batch"].append(theta)
        curves["per_batch/random"].append(1.0 if h0 < 1 else None)
    return curves


def _trials(
    replays: Replays, hits: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return each trial's repeat, whether it hit, and random's chance h0 of a hit."""
    q = replays.batch_size
    repeats, outcomes, chances = [], [], []
    for batch in range(1, replays.max_iter + 1):
        h0 = baseline.hazard(replays.pool, (batch - 1) * q, batch=q)
        if h0 >= 1:
            break
        repeat, target = np.nonzero(hits >= batch)
        repeats.append(repeat)
        outcomes.append(hits[repeat, target] == batch)
        chances.append(np.full(len(repeat), h0))
    if not repeats:
        return np.empty(0, dtype=int), np.empty(0), np.empty(0)
    return (
        np.concatenate(repeats),
        np.concatenate(outcomes).astype(float),
        np.concatenate(chances),
    )
