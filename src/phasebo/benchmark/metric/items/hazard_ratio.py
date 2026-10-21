"""Hazard ratio: BO's chance per batch of picking a target, over random's."""

import numpy as np
from attrs import define
from pydantic import Field

from phasebo.benchmark import baseline
from phasebo.benchmark.metric.base import Metric, MetricConfig, MetricFactory, Score
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
    """

    def measure(self, replays: Replays) -> Score:
        repeat, hit, h0 = _trials(replays, replays.hit_batches())
        if not hit.any():
            return Score(
                headline="hazard ratio: no target found in a batch random could miss",
                values={"theta": 0.0},
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
            "low": float(low),
            "high": float(high),
        }
        return Score(headline=headline, values=values)


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
