"""Map: how right the believed phase diagram is, batch by batch."""

from typing import ClassVar

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.benchmark.metric.base import (
    Contrast,
    Metric,
    MetricConfig,
    MetricFactory,
    Score,
    pair,
)
from phasebo.benchmark.replay import Replays


class MapConfig(MetricConfig):
    """Map settings."""

    tolerance_mev: NonNegativeFloat = Field(
        default=10.0,
        description="meV/atom above the hull within which a composition counts as "
        "stable, truly or by belief at even odds",
    )


@MetricFactory.register("map")
@define
class MapMetric(Metric[MapConfig]):
    """Share of compositions whose winning framework is believed right; stable F1."""

    reading = (
        "After each number of picks: the share of compositions whose winning framework "
        "the run names right (`winner`), and how well the compositions it believes "
        "stable match the true ones (`stable_f1`, 1 is perfect). Higher is better. A "
        "rule that fits no model believes the cheap energies; without either it has no "
        "map."
    )
    needs_beliefs: ClassVar[bool] = True

    def measure(self, replays: Replays) -> Score:
        if replays.beliefs is None:
            return Score(
                headline="map: no beliefs; a rule needs a model or a cheap source",
                values={},
            )
        winner, f1 = self._score(replays)
        groups = _group(replays)
        has_frameworks = groups.shape[1] > 1
        stable_share = float(self._find_true_stable(replays, groups).mean())
        curves: dict[str, list[float | None]] = {
            "stable_f1": f1.mean(axis=0).tolist(),
            "stable_f1/random": [stable_share] * f1.shape[1],
        }
        values = {
            "stable_f1/mean": float(f1[:, -1].mean()),
            "stable_f1/random": stable_share,
        }
        headline = f"stable-set F1 {f1[:, -1].mean():.2f}"
        if has_frameworks:
            guess = 1 / groups.shape[1]
            curves["winner"] = winner.mean(axis=0).tolist()
            curves["winner/random"] = [guess] * winner.shape[1]
            values |= {
                "winner/mean": float(winner[:, -1].mean()),
                "winner/random": guess,
            }
            headline = (
                f"right framework for {winner[:, -1].mean():.0%} of compositions, "
                f"{headline} (random {guess:.0%}, {stable_share:.2f})"
            )
        else:
            headline = f"{headline} (random {stable_share:.2f})"
        headline += f" after {replays.budget} evaluations"
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
        if base.beliefs is None or run.beliefs is None:
            return {}
        (base_winner, base_f1), (run_winner, run_f1) = (
            self._score(base),
            self._score(run),
        )
        contrasts = {
            "stable_f1": pair(
                base_f1[:, -1],
                run_f1[:, -1],
                lower_is_better=False,
                level=level,
                resamples=resamples,
                rng=rng,
            )
        }
        if _group(base).shape[1] > 1:
            contrasts["winner"] = pair(
                base_winner[:, -1],
                run_winner[:, -1],
                lower_is_better=False,
                level=level,
                resamples=resamples,
                rng=rng,
            )
        return contrasts

    def _score(self, replays: Replays) -> tuple[np.ndarray, np.ndarray]:
        """Return winner accuracy and stable F1 per repeat and step."""
        from scipy.stats import norm

        beliefs = replays.beliefs
        if beliefs is None:
            msg = "these replays recorded no beliefs"
            raise ValueError(msg)
        groups = _group(replays)
        mean_mev = np.where(groups >= 0, beliefs[..., groups, 0], np.inf)
        std_mev = np.where(groups >= 0, beliefs[..., groups, 1], 0.0)
        true_winner = _find_true_winners(replays, groups)
        winner = (np.argmin(mean_mev, axis=-1) == true_winner).mean(axis=-1)
        tolerance_mev = self.config.tolerance_mev
        with np.errstate(divide="ignore", invalid="ignore"):
            below = np.where(
                std_mev > 0,
                norm.cdf((tolerance_mev - mean_mev) / std_mev),
                mean_mev <= tolerance_mev,
            )
        # Believed stable: even odds or better that one of its points lies within
        # `tolerance_mev` of the hull, each point's energy normal with its mean and std.
        believed = 1 - np.prod(1 - below, axis=-1) >= 0.5
        truth = self._find_true_stable(replays, groups)
        hits = (believed & truth).sum(axis=-1)
        wrong = (believed & ~truth).sum(axis=-1) + (~believed & truth).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(2 * hits + wrong > 0, 2 * hits / (2 * hits + wrong), 1.0)
        return winner, f1

    def _find_true_stable(self, replays: Replays, groups: np.ndarray) -> np.ndarray:
        """Return whether each composition's lowest point is within the tolerance."""
        energies_mev = replays.points.energies_mev[replays.points.pool]
        lowest_mev = np.where(groups >= 0, energies_mev[groups], np.inf).min(axis=-1)
        return lowest_mev <= self.config.tolerance_mev


def _group(replays: Replays) -> np.ndarray:
    """Return, per explored composition, its points' positions in the pool; -1 pads."""
    pool = replays.points.pool
    names = np.asarray(replays.points.names)[pool]
    order: dict[str, list[int]] = {}
    for position, name in enumerate(names):
        order.setdefault(str(name), []).append(position)
    width = max(len(members) for members in order.values())
    return np.array(
        [members + [-1] * (width - len(members)) for members in order.values()]
    )


def _find_true_winners(replays: Replays, groups: np.ndarray) -> np.ndarray:
    """Return the column of each composition's lowest point in `groups`."""
    energies_mev = replays.points.energies_mev[replays.points.pool]
    return np.argmin(np.where(groups >= 0, energies_mev[groups], np.inf), axis=-1)
