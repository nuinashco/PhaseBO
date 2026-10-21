"""Best so far: the lowest energy above the hull known after each batch."""

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


class BestSoFarConfig(MetricConfig):
    """Best-so-far settings."""


@MetricFactory.register("best_so_far")
@define
class BestSoFarMetric(Metric[BestSoFarConfig]):
    """Mean lowest energy known after each batch, starting set included (meV/atom).

    Unlike the target metrics it needs no point on the hull, so it also reads on a
    field whose targets are not among the computed points.
    """

    reading = (
        "The lowest energy above the hull known after each number of picks, the "
        "starting set included, averaged over the games (meV/atom; 0 is stable). Lower "
        "is better: a line under the `random` run's closes in on the hull faster."
    )

    def measure(self, replays: Replays) -> Score:
        energies_mev = replays.points.energies_mev
        pool = set(replays.points.pool.tolist())
        evaluations = replays.evaluations
        best_mev = _track_best(replays)
        expected_mev = np.empty_like(best_mev)
        for row, start in enumerate(replays.list_starts()):
            floor_mev = float(energies_mev[list(start)].min())
            left_mev = energies_mev[sorted(pool - set(start))]
            for done, picks in enumerate(evaluations):
                expected_mev[row, done] = baseline.expected_min(
                    left_mev, int(picks), floor_mev=floor_mev
                )
        mean_mev, random_mev = best_mev.mean(axis=0), expected_mev.mean(axis=0)
        headline = (
            f"lowest energy known after {replays.budget} evaluations: "
            f"{mean_mev[-1]:.1f} meV/atom on average (random {random_mev[-1]:.1f})"
        )
        return Score(
            headline=headline,
            values=summarise(
                "at_budget", best_mev[:, -1], random=float(random_mev[-1])
            ),
            curves={"mean": mean_mev.tolist(), "mean/random": random_mev.tolist()},
        )

    def contrast(
        self,
        base: Replays,
        run: Replays,
        *,
        level: float,
        resamples: int,
        rng: np.random.Generator,
    ) -> dict[str, Contrast]:
        at_budget = pair(
            _track_best(base)[:, -1],
            _track_best(run)[:, -1],
            lower_is_better=True,
            level=level,
            resamples=resamples,
            rng=rng,
        )
        return {"at_budget": at_budget}


def _track_best(replays: Replays) -> np.ndarray:
    """Return each repeat's lowest energy known after each batch (meV/atom)."""
    energies_mev = replays.points.energies_mev
    best_mev = np.empty((replays.repeats, replays.max_iter + 1))
    # Repeats are numbered from 0, and groupby yields them in order.
    for row, (_, path) in enumerate(replays.paths.groupby("repeat")):
        for done in range(replays.max_iter + 1):
            known = path.loc[path["batch"] <= done, "point"].to_numpy()
            best_mev[row, done] = energies_mev[known].min()
    return best_mev
