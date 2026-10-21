"""Comparisons: how benchmark runs differ from a baseline run, repeat by repeat."""

from collections.abc import Mapping

import numpy as np

from nuikit.errors import ConfigError

from phasebo.benchmark.metric.base import Contrast, Metric
from phasebo.benchmark.replay import Replays


def compare(
    base: Replays,
    runs: Mapping[str, Replays],
    *,
    metrics: Mapping[str, Metric],
    level: float,
    resamples: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, Contrast]]:
    """Return each run's contrasts with `base`, keyed `<metric>/<outcome>`.

    Every run must pass `check_paired`: repeat r of each starts from repeat r's set of
    `base`, so a comparison is paired, game by game.
    """
    return {
        name: {
            f"{kind}/{key}": contrast
            for kind, metric in metrics.items()
            for key, contrast in metric.contrast(
                base, replays, level=level, resamples=resamples, rng=rng
            ).items()
        }
        for name, replays in runs.items()
    }


def check_paired(base: Replays, run: Replays, *, name: str) -> None:
    """Raise `ConfigError` unless `run` replays `base`'s starting sets and budget."""
    if run.budget != base.budget:
        raise ConfigError(
            f"{name} picks {run.budget} times, the baseline {base.budget}; compare "
            "runs of one batch size and benchmark.max_iter"
        )
    if _name_starts(run) != _name_starts(base):
        raise ConfigError(
            f"{name} starts from other sets than the baseline; compare runs of one "
            "seed, benchmark.start and phase field"
        )


def _name_starts(replays: Replays) -> list[list[str]]:
    """Return each repeat's starting formulas, so runs on other fields differ."""
    return [
        [replays.points.formulas[i] for i in start] for start in replays.list_starts()
    ]
