"""Random starting set: explored points off the hull, drawn uniformly."""

import numpy as np
from attrs import define
from pydantic import Field, PositiveInt

from nuikit.errors import ConfigError

from phasebo.benchmark.start.base import Start, StartConfig, StartFactory
from phasebo.field.points import Points


class RandomStartConfig(StartConfig):
    """Random starting set settings."""

    size: PositiveInt = Field(default=8, description="explored points, never a target")


@StartFactory.register("random")
@define
class RandomStart(Start[RandomStartConfig]):
    """Points drawn uniformly from the explored ones, leaving every target to find."""

    def check(self, points: Points, *, budget: int) -> None:
        size = self.config.size
        eligible = _eligible(points)
        if size > len(eligible):
            raise ConfigError(
                f"benchmark.start.size {size} exceeds the {len(eligible)} explored "
                "points off the hull; lower benchmark.start.size"
            )
        left = len(points.pool) - size
        if budget > left:
            raise ConfigError(
                f"bo.batch_size * benchmark.max_iter ({budget}) exceeds the {left} "
                "explored points left after the start; lower benchmark.max_iter"
            )

    def pick(self, points: Points, *, rng: np.random.Generator) -> list[int]:
        eligible = _eligible(points)
        return rng.choice(eligible, size=self.config.size, replace=False).tolist()


def _eligible(points: Points) -> np.ndarray:
    return np.setdiff1d(points.pool, points.targets)
