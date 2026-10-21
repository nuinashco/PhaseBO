"""Random starting set: explored points off the hull, drawn uniformly."""

import numpy as np
from attrs import define
from pydantic import Field, PositiveInt

from nuikit.errors import ConfigError

from phasebo.bo.start.base import Start, StartConfig, StartFactory
from phasebo.field.points import Points


class RandomStartConfig(StartConfig):
    """Random starting set settings."""

    size: PositiveInt = Field(default=23, description="explored points, never a target")


@StartFactory.register("random")
@define
class RandomStart(Start[RandomStartConfig]):
    """Points drawn uniformly from the explored ones, leaving every target to find."""

    def check(self, points: Points) -> None:
        eligible = _eligible(points)
        if self.config.size > len(eligible):
            raise ConfigError(
                f"bo.start.size {self.config.size} exceeds the {len(eligible)} "
                "explored points off the hull; lower bo.start.size"
            )

    def pick(self, points: Points, *, rng: np.random.Generator) -> list[int]:
        self.check(points)
        eligible = _eligible(points)
        return rng.choice(eligible, size=self.config.size, replace=False).tolist()


def _eligible(points: Points) -> np.ndarray:
    return np.setdiff1d(points.pool, points.targets)
