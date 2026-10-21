"""Starting sets: the points a benchmark replay evaluates before BO picks any."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from attrs import define

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo.field.points import Points


class StartConfig(KindConfig):
    """Settings of a starting set."""


@define
class Start[C: StartConfig](Configurable[C], ABC):
    """Rule that picks the points a replay starts from."""

    @abstractmethod
    def check(self, points: Points, *, budget: int) -> None:
        """Raise `ConfigError` unless `points` hold this set and `budget` picks more."""

    @abstractmethod
    def pick(self, points: Points, *, rng: np.random.Generator) -> list[int]:
        """Return the indices of the starting points."""


class StartFactory(ConfigurableFactory[Start[Any]]):
    """Registry of starting-set rules by `kind`."""

    config_group = "benchmark/start"
