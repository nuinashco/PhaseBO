"""Cheap sources: an energy above the hull for every point, known before BO picks."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from attrs import define

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo.field.points import Points


class CheapConfig(KindConfig):
    """Settings of a cheap energy source."""


@define
class Cheap[C: CheapConfig](Configurable[C], ABC):
    """Source of each point's cheap energy, from the processed table's rows.

    `points` are the rows' points at the modelled `fidelity`; rows at other points,
    such as excluded ones, are left out.
    """

    @abstractmethod
    def check(self, rows: pd.DataFrame, points: Points, *, fidelity: str) -> None:
        """Raise `DataError` unless `rows` give every explored point a cheap energy."""

    @abstractmethod
    def draw(
        self,
        rows: pd.DataFrame,
        points: Points,
        *,
        fidelity: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return each point's cheap energy above the hull (meV/atom); NaN if none."""


class CheapFactory(ConfigurableFactory[Cheap[Any]]):
    """Registry of cheap sources by `kind`."""

    config_group = "benchmark/cheap"
