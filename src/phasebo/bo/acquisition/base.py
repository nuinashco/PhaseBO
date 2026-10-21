"""Acquisitions: rules that pick the next batch from the candidates."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from attrs import define

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo.bo.surrogate.base import Fit


class AcquisitionConfig(KindConfig):
    """Settings of a batch-picking rule."""


@define
class Acquisition[C: AcquisitionConfig](Configurable[C], ABC):
    """Rule that picks the next batch, from a fitted surrogate or without one.

    A rule with `needs_model` False ignores the surrogate, so a replay does not fit it.
    """

    needs_model: ClassVar[bool] = True

    @abstractmethod
    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        """Return the indices of `size` distinct `choices` to evaluate next."""


class AcquisitionFactory(ConfigurableFactory[Acquisition[Any]]):
    """Registry of acquisition rules by `kind`."""

    config_group = "bo/acquisition"


def require_fit(fit: Fit | None, kind: str) -> Fit:
    """Return `fit`; fail if a model-based rule was called without one."""
    if fit is None:
        msg = f"acquisition {kind!r} needs a fitted surrogate"
        raise ValueError(msg)
    return fit


def locate(picked: np.ndarray, choices: np.ndarray) -> list[int]:
    """Return the index in `choices` of each row of `picked`."""
    return [int(np.flatnonzero((choices == row).all(axis=1))[0]) for row in picked]
