"""Random acquisition: the baseline that ignores the surrogate."""

from typing import ClassVar

import numpy as np
from attrs import define

from phasebo.bo.acquisition.base import (
    Acquisition,
    AcquisitionConfig,
    AcquisitionFactory,
)
from phasebo.bo.surrogate.base import Fit


class RandomAcquisitionConfig(AcquisitionConfig):
    """Random acquisition settings."""


@AcquisitionFactory.register("random")
@define
class RandomAcquisition(Acquisition[RandomAcquisitionConfig]):
    """Batch of distinct candidates drawn uniformly."""

    needs_model: ClassVar[bool] = False

    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
        cheap_mev: np.ndarray | None = None,
    ) -> list[int]:
        return rng.choice(len(choices), size=size, replace=False).tolist()
