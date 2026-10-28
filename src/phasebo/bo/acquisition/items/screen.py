"""Screen: DFT in cheap order, the baseline that never learns from DFT."""

from typing import ClassVar

import numpy as np
from attrs import define

from phasebo.bo.acquisition.base import (
    Acquisition,
    AcquisitionConfig,
    AcquisitionFactory,
)
from phasebo.bo.surrogate.base import Fit


class ScreenConfig(AcquisitionConfig):
    """Screen settings."""


@AcquisitionFactory.register("screen")
@define
class ScreenAcquisition(Acquisition[ScreenConfig]):
    """Batch of the choices with the lowest cheap energies; ties broken at random."""

    needs_model: ClassVar[bool] = False
    needs_cheap: ClassVar[bool] = True

    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
        cheap_mev: np.ndarray | None = None,
    ) -> list[int]:
        if cheap_mev is None:
            msg = f"acquisition {self.kind!r} needs cheap energies"
            raise ValueError(msg)
        # Ties are common: every point on the cheap hull has exactly 0.
        tiebreak = rng.random(len(cheap_mev))
        return np.lexsort((tiebreak, cheap_mev))[:size].tolist()
