"""Thompson sampling: the best candidates of posterior samples."""

import numpy as np
from attrs import define

from phasebo.bo.acquisition.base import (
    Acquisition,
    AcquisitionConfig,
    AcquisitionFactory,
    locate,
    require_fit,
)
from phasebo.bo.surrogate.base import Fit


class TSConfig(AcquisitionConfig):
    """Thompson sampling settings."""


@AcquisitionFactory.register("ts")
@define
class TSAcquisition(Acquisition[TSConfig]):
    """Batch of distinct candidates, each the best of one posterior sample."""

    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        import torch
        from botorch.generation import MaxPosteriorSampling

        model = require_fit(fit, self.kind)
        # Samples come from torch's global generator (RUN-2 deviation, CONTRIBUTING).
        sample = MaxPosteriorSampling(model.model, replacement=False)
        with torch.no_grad():
            picked = sample(torch.as_tensor(choices), num_samples=size)
        return locate(picked.numpy(), choices)
