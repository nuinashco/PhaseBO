"""Batch log expected improvement, optimised over the candidates."""

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


class QLogEIConfig(AcquisitionConfig):
    """Batch log expected improvement settings."""


@AcquisitionFactory.register("qlogei")
@define
class QLogEIAcquisition(Acquisition[QLogEIConfig]):
    """Batch of distinct candidates with the highest joint log expected improvement."""

    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        import torch
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.optim import optimize_acqf_discrete

        model = require_fit(fit, self.kind)
        # Its default MC sampler seeds itself from torch's global generator (RUN-2
        # deviation, CONTRIBUTING).
        acqf = qLogExpectedImprovement(model.model, best_f=model.best_f)
        picked, _ = optimize_acqf_discrete(
            acqf, q=size, choices=torch.as_tensor(choices), unique=True
        )
        return locate(picked.detach().numpy(), choices)
