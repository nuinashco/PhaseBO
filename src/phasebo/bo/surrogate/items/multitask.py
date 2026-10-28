"""Multi-task Gaussian process from BoTorch: cheap and DFT energies as two tasks."""

from typing import ClassVar

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    Fit,
    Surrogate,
    SurrogateFactory,
    build_kernel,
    build_noise_variances,
    fit_mll,
    quiet_gpytorch,
    require_cheap,
    stack_sources,
)
from phasebo.bo.surrogate.items.gp import GPConfig


class MultiTaskConfig(GPConfig):
    """Multi-task Gaussian process settings."""

    noise_mev: NonNegativeFloat | None = Field(
        default=0.0,
        description="std in meV/atom, of both sources; null: fitted per source",
    )


@SurrogateFactory.register("multitask")
@define
class MultiTaskSurrogate(Surrogate[MultiTaskConfig]):
    """Gaussian process of both sources that learns their correlation; predicts DFT."""

    needs_cheap: ClassVar[bool] = True

    def fit(
        self,
        coords: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
    ) -> Fit:
        known = require_cheap(cheap, self.kind)
        x, g = stack_sources(coords, energies_mev, cheap=known)
        with quiet_gpytorch():
            from botorch.models import MultiTaskGP

            model = MultiTaskGP(
                x,
                g,
                task_feature=-1,
                output_tasks=[1],
                train_Yvar=build_noise_variances(g, self.config.noise_mev),
                covar_module=build_kernel(self.config.kernel, x.shape[1] - 1),
            )
        fit_mll(model)
        return Fit(model=model, best_f=float(-energies_mev.min()))
