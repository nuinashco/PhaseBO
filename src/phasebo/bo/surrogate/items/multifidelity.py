"""Multi-fidelity Gaussian process from BoTorch: the source is a fidelity input."""

from typing import ClassVar

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    Fit,
    Surrogate,
    SurrogateConfig,
    SurrogateFactory,
    build_noise_variances,
    fit_mll,
    quiet_gpytorch,
    require_cheap,
    stack_sources,
)


class MultiFidelityConfig(SurrogateConfig):
    """Multi-fidelity Gaussian process settings."""

    noise_mev: NonNegativeFloat | None = Field(
        default=0.0, description="std in meV/atom, shared by both sources; null: fitted"
    )


@SurrogateFactory.register("multifidelity")
@define
class MultiFidelitySurrogate(Surrogate[MultiFidelityConfig]):
    """Gaussian process with BoTorch's fidelity kernel (Matern 5/2); predicts DFT."""

    needs_cheap: ClassVar[bool] = True

    def fit(
        self,
        coords: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
    ) -> Fit:
        import torch

        known = require_cheap(cheap, self.kind)
        x, g = stack_sources(coords, energies_mev, cheap=known)
        with quiet_gpytorch():
            from botorch.models import SingleTaskMultiFidelityGP
            from botorch.models.transforms.input import AppendFeatures

            model = SingleTaskMultiFidelityGP(
                x,
                g,
                train_Yvar=build_noise_variances(g, self.config.noise_mev),
                data_fidelities=[x.shape[1] - 1],
                # Predictions take coordinates alone and get fidelity 1, DFT.
                input_transform=AppendFeatures(
                    feature_set=torch.ones(1, 1, dtype=x.dtype),
                    transform_on_train=False,
                ),
            )
        fit_mll(model)
        return Fit(model=model, best_f=float(-energies_mev.min()))
