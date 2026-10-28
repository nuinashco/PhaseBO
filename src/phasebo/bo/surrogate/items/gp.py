"""Single-task Gaussian process from BoTorch."""

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    Fit,
    KernelName,
    Surrogate,
    SurrogateConfig,
    SurrogateFactory,
    build_kernel,
    build_noise_variances,
    fit_mll,
    quiet_gpytorch,
)


class GPConfig(SurrogateConfig):
    """Gaussian process settings."""

    kernel: KernelName = Field(
        default="matern52",
        description="matern12|matern32|matern52: Matern nu 1/2, 3/2, 5/2 (the paper); "
        "rbf: BoTorch's default kernel",
    )
    noise_mev: NonNegativeFloat | None = Field(
        default=0.0, description="std in meV/atom; null: fitted"
    )


@SurrogateFactory.register("gp")
@define
class GPSurrogate(Surrogate[GPConfig]):
    """Gaussian process on standardised energies, with a fixed or fitted noise."""

    def fit(
        self,
        coords: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
    ) -> Fit:
        import torch

        x = torch.as_tensor(coords, dtype=torch.float64)
        g = -torch.as_tensor(energies_mev, dtype=torch.float64)[:, None]
        with quiet_gpytorch():
            from botorch.models import SingleTaskGP

            model = SingleTaskGP(
                x,
                g,
                train_Yvar=build_noise_variances(g, self.config.noise_mev),
                covar_module=build_kernel(self.config.kernel, x.shape[1]),
            )
        fit_mll(model)
        return Fit(model=model, best_f=float(g.max()))
