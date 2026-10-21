"""Single-task Gaussian process from BoTorch."""

import warnings
from typing import Any, Literal

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.bo.surrogate.base import Fit, Surrogate, SurrogateConfig, SurrogateFactory


class GPConfig(SurrogateConfig):
    """Gaussian process settings."""

    kernel: Literal["matern", "rbf"] = Field(
        default="matern", description="matern: Matern 5/2, as in the paper; rbf"
    )
    noise_mev: NonNegativeFloat | None = Field(
        default=0.0, description="std in meV/atom; null: fitted"
    )


@SurrogateFactory.register("gp")
@define
class GPSurrogate(Surrogate[GPConfig]):
    """Gaussian process on standardised energies, with a fixed or fitted noise."""

    def fit(self, coords: np.ndarray, energies_mev: np.ndarray) -> Fit:
        import torch

        with warnings.catch_warnings():
            # `linear_operator`, a gpytorch dependency, warns on import under 3.14.
            warnings.filterwarnings(
                "ignore", message="`torch.jit.script` is not", category=FutureWarning
            )
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
            from gpytorch.utils.warnings import NumericalWarning

        x = torch.as_tensor(coords, dtype=torch.float64)
        g = -torch.as_tensor(energies_mev, dtype=torch.float64)[:, None]
        noise_mev = self.config.noise_mev
        g_var = None if noise_mev is None else torch.full_like(g, noise_mev**2)
        with warnings.catch_warnings():
            # `gpytorch` raises a zero noise to its 1e-6 floor (standardised units).
            warnings.filterwarnings(
                "ignore",
                message="Very small noise values detected",
                category=NumericalWarning,
            )
            model = SingleTaskGP(
                x, g, train_Yvar=g_var, covar_module=self._build_kernel(x.shape[1])
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        return Fit(model=model, best_f=float(g.max()))

    def _build_kernel(self, dims: int) -> Any:
        if self.config.kernel == "rbf":
            return None  # BoTorch's default.
        from gpytorch.kernels import MaternKernel, ScaleKernel
        from gpytorch.priors import GammaPrior

        # The priors of BoTorch's Matern default before 0.12.
        return ScaleKernel(
            MaternKernel(
                nu=2.5, ard_num_dims=dims, lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
