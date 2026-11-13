"""Single-task Gaussian process from BoTorch."""

from collections.abc import Sequence
from typing import ClassVar

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
    build_gp,
    fit_mll,
)


class GPConfig(SurrogateConfig):
    """Gaussian process settings."""

    kernel: KernelName = Field(
        default="matern52",
        description="matern12|matern32|matern52|rbf; matern52: the paper's; "
        "rbf: BoTorch's default kernel",
    )
    noise_mev: NonNegativeFloat | None = Field(
        default=0.0, description="std in meV/atom; null: fitted"
    )


@SurrogateFactory.register("gp")
@define
class GPSurrogate(Surrogate[GPConfig]):
    """Gaussian process on standardised energies, with a fixed or fitted noise.

    With categories, BoTorch's MixedSingleTaskGP adds a category kernel to the
    coordinates' one.
    """

    handles_categories: ClassVar[bool] = True

    def fit(
        self,
        inputs: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
        categories: Sequence[int] = (),
    ) -> Fit:
        import torch

        x = torch.as_tensor(inputs, dtype=torch.float64)
        g = -torch.as_tensor(energies_mev, dtype=torch.float64)[:, None]
        model = build_gp(
            x,
            g,
            kernel=self.config.kernel,
            noise_mev=self.config.noise_mev,
            categories=categories,
        )
        fit_mll(model)
        return Fit(model=model, best_f=float(g.max()))
