"""Joint surrogate: one Gaussian process of both sources, the source as a category."""

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
from attrs import define
from pydantic import Field, NonNegativeFloat

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    Fit,
    Surrogate,
    SurrogateFactory,
    build_gp,
    fit_mll,
    require_cheap,
    stack_sources,
)
from phasebo.bo.surrogate.items.gp import GPConfig


class JointConfig(GPConfig):
    """Joint Gaussian process settings."""

    noise_mev: NonNegativeFloat | None = Field(
        default=0.0, description="std in meV/atom, shared by both sources; null: fitted"
    )


@SurrogateFactory.register("joint")
@define
class JointSurrogate(Surrogate[JointConfig]):
    """MixedSingleTaskGP over cheap and DFT energies, the source a category; DFT's."""

    needs_cheap: ClassVar[bool] = True
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
        from botorch.models.transforms.input import AppendFeatures

        x, g = stack_sources(
            inputs, energies_mev, cheap=require_cheap(cheap, f"surrogate {self.kind!r}")
        )
        model = build_gp(
            x,
            g,
            kernel=self.config.kernel,
            noise_mev=self.config.noise_mev,
            categories=[*categories, x.shape[1] - 1],
            # Predictions take inputs alone and get source 1, DFT.
            input_transform=AppendFeatures(
                feature_set=torch.ones(1, 1, dtype=x.dtype), transform_on_train=False
            ),
        )
        fit_mll(model)
        return Fit(model=model, best_f=float(-energies_mev.min()))
