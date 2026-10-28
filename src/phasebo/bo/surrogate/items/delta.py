"""Delta surrogate: the cheap energy plus a Gaussian process of its error."""

from collections.abc import Sequence
from typing import Any, ClassVar, override

import numpy as np
from attrs import define

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    Fit,
    Surrogate,
    SurrogateFactory,
    require_cheap,
)
from phasebo.bo.surrogate.items.gp import GPConfig, GPSurrogate


class DeltaConfig(GPConfig):
    """Delta settings: those of the GP of DFT minus cheap."""


@SurrogateFactory.register("delta")
@define
class DeltaSurrogate(Surrogate[DeltaConfig]):
    """Cheap energy plus a GP of DFT minus cheap; predicts only where cheap is known."""

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

        known = require_cheap(cheap, f"surrogate {self.kind!r}")
        settings = self.config.model_dump(exclude={"kind"})
        at = _find(torch.as_tensor(inputs), torch.as_tensor(known.inputs)).numpy()
        error = GPSurrogate(GPConfig(kind="gp", **settings)).fit(
            inputs, energies_mev - known.energies_mev[at], categories=categories
        )
        return Fit(
            model=_shift(error.model, known),
            best_f=float(-energies_mev.min()),
        )


def _find(rows: Any, inputs: Any) -> Any:
    """Return each of `rows`' index in `inputs` (tensors); fail on a row not in it."""
    matches = (rows.unsqueeze(-2) == inputs).all(dim=-1)
    if not bool(matches.any(dim=-1).all()):
        msg = "delta predicts only at points with a cheap energy"
        raise ValueError(msg)
    return matches.int().argmax(dim=-1)


def _shift(error: Any, known: CheapEnergies) -> Any:
    """Return a BoTorch model: `error`'s posterior plus the cheap value, negated."""
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal

    known_inputs = torch.as_tensor(known.inputs, dtype=torch.float64)
    offsets_mev = -torch.as_tensor(known.energies_mev, dtype=torch.float64)

    class Shifted(Model):
        @property
        @override
        def num_outputs(self) -> int:
            return 1

        @override
        def posterior(
            self,
            X: torch.Tensor,
            output_indices: list[int] | None = None,
            observation_noise: bool | torch.Tensor = False,
            posterior_transform: Any = None,
        ) -> GPyTorchPosterior:
            if posterior_transform is not None:
                msg = "delta's posterior takes no posterior_transform"
                raise NotImplementedError(msg)
            if output_indices not in (None, [0]):
                msg = "delta has one output; output_indices can only be [0]"
                raise NotImplementedError(msg)
            at = _find(X, known_inputs)
            base = error.posterior(X, observation_noise=observation_noise)
            normal = base.distribution
            return GPyTorchPosterior(
                MultivariateNormal(
                    normal.mean + offsets_mev[at], normal.lazy_covariance_matrix
                )
            )

    return Shifted()
