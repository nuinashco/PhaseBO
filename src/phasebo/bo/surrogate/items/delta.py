"""Delta surrogate: the cheap energy plus a Gaussian process of its error."""

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

    def fit(
        self,
        coords: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
    ) -> Fit:
        known = require_cheap(cheap, self.kind)
        settings = self.config.model_dump(exclude={"kind"})
        error = GPSurrogate(GPConfig(kind="gp", **settings)).fit(
            coords, energies_mev - known.energies_mev[_find(coords, known.coords)]
        )
        return Fit(
            model=_shift(error.model, known),
            best_f=float(-energies_mev.min()),
        )


def _find(rows: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Return the index in `coords` of each of `rows`; fail on a row not in it."""
    matches = (rows[:, None, :] == coords[None, :, :]).all(axis=-1)
    if not matches.any(axis=1).all():
        msg = "delta predicts only at points with a cheap energy"
        raise ValueError(msg)
    return matches.argmax(axis=1)


def _shift(error: Any, known: CheapEnergies) -> Any:
    """Return a BoTorch model: `error`'s posterior plus the cheap value, negated."""
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultivariateNormal

    coords = torch.as_tensor(known.coords, dtype=torch.float64)
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
            matches = (X.unsqueeze(-2) == coords).all(dim=-1)
            if not bool(matches.any(dim=-1).all()):
                msg = "delta predicts only at points with a cheap energy"
                raise ValueError(msg)
            base = error.posterior(X, observation_noise=observation_noise)
            normal = base.distribution
            return GPyTorchPosterior(
                MultivariateNormal(
                    normal.mean + offsets_mev[matches.int().argmax(dim=-1)],
                    normal.lazy_covariance_matrix,
                )
            )

    return Shifted()
