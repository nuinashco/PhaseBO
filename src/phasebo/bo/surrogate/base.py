"""Surrogates: models of the energy above the hull over phase field coordinates."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal

import numpy as np
from attrs import define, frozen

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

type KernelName = Literal["matern12", "matern32", "matern52", "rbf"]

_MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


class SurrogateConfig(KindConfig):
    """Settings of a surrogate model."""


@frozen
class Fit:
    """A surrogate fitted to observed energies.

    BoTorch maximises, so `model` predicts the negated energy and `best_f` is the
    highest value of it observed.
    """

    model: Any  # A fitted BoTorch model; torch is imported lazily.
    best_f: float

    def predict(self, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the predicted energy above the hull and its std (meV/atom).

        The std includes the observation noise.
        """
        import torch

        with torch.no_grad():
            posterior = self.model.posterior(
                torch.as_tensor(coords, dtype=torch.float64), observation_noise=True
            )
            mean_mev = -posterior.mean.numpy().ravel()
            return mean_mev, posterior.variance.sqrt().numpy().ravel()


@define
class Surrogate[C: SurrogateConfig](Configurable[C], ABC):
    """Model fitted to the evaluated points, predicting the rest."""

    @abstractmethod
    def fit(self, coords: np.ndarray, energies_mev: np.ndarray) -> Fit:
        """Return the model fitted to energies above the hull at `coords` (meV/atom)."""


class SurrogateFactory(ConfigurableFactory[Surrogate[Any]]):
    """Registry of surrogates by `kind`."""

    config_group = "bo/surrogate"


def build_kernel(name: KernelName, dims: int) -> Any:
    """Return the kernel `name` over `dims` inputs; None for BoTorch's default."""
    nu = _MATERN_NU.get(name)
    if nu is None:
        return None
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

    # BoTorch's Matern 5/2 priors before 0.12, the paper's; kept for every nu.
    return ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=dims, lengthscale_prior=GammaPrior(3.0, 6.0)),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


def build_noise_variances(g: Any, noise_mev: float | None) -> Any:
    """Return a variance per observation of `g`, or None for a fitted noise."""
    import torch

    return None if noise_mev is None else torch.full_like(g, noise_mev**2)


@contextmanager
def quiet_gpytorch() -> Iterator[None]:
    """Silence gpytorch's warnings that say nothing about the fit."""
    with warnings.catch_warnings():
        # `linear_operator`, a gpytorch dependency, warns on import under 3.14.
        warnings.filterwarnings(
            "ignore", message="`torch.jit.script` is not", category=FutureWarning
        )
        # `gpytorch` raises a zero noise to its 1e-6 floor (standardised units).
        warnings.filterwarnings("ignore", message="Very small noise values detected")
        yield


def fit_mll(model: Any) -> None:
    """Fit `model`'s hyperparameters to its exact marginal likelihood."""
    with quiet_gpytorch():
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
