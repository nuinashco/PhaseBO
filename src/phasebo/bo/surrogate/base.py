"""Surrogates: models of the energy above the hull over phase field coordinates."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from attrs import define, frozen

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig


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
