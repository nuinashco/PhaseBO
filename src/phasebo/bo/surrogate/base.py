"""Surrogates: models of the energy above the hull over phase field coordinates."""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar, Literal

import numpy as np
from attrs import define, field, frozen

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

type KernelName = Literal["matern12", "matern32", "matern52", "rbf"]

MATERN_NU = {"matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


class SurrogateConfig(KindConfig):
    """Settings of a surrogate model."""


@frozen
class CheapEnergies:
    """Cheap energies above the hull of the points BO may pick, known before it does."""

    inputs: np.ndarray = field(eq=False)  # As `Surrogate.fit` takes them.
    energies_mev: np.ndarray = field(eq=False)


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
    """Model fitted to the evaluated points, predicting the rest.

    A surrogate with `needs_cheap` also fits the cheap energies and predicts DFT's;
    the others ignore them. One with `handles_categories` takes inputs whose
    `categories` columns are categories, such as a framework, not coordinates.
    """

    needs_cheap: ClassVar[bool] = False
    handles_categories: ClassVar[bool] = False

    @abstractmethod
    def fit(
        self,
        inputs: np.ndarray,
        energies_mev: np.ndarray,
        *,
        cheap: CheapEnergies | None = None,
        categories: Sequence[int] = (),
    ) -> Fit:
        """Return the model fitted to energies above the hull at `inputs` (meV/atom)."""


class SurrogateFactory(ConfigurableFactory[Surrogate[Any]]):
    """Registry of surrogates by `kind`."""

    config_group = "bo/surrogate"


def require_cheap[T](cheap: T | None, owner: str) -> T:
    """Return `cheap`; fail if `owner`, which needs cheap energies, was given none."""
    if cheap is None:
        msg = f"{owner} needs cheap energies"
        raise ValueError(msg)
    return cheap


def build_kernel(name: KernelName, dims: int) -> Any:
    """Return the kernel `name` over `dims` inputs; None for BoTorch's default."""
    nu = MATERN_NU.get(name)
    if nu is None:
        return None
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.priors import GammaPrior

    # BoTorch's Matern 5/2 priors before 0.12, the paper's; kept for every nu.
    return ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=dims, lengthscale_prior=GammaPrior(3.0, 6.0)),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )


def build_gp(
    x: Any,
    g: Any,
    *,
    kernel: KernelName,
    noise_mev: float | None,
    categories: Sequence[int] = (),
    input_transform: Any = None,
) -> Any:
    """Return an unfitted BoTorch GP of `g` over `x`, `categories` as categories."""
    with quiet_gpytorch():
        from botorch.models import MixedSingleTaskGP, SingleTaskGP

        variances = build_noise_variances(g, noise_mev)
        if not categories:
            return SingleTaskGP(
                x,
                g,
                train_Yvar=variances,
                covar_module=build_kernel(kernel, x.shape[1]),
                input_transform=input_transform,
            )
        return MixedSingleTaskGP(
            x,
            g,
            cat_dims=list(categories),
            train_Yvar=variances,
            cont_kernel_factory=_build_matern_factory(kernel),
            input_transform=input_transform,
        )


def _build_matern_factory(name: KernelName) -> Any:
    """Return MixedSingleTaskGP's kernel factory for `name`; None for BoTorch's own."""
    nu = MATERN_NU.get(name)
    if nu is None:
        return None

    def factory(batch_shape: Any, ard_num_dims: int, active_dims: list[int]) -> Any:
        from gpytorch.kernels import MaternKernel
        from gpytorch.priors import GammaPrior

        # BoTorch's pre-0.12 Matern 5/2 lengthscale prior, as in `build_kernel`;
        # MixedSingleTaskGP scales the kernel itself, with no outputscale prior
        # (BoTorch 0.18), and its category kernel's lengthscales have none either.
        return MaternKernel(
            nu=nu,
            ard_num_dims=ard_num_dims,
            active_dims=active_dims,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

    return factory


def build_noise_variances(g: Any, noise_mev: float | None) -> Any:
    """Return a variance per observation of `g`, or None for a fitted noise."""
    import torch

    return None if noise_mev is None else torch.full_like(g, noise_mev**2)


def stack_sources(
    inputs: np.ndarray, energies_mev: np.ndarray, *, cheap: CheapEnergies
) -> tuple[Any, Any]:
    """Return cheap then DFT inputs, with their source (0, 1) last, negated energies."""
    import torch

    def tag(rows: np.ndarray, source: float) -> Any:
        x = torch.as_tensor(rows, dtype=torch.float64)
        return torch.cat([x, torch.full((len(x), 1), source, dtype=x.dtype)], dim=1)

    x = torch.cat([tag(cheap.inputs, 0.0), tag(inputs, 1.0)])
    both_mev = np.concatenate([cheap.energies_mev, energies_mev])
    return x, -torch.as_tensor(both_mev, dtype=torch.float64)[:, None]


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
        # `linear_operator` builds sparse tensors for MultiTaskGP's task index.
        warnings.filterwarnings("ignore", message="Sparse invariant checks")
        yield


def fit_mll(model: Any) -> None:
    """Fit `model`'s hyperparameters to its exact marginal likelihood."""
    with quiet_gpytorch():
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
