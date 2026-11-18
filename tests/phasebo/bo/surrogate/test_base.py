"""Surrogates: cheap energies when they need them, categories when they handle them."""

import numpy as np
import pytest

from nuikit.errors import ConfigError

from phasebo.bo.surrogate.base import (
    CheapEnergies,
    SurrogateFactory,
    build_gp,
    fit_mll,
)

NEED_CHEAP = ["multitask", "multifidelity", "delta", "joint"]


def exact_cheap(points):
    pool = points.pool
    return CheapEnergies(
        inputs=points.coords[pool], energies_mev=points.energies_mev[pool]
    )


@pytest.mark.parametrize("kind", NEED_CHEAP)
def test_a_surrogate_that_needs_cheap_energies_refuses_to_fit_without_them(
    points, kind
):
    surrogate = SurrogateFactory.from_config({"kind": kind})
    seen = points.pool[:3]

    with pytest.raises(ValueError, match="needs cheap energies"):
        surrogate.fit(points.coords[seen], points.energies_mev[seen])


@pytest.mark.parametrize("kind", NEED_CHEAP)
def test_exact_cheap_energies_rank_the_unseen_points_for_every_surrogate(points, kind):
    surrogate = SurrogateFactory.from_config({"kind": kind})
    seen, unseen = points.pool[:3], points.pool[3:]

    fit = surrogate.fit(
        points.coords[seen], points.energies_mev[seen], cheap=exact_cheap(points)
    )

    mean, _ = fit.predict(points.coords[unseen])
    assert np.argmin(mean) == np.argmin(points.energies_mev[unseen])


# BoTorch warns when a fit attempt fails, then retries from resampled priors.
@pytest.mark.filterwarnings("ignore::botorch.exceptions.warnings.OptimizationWarning")
@pytest.mark.parametrize("kind", ["gp", "delta", "joint"])
def test_a_surrogate_with_categories_tells_frameworks_of_one_composition_apart(
    framework_points, kind
):
    pool = framework_points.pool
    seen = pool
    inputs = framework_points.inputs
    # Errors that vary by point: with a constant one, DFT is exactly cheap plus a
    # constant, and joint's zero-noise fit fails for about a third of torch seeds.
    error_mev = np.random.default_rng(0).normal(scale=5.0, size=len(pool))
    cheap = CheapEnergies(
        inputs=inputs[pool],
        energies_mev=framework_points.energies_mev[pool] + error_mev,
    )
    surrogate = SurrogateFactory.from_config({"kind": kind})

    fit = surrogate.fit(
        inputs[seen],
        framework_points.energies_mev[seen],
        cheap=cheap,
        categories=framework_points.categories,
    )

    mean, _ = fit.predict(inputs[seen])
    assert mean == pytest.approx(framework_points.energies_mev[seen], abs=1.0)


@pytest.mark.parametrize("kind", ["multitask", "multifidelity"])
def test_a_surrogate_without_categories_refuses_them(framework_points, kind):
    pool = framework_points.pool
    inputs = framework_points.inputs
    cheap = CheapEnergies(
        inputs=inputs[pool], energies_mev=framework_points.energies_mev[pool]
    )
    surrogate = SurrogateFactory.from_config({"kind": kind})

    with pytest.raises(ValueError, match="cannot model categories"):
        surrogate.fit(
            inputs[pool],
            framework_points.energies_mev[pool],
            cheap=cheap,
            categories=framework_points.categories,
        )


def test_a_new_surrogate_refuses_categories_unless_it_declares_them(
    framework_points, recording_surrogate
):
    pool = framework_points.pool

    with pytest.raises(ValueError, match="cannot model categories"):
        recording_surrogate.fit(
            framework_points.inputs[pool],
            framework_points.energies_mev[pool],
            categories=framework_points.categories,
        )


def test_a_gp_with_categories_does_not_order_them():
    import torch

    rng = np.random.default_rng(0)
    coords, labels = rng.random((30, 1)), rng.integers(0, 3, 30)
    energies = np.sin(6 * coords[:, 0]) + np.array([0.0, 1.0, -0.5])[labels]
    unseen, unseen_labels = rng.random((10, 1)), rng.integers(0, 3, 10)

    def predict(relabel):
        x = torch.as_tensor(np.column_stack([coords, relabel[labels]]))
        model = build_gp(
            x,
            torch.as_tensor(energies)[:, None],
            kernel="matern52",
            noise_mev=None,
            categories=[1],
        )
        fit_mll(model)
        at = torch.as_tensor(np.column_stack([unseen, relabel[unseen_labels]]))
        with torch.no_grad():
            return model.posterior(at).mean.numpy().ravel()

    # Swapping labels 0 and 1 changes distances between ordered numbers, not labels.
    swapped = predict(np.array([1, 0, 2]))
    assert swapped == pytest.approx(predict(np.arange(3)), abs=1e-6)


def failing_with(monkeypatch, error):
    def fit(*_args, **_kwargs):
        raise error

    monkeypatch.setattr("botorch.fit.fit_gpytorch_mll", fit)


def test_a_fit_that_fails_asks_for_a_noise_above_zero(points, monkeypatch):
    from botorch.exceptions.errors import ModelFittingError

    failing_with(monkeypatch, ModelFittingError("All attempts failed."))
    surrogate = SurrogateFactory.from_config({"kind": "gp"})
    seen = points.pool[:3]

    with pytest.raises(ConfigError, match=r"set bo\.surrogate\.noise_mev above 0"):
        surrogate.fit(points.coords[seen], points.energies_mev[seen])


def test_a_covariance_that_is_not_positive_definite_asks_for_a_noise_above_zero(
    points, monkeypatch
):
    from linear_operator.utils.errors import NotPSDError

    failing_with(monkeypatch, NotPSDError("Matrix not positive definite."))
    surrogate = SurrogateFactory.from_config({"kind": "gp"})
    seen = points.pool[:3]

    with pytest.raises(ConfigError, match=r"set bo\.surrogate\.noise_mev above 0"):
        surrogate.fit(points.coords[seen], points.energies_mev[seen])
