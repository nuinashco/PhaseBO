"""Surrogates: cheap energies when they need them, categories when they handle them."""

import numpy as np
import pytest

from phasebo.bo.surrogate.base import CheapEnergies, SurrogateFactory

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
    # Errors that vary by point: a constant one makes joint's zero-noise covariance
    # singular, since DFT is then exactly cheap plus a constant.
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
