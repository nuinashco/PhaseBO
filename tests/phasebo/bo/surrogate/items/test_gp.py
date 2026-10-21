"""The Gaussian process interpolates noise-free energies and fits noisy ones."""

import numpy as np
import pytest

from phasebo.bo.surrogate.base import SurrogateFactory


def test_zero_noise_gp_reproduces_its_training_energies(points):
    pool = points.pool
    gp = SurrogateFactory.from_config(
        {"kind": "gp", "kernel": "matern", "noise_mev": 0.0}
    )

    fit = gp.fit(points.coords[pool], points.energies_mev[pool])

    mean, _ = fit.predict(points.coords[pool])
    assert mean == pytest.approx(points.energies_mev[pool], abs=0.5)


def test_fitted_noise_rbf_gp_predicts_every_point_with_some_spread(points):
    pool = points.pool
    gp = SurrogateFactory.from_config(
        {"kind": "gp", "kernel": "rbf", "noise_mev": None}
    )

    fit = gp.fit(points.coords[pool], points.energies_mev[pool])

    mean, std = fit.predict(points.coords)
    assert (mean.shape, bool((std > 0).all())) == ((len(points.formulas),), True)
    assert fit.best_f == pytest.approx(-np.min(points.energies_mev[pool]))
