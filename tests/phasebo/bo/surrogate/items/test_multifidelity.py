"""The multi-fidelity GP predicts DFT's energies, not the cheap source's."""

import pytest

from phasebo.bo.surrogate.base import CheapEnergies, SurrogateFactory


def test_multifidelity_reproduces_the_dft_energies_it_was_fitted_to(points):
    pool, seen = points.pool, points.pool[:3]
    shifted = CheapEnergies(
        coords=points.coords[pool], energies_mev=points.energies_mev[pool] + 40.0
    )
    surrogate = SurrogateFactory.from_config({"kind": "multifidelity"})

    fit = surrogate.fit(points.coords[seen], points.energies_mev[seen], cheap=shifted)

    mean, _ = fit.predict(points.coords[seen])
    assert mean == pytest.approx(points.energies_mev[seen], abs=1.0)
