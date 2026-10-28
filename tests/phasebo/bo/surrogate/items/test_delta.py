"""Delta learns the cheap source's error from the DFT points it was fitted to."""

import pytest

from phasebo.bo.surrogate.base import CheapEnergies, SurrogateFactory


def shifted_cheap(points, shift_mev):
    pool = points.pool
    return CheapEnergies(
        coords=points.coords[pool], energies_mev=points.energies_mev[pool] + shift_mev
    )


def test_delta_removes_an_error_the_dft_points_share(points):
    seen, unseen = points.pool[:3], points.pool[3:]
    delta = SurrogateFactory.from_config({"kind": "delta"})

    fit = delta.fit(
        points.coords[seen], points.energies_mev[seen], cheap=shifted_cheap(points, 30)
    )

    mean, _ = fit.predict(points.coords[unseen])
    assert mean == pytest.approx(points.energies_mev[unseen], abs=1.0)


def test_delta_refuses_to_predict_where_no_cheap_energy_is_known(points):
    seen = points.pool[:3]
    delta = SurrogateFactory.from_config({"kind": "delta"})
    fit = delta.fit(
        points.coords[seen], points.energies_mev[seen], cheap=shifted_cheap(points, 0)
    )

    with pytest.raises(ValueError, match="only at points with a cheap energy"):
        fit.predict(points.coords[points.references])
