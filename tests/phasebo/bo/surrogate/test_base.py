"""Surrogates that need cheap energies refuse to fit without them and follow them."""

import numpy as np
import pytest

from phasebo.bo.surrogate.base import CheapEnergies, SurrogateFactory

NEED_CHEAP = ["multitask", "multifidelity", "delta"]
# Delta's own test asks more of it: removing a shared error.
RANKED = ["multitask", "multifidelity"]


def exact_cheap(points):
    pool = points.pool
    return CheapEnergies(
        coords=points.coords[pool], energies_mev=points.energies_mev[pool]
    )


@pytest.mark.parametrize("kind", NEED_CHEAP)
def test_a_surrogate_that_needs_cheap_energies_refuses_to_fit_without_them(
    points, kind
):
    surrogate = SurrogateFactory.from_config({"kind": kind})
    seen = points.pool[:3]

    with pytest.raises(ValueError, match="needs cheap energies"):
        surrogate.fit(points.coords[seen], points.energies_mev[seen])


@pytest.mark.parametrize("kind", RANKED)
def test_exact_cheap_energies_rank_the_unseen_points_for_every_surrogate(points, kind):
    surrogate = SurrogateFactory.from_config({"kind": kind})
    seen, unseen = points.pool[:3], points.pool[3:]

    fit = surrogate.fit(
        points.coords[seen], points.energies_mev[seen], cheap=exact_cheap(points)
    )

    mean, _ = fit.predict(points.coords[unseen])
    assert np.argmin(mean) == np.argmin(points.energies_mev[unseen])
