"""A GP fitted to half of the synthetic field, and the other half to pick from."""

import numpy as np
import pytest

from phasebo.bo.surrogate.base import Fit, SurrogateFactory
from phasebo.field.points import Points


@pytest.fixture
def split(points: Points) -> tuple[Fit, np.ndarray]:
    """Return a GP fitted to every other explored point, and the rest as choices."""
    pool = points.pool
    seen, choices = pool[::2], pool[1::2]
    fit = SurrogateFactory.from_config({"kind": "gp"}).fit(
        points.coords[seen], points.energies_mev[seen]
    )
    return fit, points.coords[choices]
