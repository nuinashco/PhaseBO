"""Proposals: the next compositions to compute, picked from candidate cells."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from phasebo.bo.acquisition.base import Acquisition
from phasebo.bo.surrogate.base import Surrogate
from phasebo.field.points import Points


def propose(
    points: Points,
    cells: Sequence[str],
    *,
    surrogate: Surrogate,
    acquisition: Acquisition,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[list[str], pd.DataFrame]:
    """Return the next batch among `cells` and the prediction at each.

    The surrogate is fitted to the explored points even when the acquisition ignores
    it, so every cell gets a prediction. Cells are ranked in coordinate order, so the
    batch does not depend on the order they come in.
    """
    unsorted = points.locate(cells)
    order = sorted(range(len(cells)), key=lambda i: tuple(unsorted[i]))
    cells, coords = [cells[i] for i in order], unsorted[order]
    pool = points.pool
    fit = surrogate.fit(points.coords[pool], points.energies_mev[pool])
    size = min(batch_size, len(cells))
    picked = acquisition.pick(fit, coords, size=size, rng=rng)
    mean_mev, std_mev = fit.predict(coords)
    posterior = pd.DataFrame(
        {"formula": cells, "mean_mev": mean_mev.round(1), "std_mev": std_mev.round(1)}
    ).sort_values("mean_mev", kind="stable", ignore_index=True)
    return [cells[i] for i in picked], posterior
