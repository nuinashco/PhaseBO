"""Benchmark replay: BO picks among the computed points, as if computing them."""

import numpy as np
import pandas as pd

from phasebo.bo.acquisition.base import Acquisition
from phasebo.bo.start.base import Start
from phasebo.bo.surrogate.base import Surrogate
from phasebo.field.points import Points


def replay(
    points: Points,
    *,
    start: Start,
    surrogate: Surrogate,
    acquisition: Acquisition,
    batch_size: int,
    max_iter: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return the evaluation order: the starting points, then up to `max_iter` batches.

    Each batch comes from the explored points not evaluated yet, sorted by
    coordinates; the replay ends early when none are left.
    """
    evaluated = start.pick(points, rng=rng)
    batches = [0] * len(evaluated)
    for batch in range(1, max_iter + 1):
        done = set(evaluated)
        left = sorted(
            (int(i) for i in points.pool if i not in done),
            key=lambda i: tuple(points.coords[i]),
        )
        if not left:
            break
        fit = None
        if acquisition.needs_model:
            fit = surrogate.fit(
                points.coords[evaluated], points.energies_mev[evaluated]
            )
        size = min(batch_size, len(left))
        picked = acquisition.pick(fit, points.coords[left], size=size, rng=rng)
        evaluated += [left[i] for i in picked]
        batches += [batch] * len(picked)
    targets = set(points.targets.tolist())
    return pd.DataFrame(
        {
            "formula": [points.formulas[i] for i in evaluated],
            "e_above_hull_mev": points.energies_mev[evaluated],
            "batch": batches,
            "target": [i in targets for i in evaluated],
        }
    )
