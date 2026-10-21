"""Proposals: the next compositions to compute, picked from candidate formulas."""

from collections.abc import Collection, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from nuikit.errors import DataError

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
    it, so every cell gets a prediction.
    """
    coords = points.locate(cells)
    pool = points.pool
    fit = surrogate.fit(points.coords[pool], points.energies_mev[pool])
    size = min(batch_size, len(cells))
    picked = acquisition.pick(fit, coords, size=size, rng=rng)
    mean_mev, std_mev = fit.predict(coords)
    posterior = pd.DataFrame(
        {"formula": cells, "mean_mev": mean_mev.round(1), "std_mev": std_mev.round(1)}
    ).sort_values("mean_mev", kind="stable", ignore_index=True)
    return [cells[i] for i in picked], posterior


def select_cells(
    points: Points, candidates: Sequence[str], *, excluded: Collection[str] = ()
) -> list[str]:
    """Return one cell per candidate point not computed or excluded, by coordinates.

    A point keeps its smallest candidate cell, the cheapest to compute.
    """
    taken = {
        Composition(f).reduced_composition.formula
        for f in [*points.formulas, *excluded]
    }
    smallest: dict[str, str] = {}
    for formula in candidates:
        composition = Composition(formula)
        point = composition.reduced_composition.formula
        if point in taken:
            continue
        kept = smallest.get(point)
        if kept is None or composition.num_atoms < Composition(kept).num_atoms:
            smallest[point] = formula
    if not smallest:
        raise DataError(
            "every candidate is at a computed or excluded point; give other "
            "candidates in system.candidates.file, or set it to null"
        )
    cells = list(smallest.values())
    coords = points.locate(cells)
    return [cells[i] for i in sorted(range(len(cells)), key=lambda i: tuple(coords[i]))]
