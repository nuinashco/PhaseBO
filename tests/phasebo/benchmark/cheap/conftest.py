"""The synthetic field's rows, and rows of a second fidelity at some of its points."""

from collections.abc import Callable, Sequence

import pandas as pd
import pytest

from phasebo.config import SystemConfig
from phasebo.data import store, table
from phasebo.field.points import Points

type NNRows = Callable[[Sequence[int], float], pd.DataFrame]


@pytest.fixture
def rows(system: SystemConfig) -> pd.DataFrame:
    """Return the field's processed rows, all at the dft fidelity."""
    return store.load(system.processed).rows


@pytest.fixture
def nn_rows(points: Points) -> NNRows:
    """Return a maker of nn rows: energies above the hull, dft's plus a shift."""

    def make(indices: Sequence[int], shift_mev: float) -> pd.DataFrame:
        made = table.make_rows(
            [points.formulas[i] for i in indices],
            fidelity="nn",
            reference=points.references[list(indices)].tolist(),
            e_above_hull_mev=(points.energies_mev[list(indices)] + shift_mev).tolist(),
        )
        cleaned, _ = table.clean(made, elements=points.elements, source="nn.csv")
        return cleaned

    return make
