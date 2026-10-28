"""The table source reads another fidelity of the processed table."""

import numpy as np
import pandas as pd
import pytest

from nuikit.errors import DataError

from phasebo.benchmark.cheap.base import CheapFactory


def test_table_source_gives_each_point_its_energy_at_the_other_fidelity(
    rows, nn_rows, points
):
    every = range(len(points.formulas))
    both = pd.concat([rows, nn_rows(every, 7.0)], ignore_index=True)
    source = CheapFactory.from_config({"kind": "table", "fidelity": "nn"})

    energies = source.draw(both, points, fidelity="dft", rng=np.random.default_rng(0))

    assert energies == pytest.approx(points.energies_mev + 7.0)


def test_table_source_refuses_a_fidelity_that_misses_explored_points(
    rows, nn_rows, points
):
    # A processed table holds references, so the nn rows keep them.
    kept = [*points.pool[::2], *np.flatnonzero(points.references)]
    half = pd.concat([rows, nn_rows(kept, 0.0)], ignore_index=True)
    source = CheapFactory.from_config({"kind": "table", "fidelity": "nn"})

    with pytest.raises(DataError, match=r"choose a benchmark\.cheap\.fidelity"):
        source.check(half, points, fidelity="dft")
