"""Relax is basin hopping without moves: one arrangement, relaxed, then released."""

import numpy as np
import pytest

from phasebo.csp import framework
from phasebo.csp.search.items.relax import RelaxConfig, RelaxSearch


def test_relax_matches_polybos_search_with_no_epochs(emt):
    cell = framework.build(
        "wurtzite", {"Al": 4, "N": 4}, phi=0.6, oxi={"Al": 3, "N": -3}
    )
    search = RelaxSearch(RelaxConfig(kind="relax"))

    found = search.search(
        cell, ({"Al": 4}, {"N": 4}), calculator=emt, rng=np.random.default_rng(42)
    )

    # From PolyBO's basin_hop with epochs=0, then its release, with EMT.
    assert (found.on_framework.energy_ev, found.released_energy_ev) == pytest.approx(
        (0.854079, -0.057958), abs=1e-6
    )
    assert (len(found.on_framework.trace), found.on_framework.n_unique) == (1, 1)


def test_relax_keeps_the_framework_cell_until_the_release(emt):
    cell = framework.build("cubic", {"Al": 4, "N": 4}, phi=0.6, oxi={"Al": 3, "N": -3})
    search = RelaxSearch(RelaxConfig(kind="relax"))

    found = search.search(
        cell, ({"Al": 4}, {"N": 4}), calculator=emt, rng=np.random.default_rng(0)
    )

    assert np.allclose(found.on_framework.atoms.cell.array, cell.cell)
    assert not np.allclose(found.released.cell.array, cell.cell)
