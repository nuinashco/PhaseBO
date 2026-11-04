"""Searches relax at fixed cell, push atoms off their sites, and free the cell last."""

import numpy as np
import pytest
from ase.build import bulk

from phasebo.csp import framework
from phasebo.csp.moves import Arrangement
from phasebo.csp.search.base import relax
from phasebo.csp.search.items.relax import RelaxConfig, RelaxSearch


def test_relax_keeps_the_cell_unless_it_is_freed(emt):
    squeezed = bulk("Al", "fcc", a=3.8)

    fixed, _ = relax(squeezed, emt, fmax=0.01, steps=100)
    freed, _ = relax(squeezed, emt, fmax=0.01, steps=100, cell=True)

    assert fixed.get_volume() == pytest.approx(squeezed.get_volume())
    assert freed.get_volume() > squeezed.get_volume()


def test_start_puts_every_ion_on_its_own_sublattice():
    cell = framework.build("cubic", {"Al": 3, "N": 3}, phi=0.6, oxi={"Al": 3, "N": -3})
    search = RelaxSearch(RelaxConfig(kind="relax"))

    start = search.start(cell, ({"Al": 3}, {"N": 3}), np.random.default_rng(0))

    assert (start.cations.count("Al"), start.anions.count("N")) == (3, 3)
    assert (len(start.cations), len(start.anions)) == (12, 4)


def test_score_pushes_the_atoms_by_the_rattle(emt):
    cell = framework.build("cubic", {"Al": 4, "N": 4}, phi=0.6, oxi={"Al": 3, "N": -3})
    arrangement = Arrangement(cations=("Al",) * 4 + (None,) * 8, anions=("N",) * 4)
    still = RelaxSearch(RelaxConfig(kind="relax", rattle_angstrom=0.0, steps=1))
    pushed = RelaxSearch(RelaxConfig(kind="relax", rattle_angstrom=0.3, steps=1))

    calm, _ = still.score(
        cell, arrangement, calculator=emt, rng=np.random.default_rng(0)
    )
    moved, _ = pushed.score(
        cell, arrangement, calculator=emt, rng=np.random.default_rng(0)
    )

    ideal = cell.place(arrangement).get_positions()
    assert np.abs(calm.positions - ideal).max() < np.abs(moved.positions - ideal).max()
