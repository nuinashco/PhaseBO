"""Basin hopping walks PolyBO's path: same moves, energies and choices from one seed."""

import numpy as np
import pytest

from phasebo.csp import framework
from phasebo.csp.search.items.basin_hop import BasinHopConfig, BasinHopSearch

AL4N4 = {"Al": 4, "N": 4}
IONS = ({"Al": 4}, {"N": 4})


def search(epochs):
    return BasinHopSearch(BasinHopConfig(kind="basin_hop", epochs=epochs))


def test_basin_hop_follows_polybos_trace_from_the_same_seed(emt):
    cell = framework.build("wurtzite", AL4N4, phi=0.6, oxi={"Al": 3, "N": -3})

    found = search(6).search(cell, IONS, calculator=emt, rng=np.random.default_rng(42))

    trace = found.on_framework.trace
    # From PolyBO's basin_hop and release with EMT and default_rng(42).
    assert [hop.energy_ev for hop in trace] == pytest.approx(
        [0.854079, 0.929553, 0.485998, 0.758002, 0.894807, 0.909163, 1.089463],
        abs=1e-6,
    )
    assert [hop.accepted for hop in trace] == [True, False, True] + [False] * 4
    assert found.released_energy_ev == pytest.approx(-0.052645, abs=1e-6)


def test_basin_hop_keeps_the_lowest_arrangement_it_reached(emt):
    cell = framework.build("cubic", AL4N4, phi=0.6, oxi={"Al": 3, "N": -3})

    found = search(6).search(cell, IONS, calculator=emt, rng=np.random.default_rng(1))

    best = found.on_framework
    assert best.energy_ev == min(hop.energy_ev for hop in best.trace)
    assert best.trace[-1].best_ev == best.energy_ev


def test_basin_hop_relaxes_a_revisited_arrangement_once(emt):
    cell = framework.build("cubic", {"Al": 1, "N": 1}, phi=0.6, oxi={"Al": 3, "N": -3})

    found = search(12).search(
        cell, ({"Al": 1}, {"N": 1}), calculator=emt, rng=np.random.default_rng(0)
    )

    revisits = sum(hop.revisit for hop in found.on_framework.trace)
    assert revisits > 0
    assert found.on_framework.n_unique == 13 - revisits


def test_basin_hop_stops_when_no_move_changes_the_arrangement(emt):
    cell = framework.build("cubic", {"Al": 12, "N": 4}, phi=0.6, oxi={"Al": 3, "N": -3})

    found = search(5).search(
        cell, ({"Al": 12}, {"N": 4}), calculator=emt, rng=np.random.default_rng(0)
    )

    assert len(found.on_framework.trace) == 1


def test_basin_hop_names_its_level_by_its_epochs():
    assert search(30).level == "bh30"
