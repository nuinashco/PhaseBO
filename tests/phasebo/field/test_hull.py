"""Energies above the hull: zero on it, the distance to it elsewhere."""

import pytest

from phasebo.field import hull


def test_stable_cell_is_on_the_hull_and_its_double_cell_above(totals):
    formulas = list(totals)

    energies_mev = hull.measure(formulas, list(totals.values()))

    above = dict(zip(formulas, energies_mev, strict=True))
    assert (above["Li3 S1 Cl1"], above["Li6 S2 Cl2"]) == (
        pytest.approx(0.0, abs=1e-6),
        pytest.approx(30.0),
    )
