"""Points: one per reduced composition, its lowest cell, its role in the search."""

import numpy as np
import pytest
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry

from nuikit.errors import DataError

from phasebo.data import store, table
from phasebo.field import points as field_points


def test_from_rows_keeps_the_lowest_cell_of_each_point(points):
    assert ("Li3 S1 Cl1" in points.formulas, "Li6 S2 Cl2" in points.formulas) == (
        True,
        False,
    )


def test_targets_are_the_explored_points_on_the_hull(points):
    targets = [points.formulas[i] for i in points.targets]

    assert "Li3 S1 Cl1" in targets
    assert not points.references[points.targets].any()


def test_pool_leaves_out_the_references(points):
    assert set(points.pool) == set(np.flatnonzero(~points.references))
    assert "Li1" not in [points.formulas[i] for i in points.pool]


def test_from_rows_drops_every_row_at_an_excluded_point(system):
    rows = store.load(system.processed).rows

    built = field_points.Points.from_rows(
        rows, fidelity="dft", elements=list(system.elements), excluded=["Li6 S2 Cl2"]
    )

    assert "Li3 S1 Cl1" not in built.formulas


def test_from_rows_takes_given_energies_above_the_hull():
    rows = table.make_rows(
        ["Li2 S1", "Li4 S2", "Li1"],
        fidelity="nn",
        reference=[False, False, True],
        e_above_hull_mev=[5.0, 3.0, 0.0],
    )
    cleaned, _ = table.clean(rows, elements=["Li", "S"], source="t.csv")

    built = field_points.Points.from_rows(cleaned, fidelity="nn", elements=["Li", "S"])

    assert (built.formulas, built.energies_mev.tolist()) == (
        ("Li4 S2", "Li1"),
        [3.0, 0.0],
    )


def test_from_rows_rejects_an_unknown_fidelity(system):
    rows = store.load(system.processed).rows

    with pytest.raises(DataError, match="set fidelity to one of"):
        field_points.Points.from_rows(
            rows, fidelity="nn", elements=list(system.elements)
        )


def test_locate_gives_pymatgen_phase_diagram_coordinates(points, totals):
    diagram = PhaseDiagram([ComputedEntry(f, e) for f, e in totals.items()])

    ours = points.locate(list(totals))

    assert points.elements == tuple(str(el) for el in diagram.elements)
    assert np.array_equal(ours, [diagram.pd_coords(Composition(f)) for f in totals])
