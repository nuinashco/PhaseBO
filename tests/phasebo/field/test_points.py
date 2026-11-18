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


def test_names_are_the_reduced_formulas_of_the_lowest_cells():
    rows = table.make_rows(
        ["Li2 S1", "Li4 S2", "Li1"],
        fidelity="nn",
        reference=[False, False, True],
        e_above_hull_mev=[5.0, 3.0, 0.0],
    )
    cleaned, _ = table.clean(rows, elements=["Li", "S"], source="t.csv")

    built = field_points.Points.from_rows(cleaned, fidelity="nn", elements=["Li", "S"])

    assert built.names == ("Li2 S1", "Li1")


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


def test_a_field_with_frameworks_has_a_point_per_composition_and_framework(
    framework_points,
):
    labels = framework_points.labels

    assert {"Li3 S1 Cl1 (cubic)", "Li3 S1 Cl1 (wurtzite)", "Li1"} <= set(labels)
    assert framework_points.framework_names == ("cubic", "wurtzite")


def test_winners_are_each_compositions_lowest_framework(framework_points):
    labels = framework_points.labels
    winners = {labels[i] for i in framework_points.winners}

    # Li3SCl is in the first half, where cubic is lower; Li7SCl5 in the second.
    assert {"Li3 S1 Cl1 (cubic)", "Li7 S1 Cl5 (wurtzite)"} <= winners
    names = {framework_points.names[i] for i in framework_points.pool}
    assert len(winners) == len(names)


def test_inputs_add_the_framework_as_a_category_after_the_coordinates(
    framework_points,
):
    inputs = framework_points.inputs
    column = framework_points.categories[0]

    assert column == framework_points.coords.shape[1]
    assert set(inputs[framework_points.pool, column]) == {0.0, 1.0}


def test_find_gives_minus_one_for_a_point_not_in_the_field(framework_points):
    found = framework_points.find(
        ["Li3 S1 Cl1", "Li3 S1 Cl1"], ["wurtzite", "hexagonal"]
    )

    assert found[0] >= 0
    assert found[1] == -1
