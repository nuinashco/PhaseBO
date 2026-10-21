"""Candidates are charge-neutral cells inside the atom ranges, one per new point."""

from phasebo.field import candidates

CHARGES = {"Li": 1, "S": -2, "Cl": -1}


def test_generate_lists_neutral_cells_within_max_atoms():
    ranges = {"Li": (0, 4), "S": (0, 2), "Cl": (0, 2)}

    cells = candidates.generate(CHARGES, ranges, max_atoms=5)

    assert cells == ["Li1 Cl1", "Li2 Cl2", "Li2 S1", "Li3 S1 Cl1"]


def test_generate_ranges_include_their_upper_bound():
    ranges = {"Li": (0, 2), "S": (0, 1), "Cl": (0, 0)}

    assert candidates.generate(CHARGES, ranges, max_atoms=8) == ["Li2 S1"]


def test_select_keeps_the_smallest_cell_of_each_new_point():
    cells = ["Li12 Sn2 S10", "Li6 Sn1 S5", "Li2 Cl2", "Li1 Cl1"]

    assert candidates.select(cells, explored=[]) == ["Li6 Sn1 S5", "Li1 Cl1"]


def test_select_drops_every_cell_at_an_explored_point():
    cells = ["Li2 Cl2", "Li1 Cl1", "Li2 S1"]

    assert candidates.select(cells, explored=["Li3 Cl3"]) == ["Li2 S1"]
