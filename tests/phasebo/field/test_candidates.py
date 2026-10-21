"""Candidates are charge-neutral cells inside the atom ranges, at unexplored points."""

from pymatgen.core import Composition

from phasebo.field import candidates

CHARGES = {"Li": 1, "S": -2, "Cl": -1}


def formulas(cells):
    return [cell.formula for cell in cells]


def test_generate_lists_neutral_cells_within_max_atoms():
    ranges = {"Li": (0, 4), "S": (0, 2), "Cl": (0, 2)}

    cells = candidates.generate(CHARGES, ranges, max_atoms=5)

    assert formulas(cells) == ["Li1 Cl1", "Li2 Cl2", "Li2 S1", "Li3 S1 Cl1"]


def test_generate_ranges_include_their_upper_bound():
    ranges = {"Li": (0, 2), "S": (0, 1), "Cl": (0, 0)}

    cells = candidates.generate(CHARGES, ranges, max_atoms=8)

    assert formulas(cells) == ["Li2 S1"]


def test_drop_explored_drops_every_cell_at_an_explored_point():
    cells = [Composition("Li2 Cl2"), Composition("Li1 Cl1"), Composition("Li2 S1")]

    kept = candidates.drop_explored(cells, [Composition("Li3 Cl3")])

    assert formulas(kept) == ["Li2 S1"]
