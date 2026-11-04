"""Frameworks hand a composition the cells and sites PolyBO builds for it."""

import numpy as np
import pytest

from nuikit.errors import DataError

from phasebo.csp import framework
from phasebo.csp.moves import Arrangement

SR3AL3N5 = {"Sr": 3, "Al": 3, "N": 5}
OXI = {"Sr": 2, "Al": 3, "N": -3}


def build(name, counts=SR3AL3N5, **kwargs):
    return framework.build(
        name, counts, phi=0.6, oxi=framework.oxidation_states(counts), **kwargs
    )


@pytest.mark.parametrize(
    ("name", "supercell", "lengths"),
    [
        ("cubic", (2, 1, 1), [8.141, 4.071, 4.071]),
        ("wurtzite", (2, 2, 1), [5.757, 5.757, 4.700]),
        ("hexagonal", (2, 2, 1), [6.060, 6.060, 4.242]),
    ],
)
def test_sr3al3n5_cells_are_those_of_polybos_reference_run(name, supercell, lengths):
    cell = build(name)

    assert cell.supercell == supercell
    assert np.linalg.norm(cell.cell, axis=1) == pytest.approx(lengths, abs=1e-3)


def test_a_24_atom_quaternary_gets_polybos_wurtzite_supercell():
    counts = framework.read_counts("Ga2Sc2Al8N12")

    cell = build("wurtzite", counts)

    assert (cell.supercell, len(cell.anion_sites), len(cell.cation_sites)) == (
        (2, 3, 1),
        12,
        36,
    )
    assert np.linalg.norm(cell.cell, axis=1) == pytest.approx(
        [5.9698, 8.9548, 4.8744], abs=1e-4
    )


def test_every_framework_gives_a_composition_the_same_sites():
    sites = {
        (len(cell.anion_sites), len(cell.cation_sites))
        for cell in map(build, framework.NAMES)
    }

    assert sites == {(8, 24)}


def test_anion_sites_round_up_to_four_and_leave_a_hole_per_cation():
    assert (
        framework.count_anion_sites({"Sr": 3, "Al": 2, "N": 4}, OXI),
        framework.count_anion_sites(SR3AL3N5, OXI),
        framework.count_anion_sites({"Sr": 9, "Al": 9, "N": 4}, OXI),
    ) == (4, 8, 8)


def test_cell_volume_is_the_ionic_volume_over_phi():
    cell = build("hexagonal")

    volume = abs(np.linalg.det(cell.cell))
    assert volume == pytest.approx(framework.ionic_volume(SR3AL3N5, OXI) / 0.6)


def test_c_over_a_replaces_the_prototypes_own():
    cell = build("wurtzite", c_over_a=2.0)

    a, _, c = np.linalg.norm(cell.cell, axis=1)
    assert c / (a / 2) == pytest.approx(2.0)


def test_a_composition_without_an_anion_is_refused():
    with pytest.raises(DataError, match="no anion"):
        framework.count_anion_sites({"Al": 2}, {"Al": 3})


def test_a_formula_with_fractional_atoms_is_refused():
    with pytest.raises(DataError, match="give a whole cell, such as Al1N1"):
        framework.read_counts("Al0.5N0.5")


def test_read_counts_orders_the_elements_as_pymatgen_writes_them():
    assert list(framework.read_counts("N5Al3Sr3")) == ["Sr", "Al", "N"]


def test_placed_atoms_never_overlap():
    cell = build("cubic")
    arrangement = Arrangement(
        cations=("Sr",) * 3 + ("Al",) * 3 + (None,) * 18,
        anions=("N",) * 5 + (None,) * 3,
    )

    atoms = cell.place(arrangement)

    distances = atoms.get_all_distances(mic=True)
    assert len(atoms) == 11
    assert distances[~np.eye(11, dtype=bool)].min() > 0.5
