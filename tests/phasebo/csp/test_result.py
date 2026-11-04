"""Results measure how far a cell moved, and survive a trip through JSON."""

import numpy as np
import pytest
from ase.build import bulk

from phasebo.csp import framework, result
from phasebo.csp.mlip.base import Potential
from phasebo.csp.moves import Arrangement
from phasebo.csp.search.base import Found, Hop, Relaxed


def found_on(cell, arrangement):
    atoms = cell.place(arrangement)
    hop = Hop(epoch=0, energy_ev=-1.0, best_ev=-1.0, accepted=True, revisit=False)
    released = atoms.copy()
    released.set_cell(atoms.cell.array * 1.1, scale_atoms=True)
    return Found(
        on_framework=Relaxed(
            arrangement=arrangement,
            atoms=atoms,
            energy_ev=-1.0,
            trace=(hop,),
            n_unique=1,
        ),
        released=released,
        released_energy_ev=-2.0,
    )


@pytest.fixture
def measured():
    cell = framework.build("cubic", {"Al": 4, "N": 4}, phi=0.6, oxi={"Al": 3, "N": -3})
    arrangement = Arrangement(cations=("Al",) * 4 + (None,) * 8, anions=("N",) * 4)
    return result.measure(
        found_on(cell, arrangement),
        cell,
        formula="Al4N4",
        level="relax",
        potential=Potential(kind="emt", version="3.29.0", model="emt"),
        elapsed_s=1.5,
        symprec=0.3,
    )


def test_a_cell_on_its_ideal_sites_has_no_drift(measured):
    assert measured.drift_angstrom == pytest.approx(0.0, abs=1e-12)


def test_drift_ignores_a_rigid_shift_of_every_atom():
    atoms = bulk("NaCl", "rocksalt", a=5.6, cubic=True)
    ideal = atoms.get_positions()
    atoms.positions += [0.3, -0.2, 0.1]

    assert result.drift(atoms, ideal) == pytest.approx(0.0, abs=1e-12)


def test_drift_is_the_rms_distance_from_the_ideal_sites():
    atoms = bulk("NaCl", "rocksalt", a=5.6, cubic=True)
    ideal = atoms.get_positions()
    atoms.positions[0] += [0.8, 0, 0]

    # One atom of 8 moved 0.8 A; removing the mean shift leaves 0.7 on it, 0.1 on 7.
    expected = np.sqrt((0.7**2 + 7 * 0.1**2) / 8)
    assert result.drift(atoms, ideal) == pytest.approx(expected)


def test_measure_reports_the_free_relaxs_volume_change(measured):
    assert measured.volume_change_percent == pytest.approx(100 * (1.1**3 - 1))


def test_the_fidelity_label_joins_the_mlip_and_the_level(measured):
    assert (measured.fidelity, measured.key) == (
        "emt_relax",
        ("Al4N4", "cubic", "emt_relax"),
    )


def test_a_result_reads_back_from_its_json(measured):
    back = result.Result.from_json(measured.to_json())

    assert back == measured
    assert np.array_equal(back.released.positions, measured.released.positions)


def test_space_group_names_rocksalt():
    atoms = bulk("NaCl", "rocksalt", a=5.6)

    assert result.name_space_group(atoms, symprec=0.1) == "Fm-3m (#225)"
