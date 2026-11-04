"""References are relaxed with the cell free and must cover every element."""

from ase.build import bulk

from phasebo.csp import references
from phasebo.csp.mlip.base import Potential

EMT = Potential(kind="emt", version="3.29.0", model="emt")


def test_relax_reference_frees_the_cell(emt):
    squeezed = bulk("Al", "fcc", a=3.8)

    reference = references.relax_reference(
        "Al", squeezed, emt, potential=EMT, fmax=0.01, steps=200
    )

    assert reference.structure.get_volume() > squeezed.get_volume()
    assert (reference.formula, reference.n_atoms) == ("Al1", 1)


def test_a_reference_reads_back_from_its_json(emt):
    reference = references.relax_reference(
        "Al", bulk("Al", "fcc", a=4.05), emt, potential=EMT, fmax=0.05, steps=50
    )

    assert references.Reference.from_json(reference.to_json()) == reference


def test_find_missing_elements_needs_each_element_alone():
    known = ["AlN", "Al"]

    assert references.find_missing_elements(known, ["Sc1Al3N4"]) == ["N", "Sc"]


def test_find_missing_elements_is_empty_when_every_element_has_one():
    assert references.find_missing_elements(["Al", "N2"], ["Al4N4"]) == []
