"""A run's results are written whole as they finish, and read back by name."""

import pytest
from ase.build import bulk
from ase.io import write

from nuikit.errors import DataError

from phasebo.csp import references, store
from phasebo.csp.mlip.base import Potential


@pytest.fixture
def reference(emt):
    return references.relax_reference(
        "Al",
        bulk("Al", "fcc", a=4.05),
        emt,
        potential=Potential(kind="emt", version="3.29.0", model="emt"),
        fmax=0.05,
        steps=20,
    )


def test_a_written_reference_reads_back_with_its_cif(tmp_path, reference):
    store.write_reference(tmp_path, reference)

    assert store.read_references(tmp_path) == [reference]
    assert sorted(p.name for p in (tmp_path / "references").iterdir()) == [
        "Al.cif",
        "Al.json",
    ]


def test_a_run_without_results_reads_none(tmp_path):
    assert store.read_results(tmp_path) == []


def test_read_structures_returns_each_cif_by_its_stem(references_dir):
    found = store.read_structures(references_dir)

    assert {name: len(atoms) for name, atoms in found.items()} == {"Al": 1, "N2": 2}


def test_read_structures_refuses_a_directory_without_cifs(tmp_path):
    with pytest.raises(DataError, match="no CIF files"):
        store.read_structures(tmp_path)


def test_read_structures_refuses_a_missing_directory(tmp_path):
    with pytest.raises(DataError, match=r"fix csp\.references\.directory"):
        store.read_structures(tmp_path / "absent")


def test_read_structures_refuses_a_cif_of_several_structures(tmp_path):
    write(tmp_path / "two.cif", [bulk("Al"), bulk("Cu")])

    with pytest.raises(DataError, match="several structures"):
        store.read_structures(tmp_path)
