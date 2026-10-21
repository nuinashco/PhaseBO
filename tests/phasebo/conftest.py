"""Synthetic Li-Sn-S-Cl field, processed as a script would and read as commands do."""

from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Composition

from phasebo.config import CandidatesConfig, ElementConfig, SystemConfig
from phasebo.data import store, table
from phasebo.field import points as field_points

MU = {"Li": -1.9, "Sn": -3.9, "S": -4.1, "Cl": -1.8}  # eV/atom of the pure elements.

# Formation energies (eV/atom). Li3SCl lies below the Li2S-LiCl tie line, so it is
# the one stable Li-S-Cl ternary; Li6S2Cl2 is its point in a double cell, 30 meV/atom
# higher.
EXPLORED = {
    "Li3 S1 Cl1": -0.96,
    "Li6 S2 Cl2": -0.93,
    "Li4 S1 Cl2": -0.93,
    "Li5 S2 Cl1": -0.90,
    "Li5 S1 Cl3": -0.95,
    "Li7 S2 Cl3": -0.94,
    "Li8 S3 Cl2": -0.92,
    "Li7 S1 Cl5": -0.95,
    "Li4 Sn1 S4": -0.78,
    "Li2 Sn1 S3": -0.70,
    "Li2 Sn1 Cl6": -0.80,
    "Li1 Sn1 S2 Cl1": -0.62,
    "Li3 Sn1 S3 Cl1": -0.74,
    "Li2 Sn1 S2 Cl2": -0.72,
}
REFERENCES = {
    "Li1": 0.0,
    "S1": 0.0,
    "Cl1": 0.0,
    "Li2 S1": -0.90,
    "Li1 Cl1": -1.00,
    "Sn1": 0.0,
    "Sn1 S2": -0.50,
    "Sn1 Cl4": -0.70,
}
CHARGES = {"Li": 1, "Sn": 4, "S": -2, "Cl": -1}

type SystemWriter = Callable[[Mapping[str, int]], SystemConfig]


def total_energy(formula: str, formation_ev: float) -> float:
    """Return the total energy of a cell (eV) from its formation energy per atom."""
    composition = Composition(formula)
    pure = sum(MU[str(el)] * n for el, n in composition.items())
    return pure + composition.num_atoms * formation_ev


@pytest.fixture
def totals() -> dict[str, float]:
    """Return the total energy of every cell of the field (eV)."""
    formation = {**EXPLORED, **REFERENCES}
    return {formula: total_energy(formula, e) for formula, e in formation.items()}


@pytest.fixture
def candidates() -> list[str]:
    return ["Li6 Sn1 S5", "Li3 Sn1 S2 Cl3", "Li1 Sn1 S1 Cl3"]


@pytest.fixture
def write_system(tmp_path: Path, candidates: list[str]) -> SystemWriter:
    """Return a writer of the field restricted to some elements, processed on disk."""

    def write(charges: Mapping[str, int]) -> SystemConfig:
        def inside(formula: str) -> bool:
            return {str(el) for el in Composition(formula).elements} <= set(charges)

        explored = [f for f in EXPLORED if inside(f)]
        references = [f for f in REFERENCES if inside(f)]
        formation = {**EXPLORED, **REFERENCES}
        formulas = explored + references
        totals = [total_energy(f, formation[f]) for f in formulas]
        raw = tmp_path / "raw.csv"
        pd.DataFrame({"formula": formulas, "energy": totals}).to_csv(raw, index=False)
        rows = table.make_rows(
            formulas,
            fidelity="dft",
            reference=[f in references for f in formulas],
            total_energy_ev=totals,
        )
        cleaned, _ = table.clean(rows, elements=charges, source=str(raw))
        (tmp_path / "process.py").write_text("# The script that wrote the table.\n")
        out = tmp_path / "processed.csv"
        store.write(out, cleaned, raw=[raw], script=tmp_path / "process.py")
        listed = pd.DataFrame({"formula": [c for c in candidates if inside(c)]})
        listed.to_csv(tmp_path / "candidates.csv", index=False)
        return SystemConfig(
            processed=out,
            elements={
                s: ElementConfig(charge=c, atoms=(0, 6)) for s, c in charges.items()
            },
            candidates=CandidatesConfig(max_atoms=8, file=tmp_path / "candidates.csv"),
        )

    return write


@pytest.fixture
def system(write_system: SystemWriter) -> SystemConfig:
    """Write the Li-Sn-S-Cl field with `candidates` to rank."""
    return write_system(CHARGES)


@pytest.fixture
def points(system: SystemConfig) -> field_points.Points:
    """Return the field's points at the dft fidelity."""
    rows = store.load(system.processed).rows
    return field_points.build(rows, fidelity="dft", elements=list(system.elements))
