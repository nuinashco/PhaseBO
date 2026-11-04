"""References: known structures, such as the elements, relaxed by the same MLIP.

The hull needs every element's energy at the same level of theory as the searched
compositions, and known compounds make it honest. Each one is relaxed once, with
the cell free, from the structure given in its CIF file.
"""

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from attrs import field, frozen
from pymatgen.core import Composition

from phasebo.csp.framework import name_cell, read_counts
from phasebo.csp.mlip.base import Potential
from phasebo.csp.result import decode_atoms, encode_atoms
from phasebo.csp.search.base import relax

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


@frozen
class Reference:
    """A known structure relaxed with the cell free."""

    name: str  # Its CIF file's stem.
    formula: str  # The cell, as Al2N2.
    mlip: str
    mlip_version: str
    model: str
    n_atoms: int
    energy_ev: float  # Per cell.
    elapsed_s: float
    structure: Atoms = field(eq=False)

    def to_json(self) -> dict[str, Any]:
        """Return the reference as JSON, with its energy per atom beside per cell."""
        return {
            "name": self.name,
            "formula": self.formula,
            "mlip": self.mlip,
            "mlip_version": self.mlip_version,
            "model": self.model,
            "n_atoms": self.n_atoms,
            "energy_ev": self.energy_ev,
            "ev_per_atom": self.energy_ev / self.n_atoms,
            "elapsed_s": self.elapsed_s,
            "structure": encode_atoms(self.structure),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Reference:
        """Return the reference that `to_json` wrote."""
        fields = {
            k: v for k, v in data.items() if k not in {"ev_per_atom", "structure"}
        }
        return cls(**fields, structure=decode_atoms(data["structure"]))


def relax_reference(
    name: str,
    atoms: Atoms,
    calculator: Calculator,
    *,
    potential: Potential,
    fmax: float,
    steps: int,
) -> Reference:
    """Return the structure relaxed with the cell free, and its energy.

    `fmax` is in eV/A.
    """
    started = time.perf_counter()
    relaxed, energy_ev = relax(atoms, calculator, fmax=fmax, steps=steps, cell=True)
    return Reference(
        name=name,
        formula=name_cell(read_counts(atoms.get_chemical_formula())),
        mlip=potential.kind,
        mlip_version=potential.version,
        model=potential.model,
        n_atoms=len(atoms),
        energy_ev=energy_ev,
        elapsed_s=time.perf_counter() - started,
        structure=relaxed,
    )


def find_missing_elements(
    references: Iterable[str], formulas: Iterable[str]
) -> list[str]:
    """Return the elements of `formulas` that no single-element reference covers."""
    covered = {
        str(composition.elements[0])
        for composition in map(Composition, references)
        if len(composition.elements) == 1
    }
    needed = {str(el) for formula in formulas for el in Composition(formula).elements}
    return sorted(needed - covered)
