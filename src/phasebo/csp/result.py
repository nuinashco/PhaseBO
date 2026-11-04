"""Results: what one search found for one composition, framework and level."""

from typing import Any

import numpy as np
from ase import Atoms
from attrs import field, frozen

from phasebo.csp.framework import Framework, FrameworkName
from phasebo.csp.mlip.base import Potential
from phasebo.csp.search.base import Found, Hop


@frozen
class Result:
    """One composition in one framework at one level: energies, geometry, cost.

    Energies are in eV per cell: `framework_energy_ev` relaxed at the framework's
    cell, `released_energy_ev` after the free relax.
    """

    formula: str  # The cell, as Sr3Al3N5.
    framework: FrameworkName
    level: str
    mlip: str
    mlip_version: str
    model: str
    supercell: tuple[int, int, int]
    n_atoms: int
    framework_energy_ev: float
    released_energy_ev: float
    # RMS distance (A) of the atoms from their ideal sites, rigid shift removed.
    drift_angstrom: float
    volume_change_percent: float  # Of the free relax.
    space_group: str  # At a loose tolerance, so indicative only.
    n_unique: int  # Distinct arrangements relaxed.
    elapsed_s: float
    trace: tuple[Hop, ...]
    on_framework: Atoms = field(eq=False)
    released: Atoms = field(eq=False)

    @property
    def fidelity(self) -> str:
        """Return the fidelity label of the processed table, as chgnet_bh50."""
        return f"{self.mlip}_{self.level}"

    @property
    def key(self) -> tuple[str, str, str]:
        """Return what names the job: the cell, the framework and the fidelity."""
        return (self.formula, self.framework, self.fidelity)

    def to_json(self) -> dict[str, Any]:
        """Return the result as JSON, with energies per atom beside those per cell."""
        return {
            "formula": self.formula,
            "framework": self.framework,
            "level": self.level,
            "fidelity": self.fidelity,
            "mlip": self.mlip,
            "mlip_version": self.mlip_version,
            "model": self.model,
            "supercell": list(self.supercell),
            "n_atoms": self.n_atoms,
            "framework_energy_ev": self.framework_energy_ev,
            "released_energy_ev": self.released_energy_ev,
            "framework_ev_per_atom": self.framework_energy_ev / self.n_atoms,
            "released_ev_per_atom": self.released_energy_ev / self.n_atoms,
            "drift_angstrom": self.drift_angstrom,
            "volume_change_percent": self.volume_change_percent,
            "space_group": self.space_group,
            "n_unique": self.n_unique,
            "elapsed_s": self.elapsed_s,
            "trace": [
                {
                    "epoch": hop.epoch,
                    "energy_ev": hop.energy_ev,
                    "best_ev": hop.best_ev,
                    "accepted": hop.accepted,
                    "revisit": hop.revisit,
                }
                for hop in self.trace
            ],
            "on_framework": encode_atoms(self.on_framework),
            "released": encode_atoms(self.released),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Result:
        """Return the result that `to_json` wrote."""
        derived = {"fidelity", "framework_ev_per_atom", "released_ev_per_atom"}
        rebuilt = {"supercell", "trace", "on_framework", "released"}
        fields = {k: v for k, v in data.items() if k not in derived | rebuilt}
        return cls(
            **fields,
            supercell=tuple(data["supercell"]),
            trace=tuple(Hop(**hop) for hop in data["trace"]),
            on_framework=decode_atoms(data["on_framework"]),
            released=decode_atoms(data["released"]),
        )


def measure(
    found: Found,
    framework: Framework,
    *,
    formula: str,
    level: str,
    potential: Potential,
    elapsed_s: float,
    symprec: float,
) -> Result:
    """Return the result of a search, with its geometry measured.

    `symprec` is the space group's tolerance, in A.
    """
    best = found.on_framework
    ideal = framework.place(best.arrangement).get_positions()
    return Result(
        formula=formula,
        framework=framework.name,
        level=level,
        mlip=potential.kind,
        mlip_version=potential.version,
        model=potential.model,
        supercell=framework.supercell,
        n_atoms=len(best.atoms),
        framework_energy_ev=best.energy_ev,
        released_energy_ev=found.released_energy_ev,
        drift_angstrom=drift(best.atoms, ideal),
        volume_change_percent=100
        * (found.released.get_volume() / best.atoms.get_volume() - 1),
        space_group=name_space_group(found.released, symprec=symprec),
        n_unique=best.n_unique,
        elapsed_s=elapsed_s,
        trace=best.trace,
        on_framework=best.atoms,
        released=found.released,
    )


def drift(atoms: Atoms, ideal: np.ndarray) -> float:
    """Return the RMS distance (A) of `atoms` from `ideal`, rigid shift removed.

    `ideal` holds Cartesian positions in the same order and cell.
    """
    lattice = atoms.cell.array
    ideal_frac = np.linalg.solve(lattice.T, ideal.T).T
    shift = atoms.get_scaled_positions(wrap=False) - ideal_frac
    shift -= np.round(shift)
    shift -= shift.mean(axis=0)
    return float(np.sqrt((np.linalg.norm(shift @ lattice, axis=1) ** 2).mean()))


def name_space_group(atoms: Atoms, *, symprec: float) -> str:
    """Return the space group as `P1 (#1)`, or `?` when spglib finds none."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    try:
        analyzer = SpacegroupAnalyzer(
            # `pymatgen` annotates it with a stand-in of ASE's Atoms.
            AseAtomsAdaptor.get_structure(atoms),  # pyright: ignore[reportArgumentType]
            symprec=symprec,
        )
        return (
            f"{analyzer.get_space_group_symbol()} "
            f"(#{analyzer.get_space_group_number()})"
        )
    except Exception:  # noqa: BLE001 - an indicative label must not cost a job its result.
        return "?"


def encode_atoms(atoms: Atoms) -> dict[str, Any]:
    """Return a cell as JSON: lattice rows and Cartesian positions in A, symbols."""
    return {
        "symbols": atoms.get_chemical_symbols(),
        "cell": atoms.cell.array.tolist(),
        "positions": atoms.get_positions().tolist(),
    }


def decode_atoms(data: dict[str, Any]) -> Atoms:
    """Return the periodic cell that `encode_atoms` wrote."""
    return Atoms(
        data["symbols"], positions=data["positions"], cell=data["cell"], pbc=True
    )
