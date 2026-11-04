"""Structure searches: arrangements of a composition in one framework, by MLIP energy.

Ported from PolyBO, the group's adaptation of ChemDASH (Sharp et al., PCCP 2020, 22,
18205). Every kind ends the same way: the best arrangement, relaxed at the
framework's cell, is relaxed again with the cell free.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from attrs import define, field, frozen
from pydantic import Field, NonNegativeFloat, PositiveFloat, PositiveInt

from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

from phasebo.csp import moves
from phasebo.csp.framework import Framework
from phasebo.csp.moves import Arrangement

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator


class StructureSearchConfig(KindConfig):
    """Settings every structure search shares: how each arrangement is relaxed."""

    rattle_angstrom: NonNegativeFloat = Field(
        default=0.1, description="Gaussian std per coordinate"
    )
    fmax: PositiveFloat = Field(default=0.1, description="eV/A; at fixed cell")
    steps: PositiveInt = 200
    release_fmax: PositiveFloat = Field(
        default=0.05, description="eV/A; of the final relax with the cell free"
    )
    release_steps: PositiveInt = 500


@frozen
class Hop:
    """One step of a search: the energy it reached and what the search kept."""

    epoch: int
    energy_ev: float  # Per cell.
    best_ev: float  # Per cell.
    accepted: bool
    revisit: bool  # The arrangement was relaxed before, so it was not again.


@frozen
class Relaxed:
    """The best arrangement a search reached and its cell, relaxed at fixed cell."""

    arrangement: Arrangement
    atoms: Atoms = field(eq=False)
    energy_ev: float  # Per cell.
    trace: tuple[Hop, ...]
    n_unique: int  # Distinct arrangements relaxed.


@frozen
class Found:
    """What a search found, on the framework and after the free relax."""

    on_framework: Relaxed
    released: Atoms = field(eq=False)
    released_energy_ev: float  # Per cell.


@define
class StructureSearch[C: StructureSearchConfig](Configurable[C], ABC):
    """Search for a composition's lowest arrangement in one framework."""

    @property
    @abstractmethod
    def level(self) -> str:
        """Return the level's name in fidelity labels, such as relax or bh50."""

    @abstractmethod
    def hop(
        self,
        framework: Framework,
        ions: tuple[dict[str, int], dict[str, int]],
        *,
        calculator: Calculator,
        rng: np.random.Generator,
    ) -> Relaxed:
        """Return the best arrangement of the cations and anions in `ions`."""

    def search(
        self,
        framework: Framework,
        ions: tuple[dict[str, int], dict[str, int]],
        *,
        calculator: Calculator,
        rng: np.random.Generator,
    ) -> Found:
        """Return the best arrangement, then its cell relaxed with the cell free."""
        best = self.hop(framework, ions, calculator=calculator, rng=rng)
        released, energy_ev = relax(
            best.atoms,
            calculator,
            fmax=self.config.release_fmax,
            steps=self.config.release_steps,
            cell=True,
        )
        return Found(on_framework=best, released=released, released_energy_ev=energy_ev)

    def start(
        self,
        framework: Framework,
        ions: tuple[dict[str, int], dict[str, int]],
        rng: np.random.Generator,
    ) -> Arrangement:
        """Return the cations on random holes, then the anions on random sites."""
        cations, anions = ions
        return Arrangement(
            cations=moves.fill(len(framework.cation_sites), cations, rng),
            anions=moves.fill(len(framework.anion_sites), anions, rng),
        )

    def score(
        self,
        framework: Framework,
        arrangement: Arrangement,
        *,
        calculator: Calculator,
        rng: np.random.Generator,
    ) -> tuple[Atoms, float]:
        """Return the arrangement pushed off its ideal sites, relaxed at fixed cell."""
        atoms = framework.place(arrangement)
        atoms.positions += rng.normal(
            scale=self.config.rattle_angstrom, size=atoms.positions.shape
        )
        return relax(atoms, calculator, fmax=self.config.fmax, steps=self.config.steps)


class StructureSearchFactory(ConfigurableFactory[StructureSearch[Any]]):
    """Registry of structure searches by `kind`; a run computes several levels."""

    config_group = "csp/search"
    multiple = True


def relax(
    atoms: Atoms,
    calculator: Calculator,
    *,
    fmax: float,
    steps: int,
    cell: bool = False,
) -> tuple[Atoms, float]:
    """Return a copy of `atoms` relaxed by FIRE, with the cell free if `cell`.

    Also return its energy (eV per cell). `fmax` is in eV/A.
    """
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = atoms.copy()
    atoms.calc = calculator
    target = FrechetCellFilter(atoms) if cell else atoms
    # ASE's annotations miss that FIRE takes a filter, and logfile None for no log.
    FIRE(target, logfile=None).run(  # pyright: ignore[reportArgumentType]
        fmax=fmax, steps=steps
    )
    return atoms, float(atoms.get_potential_energy())
