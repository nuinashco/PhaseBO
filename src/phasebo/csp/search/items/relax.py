"""Relax: the cheap level, one random arrangement relaxed once."""

from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from phasebo.csp.framework import Framework
from phasebo.csp.search.base import (
    Hop,
    Relaxed,
    StructureSearch,
    StructureSearchConfig,
    StructureSearchFactory,
)

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class RelaxConfig(StructureSearchConfig):
    """Relax settings."""


@StructureSearchFactory.register("relax")
@define
class RelaxSearch(StructureSearch[RelaxConfig]):
    """One random arrangement on the ideal sites, pushed off them and relaxed.

    The same as basin hopping with no epochs, so it shares that search's start.
    """

    @property
    def level(self) -> str:
        return "relax"

    def hop(
        self,
        framework: Framework,
        ions: tuple[dict[str, int], dict[str, int]],
        *,
        calculator: Calculator,
        rng: np.random.Generator,
    ) -> Relaxed:
        arrangement = self.start(framework, ions, rng)
        atoms, energy_ev = self.score(
            framework, arrangement, calculator=calculator, rng=rng
        )
        first = Hop(
            epoch=0,
            energy_ev=energy_ev,
            best_ev=energy_ev,
            accepted=True,
            revisit=False,
        )
        return Relaxed(
            arrangement=arrangement,
            atoms=atoms,
            energy_ev=energy_ev,
            trace=(first,),
            n_unique=1,
        )
