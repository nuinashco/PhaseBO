"""Basin hopping: the expensive level, PolyBO's Metropolis search over arrangements."""

from typing import TYPE_CHECKING

import numpy as np
import structlog
from attrs import define
from pydantic import Field, NonNegativeInt, PositiveFloat

from phasebo.csp import moves
from phasebo.csp.framework import Framework
from phasebo.csp.moves import Arrangement
from phasebo.csp.search.base import (
    Hop,
    Relaxed,
    StructureSearch,
    StructureSearchConfig,
    StructureSearchFactory,
)

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

log = structlog.stdlib.get_logger()


class BasinHopConfig(StructureSearchConfig):
    """Basin hopping settings."""

    epochs: NonNegativeInt = Field(
        default=50, description="moves after the start; names the level bh<epochs>"
    )
    kt_ev: PositiveFloat = Field(default=0.05, description="per cell, not per atom")
    max_swaps: int = Field(
        default=3, ge=2, description="most occupants one move permutes"
    )
    size_decay: PositiveFloat = Field(
        default=0.5, description="odds of a move of k + 1 occupants against one of k"
    )


@StructureSearchFactory.register("basin_hop")
@define
class BasinHopSearch(StructureSearch[BasinHopConfig]):
    """Moves on one sublattice, each relaxed at fixed cell and kept by Metropolis."""

    @property
    def level(self) -> str:
        return f"bh{self.config.epochs}"

    def hop(
        self,
        framework: Framework,
        ions: tuple[dict[str, int], dict[str, int]],
        *,
        calculator: Calculator,
        rng: np.random.Generator,
    ) -> Relaxed:
        config = self.config
        # Each distinct arrangement is relaxed once; a revisit takes no draws.
        seen: dict[Arrangement, tuple[Atoms, float]] = {}

        def evaluate(arrangement: Arrangement) -> tuple[Atoms, float]:
            if arrangement not in seen:
                seen[arrangement] = self.score(
                    framework, arrangement, calculator=calculator, rng=rng
                )
            return seen[arrangement]

        current = self.start(framework, ions, rng)
        best_atoms, current_ev = evaluate(current)
        best, best_ev = current, current_ev
        trace = [
            Hop(
                epoch=0,
                energy_ev=current_ev,
                best_ev=best_ev,
                accepted=True,
                revisit=False,
            )
        ]
        for epoch in range(1, config.epochs + 1):
            proposed = moves.propose(
                current, rng, max_swaps=config.max_swaps, size_decay=config.size_decay
            )
            if proposed is None:
                log.info("no_move", framework=framework.name, epoch=epoch)
                break
            revisit = proposed in seen
            atoms, energy_ev = evaluate(proposed)
            # The draw is taken only for an uphill move, as in PolyBO.
            accepted = bool(
                energy_ev < current_ev
                or rng.random() < np.exp(-(energy_ev - current_ev) / config.kt_ev)
            )
            if accepted:
                current, current_ev = proposed, energy_ev
                if energy_ev < best_ev:
                    best, best_atoms, best_ev = proposed, atoms, energy_ev
            trace.append(
                Hop(
                    epoch=epoch,
                    energy_ev=energy_ev,
                    best_ev=best_ev,
                    accepted=accepted,
                    revisit=revisit,
                )
            )
            log.debug("hop", epoch=epoch, energy_ev=energy_ev, accepted=accepted)
        return Relaxed(
            arrangement=best,
            atoms=best_atoms,
            energy_ev=best_ev,
            trace=tuple(trace),
            n_unique=len(seen),
        )
