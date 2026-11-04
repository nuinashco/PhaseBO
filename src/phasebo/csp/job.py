"""Jobs: one composition in one framework at one level, each with its own seed."""

import hashlib
import time
from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from attrs import field, frozen

from phasebo.csp import framework as frameworks
from phasebo.csp.framework import FrameworkName
from phasebo.csp.mlip.base import Potential
from phasebo.csp.result import Result, measure
from phasebo.csp.search.base import StructureSearch

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


@frozen
class Recipe:
    """How a job builds its cell and measures the result."""

    phi: float  # The cell's volume is the ions' Shannon volume over phi.
    c_over_a: float | None  # None: each prototype's own.
    symprec: float  # Of the released cell's space group, in A.


@frozen
class Job:
    """One composition in one framework, searched at one level."""

    formula: str  # The cell, as Sr3Al3N5.
    framework: FrameworkName
    search: StructureSearch[Any] = field(eq=False)

    def key(self, mlip: str) -> tuple[str, str, str]:
        """Return the job's name: the cell, the framework and the fidelity label."""
        return (self.formula, self.framework, f"{mlip}_{self.search.level}")


def plan(
    formulas: Sequence[str],
    names: Sequence[FrameworkName],
    searches: Sequence[StructureSearch[Any]],
    *,
    mlip: str,
    done: Collection[tuple[str, str, str]],
) -> list[Job]:
    """Return every job of the formulas, frameworks and searches not `done` yet.

    `done` holds the keys of the results a run already has.
    """
    jobs = [
        Job(formula=formula, framework=name, search=search)
        for formula in formulas
        for name in names
        for search in searches
    ]
    return [job for job in jobs if job.key(mlip) not in done]


def seed(run_seed: int, job: Job) -> np.random.Generator:
    """Return the job's generator, spawned from the run's seed by the job's name.

    The name, not the job's place in the run, picks the stream, so a result does not
    depend on the order, the workers or what else the run computes.
    """
    words = (job.formula, job.framework, job.search.level)
    key = tuple(int.from_bytes(hashlib.sha256(w.encode()).digest()[:4]) for w in words)
    return np.random.default_rng(np.random.SeedSequence(run_seed, spawn_key=key))


def compute(
    job: Job,
    *,
    calculator: Calculator,
    potential: Potential,
    recipe: Recipe,
    rng: np.random.Generator,
) -> Result:
    """Return the job's result: the framework built, searched and measured."""
    counts = frameworks.read_counts(job.formula)
    oxi = frameworks.oxidation_states(counts)
    cell = frameworks.build(
        job.framework, counts, phi=recipe.phi, oxi=oxi, c_over_a=recipe.c_over_a
    )
    started = time.perf_counter()
    found = job.search.search(
        cell, frameworks.split_ions(counts, oxi), calculator=calculator, rng=rng
    )
    return measure(
        found,
        cell,
        formula=job.formula,
        level=job.search.level,
        potential=potential,
        elapsed_s=time.perf_counter() - started,
        symprec=recipe.symprec,
    )
