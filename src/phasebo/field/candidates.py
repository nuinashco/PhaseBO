"""Candidate compositions: charge-balanced cells within per-element atom ranges."""

from collections.abc import Iterable, Mapping
from itertools import product

from pymatgen.core import Composition


def generate(
    charges: Mapping[str, int],
    ranges: Mapping[str, tuple[int, int]],
    *,
    max_atoms: int,
) -> list[Composition]:
    """Return every charge-neutral cell of at most `max_atoms` atoms.

    `charges` maps each element to its oxidation state, `ranges` to its inclusive
    [min, max] atoms per cell.
    """
    symbols = list(charges)
    spans = [range(ranges[s][0], ranges[s][1] + 1) for s in symbols]
    cells: list[Composition] = []
    for amounts in product(*spans):
        if not 0 < sum(amounts) <= max_atoms:
            continue
        if sum(charges[s] * n for s, n in zip(symbols, amounts, strict=True)):
            continue
        cells.append(Composition(dict(zip(symbols, amounts, strict=True))))
    return cells


def drop_explored(
    candidates: Iterable[Composition], explored: Iterable[Composition]
) -> list[Composition]:
    """Return the candidates whose point no explored composition shares."""
    seen = {c.reduced_composition for c in explored}
    return [c for c in candidates if c.reduced_composition not in seen]
