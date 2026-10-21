"""Candidate compositions: charge-balanced cells within per-element atom ranges."""

from collections.abc import Iterable, Mapping
from itertools import product

from pymatgen.core import Composition


def generate(
    charges: Mapping[str, int],
    ranges: Mapping[str, tuple[int, int]],
    *,
    max_atoms: int,
) -> list[str]:
    """Return the formula of every charge-neutral cell of at most `max_atoms` atoms.

    `charges` maps each element to its oxidation state, `ranges` to its inclusive
    [min, max] atoms per cell.
    """
    symbols = list(charges)
    spans = [range(ranges[s][0], ranges[s][1] + 1) for s in symbols]
    cells: list[str] = []
    for amounts in product(*spans):
        if not 0 < sum(amounts) <= max_atoms:
            continue
        if sum(charges[s] * n for s, n in zip(symbols, amounts, strict=True)):
            continue
        cells.append(Composition(dict(zip(symbols, amounts, strict=True))).formula)
    return cells


def select(cells: Iterable[str], *, explored: Iterable[str]) -> list[str]:
    """Return one cell per point no explored formula shares, in first-seen order.

    A point keeps its smallest cell, the cheapest to compute.
    """
    taken = {Composition(f).reduced_composition.formula for f in explored}
    smallest: dict[str, Composition] = {}
    for formula in cells:
        cell = Composition(formula)
        point = cell.reduced_composition.formula
        if point in taken:
            continue
        kept = smallest.get(point)
        if kept is None or cell.num_atoms < kept.num_atoms:
            smallest[point] = cell
    return [cell.formula for cell in smallest.values()]
