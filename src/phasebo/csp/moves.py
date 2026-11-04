"""Arrangements of a composition on a framework's sites, and ChemDASH swaps of them.

Ported from PolyBO, the group's adaptation of ChemDASH (Sharp et al., PCCP 2020, 22,
18205). A move permutes 2 to `max_swaps` occupants of one sublattice so that each lands
on a site that held a different one. Empty sites take part, so the same move gives
both swaps (Sr <-> Al) and hops into a hole. Nothing moves between sublattices, so the
anion packing stays the framework it started as.
"""

import numpy as np
from attrs import frozen

# Occupants of one sublattice's sites: an element, or None for an empty site.
type Occupants = tuple[str | None, ...]

# PolyBO's retry budgets: random draws tried before a swap or derangement gives up.
_SWAP_TRIES = 50
_DERANGE_TRIES = 50


@frozen
class Arrangement:
    """The occupant of every cation hole and every anion site."""

    cations: Occupants
    anions: Occupants


def fill(n_sites: int, counts: dict[str, int], rng: np.random.Generator) -> Occupants:
    """Return `counts` atoms spread at random over `n_sites` sites, the rest empty."""
    pool: list[str | None] = [el for el, n in counts.items() for _ in range(n)]
    if len(pool) > n_sites:
        msg = f"{len(pool)} atoms do not fit on {n_sites} sites"
        raise ValueError(msg)
    pool += [None] * (n_sites - len(pool))
    return tuple(pool[i] for i in rng.permutation(len(pool)))


def swap(
    occupants: Occupants,
    rng: np.random.Generator,
    *,
    max_swaps: int,
    size_decay: float,
) -> Occupants | None:
    """Return the occupants with 2 to `max_swaps` of them permuted; None if impossible.

    A move of k + 1 occupants is `size_decay` times as likely as one of k.
    """
    if len(set(occupants)) < 2:
        return None
    sizes = np.arange(2, min(max_swaps, len(occupants)) + 1)
    weights = size_decay ** (sizes - 2)
    k = int(rng.choice(sizes, p=weights / weights.sum()))
    for _ in range(_SWAP_TRIES):
        picked = rng.choice(len(occupants), k, replace=False)
        if len({occupants[i] for i in picked}) > 1:
            moved = list(occupants)
            order = _derange([occupants[i] for i in picked], rng)
            for slot, j in zip(picked, order, strict=True):
                moved[slot] = occupants[picked[j]]
            return tuple(moved)
    return None


def propose(
    arrangement: Arrangement,
    rng: np.random.Generator,
    *,
    max_swaps: int,
    size_decay: float,
) -> Arrangement | None:
    """Return one move on a sublattice picked at random; None if neither can move."""
    # PolyBO tries either sublattice first with even odds.
    cations_first = rng.random() < 0.5
    for on_cations in (cations_first, not cations_first):
        occupants = arrangement.cations if on_cations else arrangement.anions
        moved = swap(occupants, rng, max_swaps=max_swaps, size_decay=size_decay)
        if moved is None:
            continue
        if on_cations:
            return Arrangement(cations=moved, anions=arrangement.anions)
        return Arrangement(cations=arrangement.cations, anions=moved)
    return None


def _derange(occupants: list[str | None], rng: np.random.Generator) -> np.ndarray:
    """Return a permutation that moves each occupant onto a different one, if found.

    Otherwise the one, of those tried, that leaves the fewest in place.
    """
    best: tuple[int, np.ndarray] | None = None
    for _ in range(_DERANGE_TRIES):
        order = rng.permutation(len(occupants))
        same = sum(occupants[order[i]] == occupants[i] for i in range(len(occupants)))
        if best is None or same < best[0]:
            best = (same, order)
        if not same:
            break
    assert best is not None  # At least one try ran.
    return best[1]
