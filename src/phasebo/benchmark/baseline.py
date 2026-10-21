"""Random picking's exact odds: the baseline every benchmark metric compares with."""

import math

import numpy as np


def survival(pool: int, picks: int, *, targets: int = 1) -> float:
    """Return the chance that `picks` random draws from `pool` points miss `targets`."""
    return math.comb(pool - targets, picks) / math.comb(pool, picks)


def hazard(pool: int, picked: int, *, batch: int) -> float:
    """Return the chance that `batch` draws find a target `picked` draws missed."""
    return min(1.0, batch / (pool - picked))


def expected_min(values_mev: np.ndarray, picks: int, *, floor_mev: float) -> float:
    """Return the expected lowest of `floor_mev` and `picks` values drawn at random."""
    if picks == 0:
        return floor_mev
    ordered = np.sort(values_mev)
    n = len(ordered)
    # The lowest draw ranks j or later when every draw comes from the n - j highest.
    tail = np.array([math.comb(n - j, picks) for j in range(n + 1)], dtype=float)
    lowest = (tail[:-1] - tail[1:]) / math.comb(n, picks)
    return float(np.sum(lowest * np.minimum(ordered, floor_mev)))
