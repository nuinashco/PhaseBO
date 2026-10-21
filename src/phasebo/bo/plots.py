"""Figures of a phase field and of a replay, saved as PNG files."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from phasebo.field.points import Points


def save_hull_map(
    points: Points,
    path: Path,
    *,
    elements: Sequence[str],
    dpi: int,
    show: bool,
) -> bool:
    """Save the energy above the hull over a 4-element field's square; else skip.

    The square's axes are the fractions of the first element among the first two,
    and of the third among the last two, in config order. Return whether it saved.
    """
    if len(elements) != 4:
        return False
    import matplotlib.pyplot as plt
    from scipy import interpolate

    square = _place_on_square(points.formulas, elements)
    grid = np.arange(0, 1, 0.005)
    surface_mev = interpolate.LinearNDInterpolator(square, points.energies_mev)
    figure, axes = plt.subplots()
    filled = axes.contourf(
        grid, grid, surface_mev(*np.meshgrid(grid, grid)), 50, cmap="gist_heat"
    )
    figure.colorbar(filled, ax=axes).set_label(
        "Energy above hull (meV/atom)", fontsize=14
    )
    axes.scatter(*square.T, c="lime", marker="3", lw=1.5, label="Computed compositions")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    a, b, c, d = elements
    axes.set_xlabel(f"{a}/({a}+{b})", fontsize=14)
    axes.set_ylabel(f"{c}/({c}+{d})", fontsize=14)
    axes.legend(bbox_to_anchor=(0.5, 1.1), fontsize=10)
    _save(figure, path, dpi=dpi, show=show)
    return True


def save_convergence(
    path_rows: pd.DataFrame, path: Path, *, dpi: int, show: bool
) -> None:
    """Save each evaluation's energy above the hull and the best so far."""
    import matplotlib.pyplot as plt

    energies_mev = path_rows["e_above_hull_mev"].to_numpy()
    evaluation = np.arange(1, len(energies_mev) + 1)
    starts = int((path_rows["batch"] == 0).sum())
    figure, axes = plt.subplots()
    axes.plot(evaluation, energies_mev, "o", alpha=0.5, label="Evaluated")
    axes.plot(evaluation, np.minimum.accumulate(energies_mev), "-", label="Best so far")
    axes.axvline(starts + 0.5, ls="--", c="grey", label="End of starting set")
    axes.set_xlabel("Evaluation")
    axes.set_ylabel("Energy above hull (meV/atom)")
    axes.legend()
    _save(figure, path, dpi=dpi, show=show)


def _place_on_square(formulas: Sequence[str], elements: Sequence[str]) -> np.ndarray:
    a, b, c, d = elements
    rows = []
    for formula in formulas:
        amount = Composition(formula).get_el_amt_dict()
        first, second = amount.get(a, 0.0), amount.get(b, 0.0)
        third, fourth = amount.get(c, 0.0), amount.get(d, 0.0)
        # A cell missing both elements of a pair sits at 1, as the original plot put it.
        x = first / (first + second) if first + second else 1.0
        y = third / (third + fourth) if third + fourth else 1.0
        rows.append((x, y))
    return np.array(rows)


def _save(figure: Any, path: Path, *, dpi: int, show: bool) -> None:
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
