"""Energies above the convex hull, from total energies (pymatgen)."""

from collections.abc import Sequence

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry


def measure(formulas: Sequence[str], totals_ev: Sequence[float]) -> np.ndarray:
    """Return each row's energy above the hull of all rows (meV/atom).

    This is the oracle hull: built from every row, as if all were computed already.
    `totals_ev` are total energies per cell.
    """
    entries = [
        ComputedEntry(formula, total)
        for formula, total in zip(formulas, totals_ev, strict=True)
    ]
    diagram = PhaseDiagram(entries)
    energies_mev = []
    for entry in entries:
        above_ev = diagram.get_e_above_hull(entry)
        if above_ev is None:
            # Returned only with on_error="ignore"; the default raises instead.
            msg = f"no energy above the hull for {entry.composition}"
            raise ValueError(msg)
        energies_mev.append(1000 * above_ev)
    return np.array(energies_mev)
