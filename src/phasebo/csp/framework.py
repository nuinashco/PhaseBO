"""Frameworks a composition is placed in: an anion packing and its cation holes.

Ported from PolyBO, the group's adaptation of ChemDASH (Sharp et al., PCCP 2020, 22,
18205). Each framework offers three cation holes per anion, so the three compare
directly:

    cubic      ccp anions: 1 octahedral (rocksalt) + 2 tetrahedral (zincblende) holes
    wurtzite   hcp anions: 2 tetrahedral (wurtzite) + 1 octahedral (NiAs) holes
    hexagonal  AA-stacked layers: 2 in-plane (h-BN type) + 1 interlayer hole

The cell is the prototype scaled to the ions' Shannon volume over a packing fraction,
so a big cation such as Sr(2+) opens the framework up.
"""

from math import ceil, pi, sqrt
from typing import Literal, get_args

import numpy as np
from ase import Atoms
from attrs import field, frozen
from pymatgen.core import Composition, Element, Species

from nuikit.errors import DataError

from phasebo.csp.moves import Arrangement

type FrameworkName = Literal["cubic", "wurtzite", "hexagonal"]
NAMES: tuple[FrameworkName, ...] = get_args(FrameworkName.__value__)

# The lcm of the prototypes' anion counts (4 and 2): every framework builds this many.
_SITE_LCM = 4
# Cation holes per anion site, in every prototype.
_HOLES_PER_ANION = 3

type Site = tuple[float, float, float]


@frozen
class Prototype:
    """One framework's unit cell with a = 1: anion sites, cation holes, their types."""

    hexagonal: bool
    c_over_a: float
    anions: tuple[Site, ...]  # Fractional.
    cations: tuple[Site, ...]  # Fractional.
    holes: tuple[str, ...]  # Each cation site's hole type.

    def lattice(self, *, c_over_a: float | None = None) -> np.ndarray:
        """Return the lattice vectors as rows, with a = 1."""
        if not self.hexagonal:
            return np.eye(3)
        ratio = self.c_over_a if c_over_a is None else c_over_a
        return np.array([[1.0, 0, 0], [-0.5, sqrt(3) / 2, 0], [0, 0, ratio]])


_PROTOTYPES: dict[FrameworkName, Prototype] = {
    "cubic": Prototype(
        hexagonal=False,
        c_over_a=1.0,
        anions=((0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)),
        cations=(
            *((0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)),
            *((0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75)),
            *((0.25, 0.75, 0.75), (0.75, 0.75, 0.75), (0.25, 0.25, 0.75)),
            *((0.25, 0.75, 0.25), (0.75, 0.25, 0.25)),
        ),
        holes=("oct",) * 4 + ("tet",) * 8,
    ),
    # The ideal hcp c/a, sqrt(8/3), to three decimals as PolyBO has it.
    "wurtzite": Prototype(
        hexagonal=True,
        c_over_a=1.633,
        anions=((1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)),
        cations=(
            *((0, 0, 0), (0, 0, 0.5)),
            *((1 / 3, 2 / 3, 0.625), (2 / 3, 1 / 3, 0.125)),
            *((1 / 3, 2 / 3, 0.875), (2 / 3, 1 / 3, 0.375)),
        ),
        holes=("oct",) * 2 + ("tet",) * 4,
    ),
    "hexagonal": Prototype(
        hexagonal=True,
        c_over_a=1.40,  # PolyBO's choice for straight-stacked layers.
        anions=((1 / 3, 2 / 3, 0.25), (2 / 3, 1 / 3, 0.75)),
        cations=(
            *((2 / 3, 1 / 3, 0.25), (1 / 3, 2 / 3, 0.75)),
            *((0, 0, 0.25), (0, 0, 0.75)),
            *((0, 0, 0), (0, 0, 0.5)),
        ),
        holes=("plane",) * 4 + ("inter",) * 2,
    ),
}


@frozen
class Framework:
    """A scaled supercell of one prototype: its anion sites and cation holes."""

    name: FrameworkName
    cell: np.ndarray = field(eq=False)  # Lattice vectors as rows, in angstrom.
    anion_sites: np.ndarray = field(eq=False)  # Fractional, one row per site.
    cation_sites: np.ndarray = field(eq=False)  # Fractional, one row per hole.
    holes: tuple[str, ...]
    supercell: tuple[int, int, int]

    def place(self, arrangement: Arrangement) -> Atoms:
        """Return the cell with each occupant on its ideal site, cations first."""
        cations, anions = arrangement.cations, arrangement.anions
        symbols = [s for s in cations if s] + [s for s in anions if s]
        frac = np.vstack(
            [
                self.cation_sites[[i for i, s in enumerate(cations) if s]],
                self.anion_sites[[i for i, s in enumerate(anions) if s]],
            ]
        )
        return Atoms(symbols, scaled_positions=frac, cell=self.cell, pbc=True)


def read_counts(formula: str) -> dict[str, int]:
    """Return the atoms per cell of `formula`, in `pymatgen` formula order."""
    try:
        composition = Composition(formula)
    except ValueError as e:
        raise DataError(f"cannot read formula {formula!r}; fix it") from e
    if any(abs(n - round(n)) > 1e-6 for n in composition.values()):
        raise DataError(
            f"{formula} has fractional atoms; give a whole cell, such as "
            f"{(composition * 2).formula.replace(' ', '')}"
        )
    ordered = Composition(composition.formula)
    return {el.symbol: round(n) for el, n in ordered.items()}


def name_cell(counts: dict[str, int]) -> str:
    """Return the cell formula, as `Sr3Al3N5`."""
    return "".join(f"{el}{n}" for el, n in counts.items())


def oxidation_states(counts: dict[str, int]) -> dict[str, int]:
    """Return charge-balanced oxidation states, else each element's most common."""
    guesses = Composition(counts).oxi_state_guesses()
    if guesses:
        return {el: round(v) for el, v in guesses[0].items()}
    return {el: int(Element(el).common_oxidation_states[0]) for el in counts}


def split_ions(
    counts: dict[str, int], oxi: dict[str, int]
) -> tuple[dict[str, int], dict[str, int]]:
    """Return the cations' counts and the anions' (negative oxidation state)."""
    cations = {el: n for el, n in counts.items() if oxi[el] > 0}
    anions = {el: n for el, n in counts.items() if oxi[el] < 0}
    return cations, anions


def ionic_volume(counts: dict[str, int], oxi: dict[str, int]) -> float:
    """Return the sum of the ions' Shannon sphere volumes (angstrom^3).

    An ion without a Shannon radius takes its element's atomic radius.
    """
    volume = 0.0
    for el, n in counts.items():
        radius = Species(el, oxi[el]).ionic_radius or Element(el).atomic_radius
        if radius is None:
            raise DataError(f"{el} has no ionic or atomic radius; leave it out")
        volume += n * 4 / 3 * pi * float(radius) ** 3
    return volume


def count_anion_sites(counts: dict[str, int], oxi: dict[str, int]) -> int:
    """Return the anion sites every framework gets, so the three compare directly.

    The smallest multiple of 4 that holds the anions and offers a hole per cation.
    """
    n_anions = sum(n for el, n in counts.items() if oxi[el] < 0)
    n_cations = sum(counts.values()) - n_anions
    if not n_anions:
        raise DataError(
            f"{name_cell(counts)} has no anion (oxidation states {oxi}); "
            "give a composition with an element of negative oxidation state"
        )
    sites = _SITE_LCM * max(1, ceil(n_anions / _SITE_LCM))
    while _HOLES_PER_ANION * sites < n_cations:
        sites += _SITE_LCM
    return sites


def build(
    name: FrameworkName,
    counts: dict[str, int],
    *,
    phi: float,
    oxi: dict[str, int],
    c_over_a: float | None = None,
) -> Framework:
    """Return the supercell of `name` that holds `counts`, at the ionic volume / phi.

    `c_over_a` None keeps the prototype's own.
    """
    prototype = _PROTOTYPES[name]
    unit = prototype.lattice(c_over_a=c_over_a)
    repeats = count_anion_sites(counts, oxi) // len(prototype.anions)
    supercell = _choose_supercell(repeats, np.linalg.norm(unit, axis=1))
    cell = (unit.T * supercell).T
    scale = (ionic_volume(counts, oxi) / phi / abs(np.linalg.det(cell))) ** (1 / 3)
    return Framework(
        name=name,
        cell=cell * scale,
        anion_sites=_tile(prototype.anions, supercell),
        cation_sites=_tile(prototype.cations, supercell),
        holes=prototype.holes * repeats,
        supercell=supercell,
    )


def _choose_supercell(repeats: int, lengths: np.ndarray) -> tuple[int, int, int]:
    """Return the factorisation n1 * n2 * n3 = repeats with the most even edges.

    Ties go to the larger n1, then the larger n2.
    """
    best: tuple[tuple[float, int, int], tuple[int, int, int]] | None = None
    for n1 in range(1, repeats + 1):
        if repeats % n1:
            continue
        for n2 in range(1, repeats // n1 + 1):
            if (repeats // n1) % n2:
                continue
            n3 = repeats // (n1 * n2)
            edges = np.array([n1, n2, n3]) * lengths
            score = (float(edges.max() / edges.min()), -n1, -n2)
            if best is None or score < best[0]:
                best = (score, (n1, n2, n3))
    assert best is not None  # The factorisation 1 * 1 * repeats always exists.
    return best[1]


def _tile(frac: tuple[Site, ...], supercell: tuple[int, int, int]) -> np.ndarray:
    """Return the fractional sites repeated over the supercell, cell by cell."""
    n1, n2, n3 = supercell
    shifts = np.array(
        [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)]
    )
    return ((np.array(frac)[None] + shifts[:, None]) / np.array(supercell)).reshape(
        -1, 3
    )
