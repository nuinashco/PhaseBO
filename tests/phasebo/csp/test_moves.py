"""Moves keep each sublattice's occupants and draw as PolyBO does from the same seed."""

import numpy as np
import pytest

from phasebo.csp import moves
from phasebo.csp.moves import Arrangement

E = None  # An empty site, short enough to line up the expected arrangements.


def test_fill_places_every_atom_and_leaves_the_rest_empty():
    occupants = moves.fill(12, {"Sr": 3, "Al": 2}, np.random.default_rng(1))

    assert (occupants.count("Sr"), occupants.count("Al"), occupants.count(None)) == (
        3,
        2,
        7,
    )


def test_fill_refuses_more_atoms_than_sites():
    with pytest.raises(ValueError, match="do not fit"):
        moves.fill(2, {"Sr": 3}, np.random.default_rng(1))


def test_swap_keeps_the_occupants_and_moves_them():
    rng = np.random.default_rng(2)
    occupants = moves.fill(12, {"Sr": 3, "Al": 2}, rng)

    moved = [moves.swap(occupants, rng, max_swaps=4, size_decay=0.5) for _ in range(50)]

    assert all(
        m is not None and sorted(m, key=str) == sorted(occupants, key=str)
        for m in moved
    )
    assert all(m != occupants for m in moved)


def test_swap_needs_two_kinds_of_occupant():
    rng = np.random.default_rng(3)

    assert moves.swap(("Sr",) * 5, rng, max_swaps=3, size_decay=0.5) is None


def test_propose_never_mixes_the_sublattices():
    rng = np.random.default_rng(4)
    arrangement = Arrangement(
        cations=moves.fill(12, {"Sr": 3, "Al": 2}, rng), anions=("N",) * 4
    )

    for _ in range(50):
        moved = moves.propose(arrangement, rng, max_swaps=3, size_decay=0.5)
        assert moved is not None
        assert moved.anions == arrangement.anions
        assert sorted(moved.cations, key=str) == sorted(arrangement.cations, key=str)
        arrangement = moved


def test_propose_can_move_an_anion_into_a_vacancy():
    rng = np.random.default_rng(5)
    arrangement = Arrangement(
        cations=moves.fill(24, {"Sr": 3, "Al": 3}, rng),
        anions=moves.fill(8, {"N": 5}, rng),
    )

    proposals = [
        moves.propose(arrangement, rng, max_swaps=3, size_decay=0.5) for _ in range(50)
    ]

    assert any(p is not None and p.anions != arrangement.anions for p in proposals)


def test_propose_gives_up_when_no_sublattice_can_move():
    arrangement = Arrangement(cations=("Li",) * 12, anions=("N",) * 4)

    moved = moves.propose(
        arrangement, np.random.default_rng(6), max_swaps=3, size_decay=0.5
    )

    assert moved is None


def test_moves_draw_what_polybo_draws_from_the_same_seed():
    rng = np.random.default_rng(0)
    arrangement = Arrangement(
        cations=moves.fill(12, {"Sc": 1, "Al": 2}, rng),
        anions=moves.fill(4, {"N": 3}, rng),
    )

    steps = [arrangement]
    for _ in range(3):
        moved = moves.propose(steps[-1], rng, max_swaps=3, size_decay=0.5)
        assert moved is not None
        steps.append(moved)

    # From PolyBO's random_fill and propose with default_rng(0).
    assert [(s.cations, s.anions) for s in steps] == [
        ((E, "Al", E, E, E, E, "Sc", E, E, E, E, "Al"), ("N", E, "N", "N")),
        ((E, "Al", E, E, E, E, "Sc", E, E, E, E, "Al"), (E, "N", "N", "N")),
        ((E, "Al", E, E, E, E, "Sc", "Al", E, E, E, E), (E, "N", "N", "N")),
        ((E, "Sc", E, "Al", E, E, E, "Al", E, E, E, E), (E, "N", "N", "N")),
    ]
