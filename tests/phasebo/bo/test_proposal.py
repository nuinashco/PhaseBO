"""Proposals pick a batch among new cells and predict each."""

import numpy as np

from phasebo.bo import proposal
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory

CELLS = ["Li6 Sn1 S5", "Li1 Sn1 S1 Cl3", "Li2 Sn1 S1 Cl4"]


def propose(points, cells):
    return proposal.propose(
        points,
        cells,
        surrogate=SurrogateFactory.from_config({"kind": "gp"}),
        acquisition=AcquisitionFactory.from_config({"kind": "qlogei"}),
        batch_size=2,
        rng=np.random.default_rng(0),
    )


def test_propose_picks_a_batch_and_predicts_every_cell(points):
    picks, posterior = propose(points, CELLS)

    assert (len(set(picks)), set(picks) <= set(CELLS)) == (2, True)
    assert sorted(posterior["formula"]) == sorted(CELLS)


def test_propose_does_not_depend_on_the_order_of_the_cells(points):
    picks, _ = propose(points, CELLS)

    assert propose(points, CELLS[::-1])[0] == picks
