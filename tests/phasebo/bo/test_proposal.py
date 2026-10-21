"""Proposals rank new candidate points, each by its smallest cell."""

import numpy as np
import pytest

from nuikit.errors import DataError

from phasebo.bo import proposal
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory


def test_select_cells_keeps_one_smallest_cell_per_new_point(points):
    candidates = ["Li12 Sn2 S10", "Li6 Sn1 S5", "Li3 S1 Cl1", "Li1 Sn1 S1 Cl3"]

    cells = proposal.select_cells(points, candidates)

    assert sorted(cells) == ["Li1 Sn1 S1 Cl3", "Li6 Sn1 S5"]


def test_select_cells_leaves_out_excluded_points(points):
    cells = proposal.select_cells(
        points, ["Li6 Sn1 S5", "Li1 Sn1 S1 Cl3"], excluded=["Li12 Sn2 S10"]
    )

    assert cells == ["Li1 Sn1 S1 Cl3"]


def test_select_cells_refuses_candidates_that_are_all_computed(points):
    with pytest.raises(DataError, match="computed or excluded point"):
        proposal.select_cells(points, ["Li3 S1 Cl1", "Li6 S2 Cl2"])


def test_propose_picks_a_batch_and_predicts_every_cell(points):
    cells = ["Li6 Sn1 S5", "Li1 Sn1 S1 Cl3", "Li2 Sn1 S1 Cl4"]

    picks, posterior = proposal.propose(
        points,
        cells,
        surrogate=SurrogateFactory.from_config({"kind": "gp"}),
        acquisition=AcquisitionFactory.from_config({"kind": "qlogei"}),
        batch_size=2,
        rng=np.random.default_rng(0),
    )

    assert (len(set(picks)), set(picks) <= set(cells)) == (2, True)
    assert sorted(posterior["formula"]) == sorted(cells)
