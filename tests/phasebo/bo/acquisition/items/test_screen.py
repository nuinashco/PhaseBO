"""The screen picks in cheap order, without a model."""

import numpy as np
import pytest

from phasebo.bo.acquisition.base import AcquisitionFactory


def test_screen_picks_the_lowest_cheap_energies_without_a_model():
    screen = AcquisitionFactory.from_config({"kind": "screen"})
    choices = np.arange(10.0).reshape(5, 2)

    picked = screen.pick(
        None,
        choices,
        size=2,
        rng=np.random.default_rng(0),
        cheap_mev=np.array([30.0, 5.0, 20.0, 5.0, 1.0]),
    )

    assert (screen.needs_model, screen.needs_cheap, picked) == (False, True, [4, 1])


def test_screen_refuses_to_pick_without_cheap_energies():
    screen = AcquisitionFactory.from_config({"kind": "screen"})

    with pytest.raises(ValueError, match="needs cheap energies"):
        screen.pick(None, np.zeros((3, 2)), size=1, rng=np.random.default_rng(0))
