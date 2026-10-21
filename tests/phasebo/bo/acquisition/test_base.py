"""Helpers of the acquisition rules: the fit they need, the rows they picked."""

import numpy as np
import pytest

from phasebo.bo.acquisition.base import locate, require_fit


def test_require_fit_refuses_a_missing_fit():
    with pytest.raises(ValueError, match="'qlogei' needs a fitted surrogate"):
        require_fit(None, "qlogei")


def test_locate_returns_the_index_of_each_picked_row():
    choices = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    assert locate(choices[[2, 0]], choices) == [2, 0]
