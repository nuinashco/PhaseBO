"""Random acquisition needs no model and repeats with its generator."""

import numpy as np

from phasebo.bo.acquisition.base import AcquisitionFactory


def test_random_picks_distinct_choices_without_a_model():
    random = AcquisitionFactory.from_config({"kind": "random"})
    choices = np.arange(20.0).reshape(10, 2)

    first = random.pick(None, choices, size=4, rng=np.random.default_rng(3))
    again = random.pick(None, choices, size=4, rng=np.random.default_rng(3))

    assert (random.needs_model, len(set(first)), first == again) == (False, 4, True)
