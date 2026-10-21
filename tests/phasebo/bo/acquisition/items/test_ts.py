"""Thompson sampling picks distinct choices."""

import numpy as np

from phasebo.bo.acquisition.base import AcquisitionFactory


def test_ts_picks_distinct_choices(split):
    fit, choices = split

    picked = AcquisitionFactory.from_config({"kind": "ts"}).pick(
        fit, choices, size=2, rng=np.random.default_rng(0)
    )

    assert (len(set(picked)), all(0 <= i < len(choices) for i in picked)) == (2, True)
