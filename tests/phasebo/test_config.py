"""Config rules that pydantic's types alone do not enforce."""

import pytest
from pydantic import ValidationError

from nuikit.errors import ConfigError

from phasebo.config import ElementConfig, SuggestConfig


def test_element_atoms_must_be_min_then_max():
    with pytest.raises(ValidationError, match="put the minimum first"):
        ElementConfig(charge=1, atoms=(5, 2))


def test_family_fields_are_validated_by_their_kind(system):
    cfg = SuggestConfig.model_validate(
        {
            "system": system.model_dump(),
            "bo": {
                "surrogate": {"kind": "gp", "kernel": "rbf"},
                "acquisition": {"kind": "ts"},
            },
        }
    )

    assert (type(cfg.bo.surrogate).__name__, type(cfg.bo.acquisition).__name__) == (
        "GPConfig",
        "TSConfig",
    )


def test_an_unknown_kind_names_the_registered_ones(system):
    with pytest.raises(ConfigError, match="registered: qlogei, random, ts"):
        SuggestConfig.model_validate(
            {
                "system": system.model_dump(),
                "bo": {"surrogate": {"kind": "gp"}, "acquisition": {"kind": "ucb"}},
            }
        )
