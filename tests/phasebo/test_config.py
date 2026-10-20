"""Config rules that pydantic's types alone do not enforce."""

import pytest
from pydantic import ValidationError

from phasebo.config import ElementConfig


def test_element_atoms_must_be_min_then_max():
    with pytest.raises(ValidationError, match="put the minimum first"):
        ElementConfig(charge=1, atoms=(5, 2))
