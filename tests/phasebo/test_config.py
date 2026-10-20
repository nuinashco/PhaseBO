"""Config rules that pydantic's types alone do not enforce."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from phasebo.config import SystemConfig

IONS = {"Li": 1, "S": -2, "Cl": -1}


def test_limits_must_name_exactly_the_ions():
    with pytest.raises(ValidationError, match="limits must name exactly the ions"):
        SystemConfig(
            inputfile=Path("f.csv"),
            reference_index=0,
            ions=IONS,
            max_atoms=8,
            limits={"Li": (0, 4), "S": (0, 4)},
        )
