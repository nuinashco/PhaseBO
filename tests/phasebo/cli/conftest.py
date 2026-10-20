"""Synthetic phase field on disk, as the commands read it."""

import pandas as pd
import pytest

from phasebo.config import SystemConfig


@pytest.fixture
def candidates() -> list[str]:
    return ["Li6 Sn1 S5", "Li3 Sn1 S2 Cl3", "Li1 Sn1 S1 Cl3"]


@pytest.fixture
def system(li_sn_s_cl, candidates, tmp_path) -> SystemConfig:
    """Write the li_sn_s_cl field as a CSV, with `candidates` as its compositionfile."""
    compositions, references, ions = li_sn_s_cl
    pd.DataFrame(compositions).to_csv(tmp_path / "field.csv", index=False)
    pd.DataFrame({"formula": candidates}).to_csv(
        tmp_path / "candidates.csv", index=False
    )
    return SystemConfig(
        inputfile=tmp_path / "field.csv",
        reference_index=len(compositions) - len(references),
        ions=ions,
        max_atoms=8,
        limits=dict.fromkeys(ions, (0, 7)),
        compositionfile=tmp_path / "candidates.csv",
    )
