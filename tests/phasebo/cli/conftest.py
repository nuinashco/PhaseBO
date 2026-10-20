"""Synthetic phase field on disk, raw and processed, as the commands read it."""

import pandas as pd
import pytest

from phasebo.cli import data
from phasebo.config import CandidatesConfig, ElementConfig, SystemConfig
from phasebo.dataset.base import ReferencesConfig
from phasebo.dataset.items.total_energy_csv import TotalEnergyCSVConfig


@pytest.fixture
def candidates() -> list[str]:
    return ["Li6 Sn1 S5", "Li3 Sn1 S2 Cl3", "Li1 Sn1 S1 Cl3"]


@pytest.fixture
def system(li_sn_s_cl, candidates, tmp_path) -> SystemConfig:
    """Write the li_sn_s_cl field as raw and processed CSVs, `candidates` to rank."""
    compositions, references, ions = li_sn_s_cl
    raw = tmp_path / "raw.csv"
    pd.DataFrame(compositions, columns=["formula", "energy"]).to_csv(raw, index=False)
    pd.DataFrame({"formula": candidates}).to_csv(
        tmp_path / "candidates.csv", index=False
    )
    system = SystemConfig(
        dataset=TotalEnergyCSVConfig(
            kind="total_energy_csv",
            raw=raw,
            references=ReferencesConfig(start_row=len(compositions) - len(references)),
        ),
        processed=tmp_path / "processed.csv",
        elements={
            symbol: ElementConfig(charge=charge, atoms=(0, 6))
            for symbol, charge in ions.items()
        },
        candidates=CandidatesConfig(max_atoms=8, file=tmp_path / "candidates.csv"),
    )
    text, _ = data.build(system)
    system.processed.write_text(text, encoding="utf-8")
    return system
