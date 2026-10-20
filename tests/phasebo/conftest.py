"""Synthetic phase field on disk, raw and processed, as the commands read it."""

import pandas as pd
import pytest

from phasebo import processed, table
from phasebo.config import CandidatesConfig, ElementConfig, SystemConfig


@pytest.fixture
def candidates() -> list[str]:
    return ["Li6 Sn1 S5", "Li3 Sn1 S2 Cl3", "Li1 Sn1 S1 Cl3"]


@pytest.fixture
def system(li_sn_s_cl, candidates, tmp_path) -> SystemConfig:
    """Process the li_sn_s_cl field as a script would, with `candidates` to rank."""
    compositions, references, ions = li_sn_s_cl
    raw = tmp_path / "raw.csv"
    pd.DataFrame(compositions, columns=["formula", "energy"]).to_csv(raw, index=False)
    pd.DataFrame({"formula": candidates}).to_csv(
        tmp_path / "candidates.csv", index=False
    )
    first_reference = len(compositions) - len(references)
    rows = table.make_rows(
        [str(f) for f in compositions[:, 0]],
        fidelity="dft",
        reference=[row >= first_reference for row in range(len(compositions))],
        total_energy_ev=[float(e) for e in compositions[:, 1]],
    )
    cleaned, _ = table.clean(rows, elements=ions)
    out = tmp_path / "processed.csv"
    processed.write(out, cleaned, raw=[raw], script=tmp_path / "process.py")
    return SystemConfig(
        processed=out,
        elements={
            symbol: ElementConfig(charge=charge, atoms=(0, 6))
            for symbol, charge in ions.items()
        },
        candidates=CandidatesConfig(max_atoms=8, file=tmp_path / "candidates.csv"),
    )
