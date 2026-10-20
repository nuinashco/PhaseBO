"""Config rules that pydantic's types alone do not enforce."""

import pytest
from pydantic import ValidationError

from phasebo.config import ElementConfig, SystemConfig
from phasebo.dataset.items.hull_energy_csv import HullEnergyCSVConfig

SYSTEM = {
    "processed": "p.csv",
    "elements": {"Li": {"charge": 1}, "S": {"charge": -2}},
    "candidates": {"max_atoms": 8},
}
HULL = {
    "kind": "hull_energy_csv",
    "raw": "r.csv",
    "references": {"start_row": 0},
    "fidelities": {"DFT": "dft"},
}


def test_element_atoms_must_be_min_then_max():
    with pytest.raises(ValidationError, match="put the minimum first"):
        ElementConfig(charge=1, atoms=(5, 2))


def test_system_dataset_is_validated_by_its_kind():
    system = SystemConfig.model_validate({**SYSTEM, "dataset": HULL})

    assert isinstance(system.dataset, HullEnergyCSVConfig)
    assert system.model_dump()["dataset"]["fidelities"] == {"DFT": "dft"}


def test_system_dataset_rejects_options_of_another_kind():
    dataset = {**HULL, "kind": "total_energy_csv"}

    with pytest.raises(ValidationError, match="fidelities"):
        SystemConfig.model_validate({**SYSTEM, "dataset": dataset})
