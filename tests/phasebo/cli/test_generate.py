"""`phasebo-generate` lists charge-balanced candidates within the limits."""

from pymatgen.core import Composition

from phasebo.cli import generate
from phasebo.config import GenerateConfig


def test_generate_writes_charge_balanced_candidates(system, tmp_path):
    cfg = GenerateConfig(run_name="r", system=system)

    run_dir = generate.run(cfg, runs_dir=tmp_path / "runs")

    lines = (run_dir / "candidates_list.csv").read_text().split()
    charge = {"Li": 1, "Sn": 4, "S": -2, "Cl": -1}
    for formula in lines:
        amounts = Composition(formula).get_el_amt_dict()
        assert sum(charge[el] * n for el, n in amounts.items()) == 0, formula
        assert sum(amounts.values()) <= system.max_atoms, formula
    assert lines
