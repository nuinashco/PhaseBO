"""`phasebo-generate` writes the generated candidates with a header, and data.json."""

import pandas as pd

from phasebo.cli import _launch, generate
from phasebo.config import GenerateConfig


def test_generate_writes_the_generated_candidates(system, tmp_path):
    cfg = GenerateConfig(run_name="r", system=system)

    run_dir = generate.run(cfg, runs_dir=tmp_path / "runs")

    written = pd.read_csv(run_dir / "candidates_list.csv")["formula"].tolist()
    rows = _launch.load_table(system).rows
    assert written == _launch.generate_formulas(system, rows, excluded=None)
    assert (run_dir / "data.json").is_file()
