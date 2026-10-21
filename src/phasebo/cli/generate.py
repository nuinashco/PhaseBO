"""`phasebo-generate`: list the charge-balanced candidate compositions of a phase field.

Usage:

    phasebo-generate                 # configs/generate.yaml as it is

    phasebo-generate system.candidates.max_atoms=12

Writes runs/<run_name>/candidates_list.csv: one charge-balanced cell per row, at
points not computed yet. Edit it and pass it to `phasebo-suggest` as
system.candidates.file.
"""

from pathlib import Path

import click
import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.cli import _launch
from phasebo.config import GenerateConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="generate")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(GenerateConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: GenerateConfig, *, runs_dir: Path) -> Path:
    """Write the candidates at points not computed yet; return the run directory."""
    excluded = _launch.read_formulas(
        cfg.system.candidates.exclude, key="system.candidates.exclude"
    )
    loaded = _launch.load_table(cfg.system)
    rows = loaded.rows
    formulas = _launch.generate_formulas(cfg.system, rows, excluded=excluded)
    run_dir, _ = _launch.start(cfg, runs_dir=runs_dir)
    _launch.write_data(run_dir, loaded)
    path = run_dir / "candidates_list.csv"
    pd.DataFrame({"formula": formulas}).to_csv(path, index=False)
    click.echo(f"{len(formulas)} candidates written to {path}")
    return run_dir
