"""`phasebo-generate`: list the charge-balanced candidate compositions of a phase field.

Usage:

    phasebo-generate                 # configs/generate.yaml as it is

    phasebo-generate system.max_atoms=12

Writes runs/<run_name>/candidates_list.csv, which can be edited and passed to
`phasebo-suggest` as system.compositionfile.
"""

from pathlib import Path

import click
import hydra
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
    """Write the candidates that are not computed yet; return the run directory."""
    from phasebo.phase_field_bo import PhaseFieldBO

    compositions, references = _launch.read_field(cfg.system)
    excluded = _launch.read_formulas(cfg.system.excludefile, key="system.excludefile")
    run_dir = _launch.start(cfg, runs_dir=runs_dir)
    bopt = PhaseFieldBO(
        compositions=compositions,
        references=references,
        ions=cfg.system.ions,
        mode="generate",
        Ntot=cfg.system.max_atoms,
        limits=cfg.system.limits,
        exceptions=excluded,
        output_dir=str(run_dir),
    )
    path = run_dir / "candidates_list.csv"
    click.echo(f"{len(bopt.next_formulas or [])} candidates written to {path}")
    return run_dir
