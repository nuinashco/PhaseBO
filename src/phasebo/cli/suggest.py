"""`phasebo-suggest`: pick the next compositions to compute from the computed ones.

Usage:

    phasebo-suggest                           # configs/suggest.yaml as it is

    phasebo-suggest bo.acquisition=ts seed=1  # Hydra overrides

Writes runs/<run_name>/ with next.csv (the batch to compute next), posterior.csv
(each candidate's predicted energy above the hull) and convex_hull.png.
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
from phasebo.config import SuggestConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="suggest")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(SuggestConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: SuggestConfig, *, runs_dir: Path) -> Path:
    """Fit the GP to the computed compositions and pick the next batch."""
    from phasebo.phase_field_bo import PhaseFieldBO

    compositions, references = _launch.read_field(cfg.system)
    candidates = _launch.read_formulas(
        cfg.system.compositionfile, key="system.compositionfile"
    )
    excluded = _launch.read_formulas(cfg.system.excludefile, key="system.excludefile")
    run_dir = _launch.start(cfg, runs_dir=runs_dir)
    bopt = PhaseFieldBO(
        compositions=compositions,
        references=references,
        ions=cfg.system.ions,
        mode="suggest",
        Ntot=cfg.system.max_atoms,
        limits=cfg.system.limits,
        batch=cfg.bo.batch_size,
        acquisition=cfg.bo.acquisition,
        kernel=cfg.bo.kernel,
        noise=cfg.bo.noise_mev,
        next_formulas=candidates,
        exceptions=excluded,
        output_dir=str(run_dir),
    )
    bopt.print_results()
    bopt.get_uncertainty()
    picks = [bopt.next_list[bopt.fcsym(x)] for x in bopt.next]
    pd.DataFrame({"composition": picks}).to_csv(run_dir / "next.csv", index=False)
    click.echo("\n".join(picks))
    bopt.plot_convex()
    _launch.save_plot(
        run_dir / "convex_hull.png", dpi=cfg.plot_dpi, show=cfg.show_plots
    )
    return run_dir
