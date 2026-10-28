"""`phasebo-suggest`: pick the next compositions to compute from the computed ones.

Usage:

    phasebo-suggest                          # configs/suggest.yaml as it is

    phasebo-suggest bo/acquisition=ts seed=1    # Hydra overrides

Writes runs/<run_name>/ with next.csv (the batch to compute next) and posterior.csv
(each candidate point's predicted energy above the hull); scripts/plot_run.py draws
its figures.
"""

from pathlib import Path

import click
import hydra
import pandas as pd
import rootutils
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.bo import proposal
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory
from phasebo.cli import _launch
from phasebo.config import SuggestConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="suggest")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(SuggestConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: SuggestConfig, *, runs_dir: Path) -> Path:
    """Fit the surrogate to the explored points and pick the next batch."""
    loaded, points, excluded = _launch.load_points(cfg)
    listed = _launch.read_formulas(
        cfg.system.candidates.file, key="system.candidates.file"
    )
    # A point computed at another fidelity only is still a candidate at this one.
    cells = _launch.select_candidates(
        cfg.system, explored=[*points.formulas, *excluded], listed=listed
    )
    surrogate = SurrogateFactory.from_config(cfg.bo.surrogate)
    acquisition = AcquisitionFactory.from_config(cfg.bo.acquisition)
    _launch.require_cheap(
        {"bo.surrogate": surrogate, "bo.acquisition": acquisition},
        fix="only phasebo-benchmark has them, so choose another kind",
    )
    tags = _launch.tag_families(
        surrogate=cfg.bo.surrogate, acquisition=cfg.bo.acquisition
    )
    run_dir, rng = _launch.start(
        cfg,
        runs_dir=runs_dir,
        job_type="suggest",
        group=_launch.name_field(cfg.system),
        tags=tags,
    )
    _launch.write_data(run_dir, loaded)
    picks, posterior = proposal.propose(
        points,
        cells,
        surrogate=surrogate,
        acquisition=acquisition,
        batch_size=cfg.bo.batch_size,
        rng=rng,
    )
    pd.DataFrame({"formula": picks}).to_csv(run_dir / "next.csv", index=False)
    posterior.to_csv(run_dir / "posterior.csv", index=False)
    click.echo("\n".join(picks))
    return run_dir
