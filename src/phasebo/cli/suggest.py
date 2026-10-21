"""`phasebo-suggest`: pick the next compositions to compute from the computed ones.

Usage:

    phasebo-suggest                          # configs/suggest.yaml as it is

    phasebo-suggest bo/acquisition=ts seed=1    # Hydra overrides

Writes runs/<run_name>/ with next.csv (the batch to compute next), posterior.csv
(each candidate point's predicted energy above the hull) and, for 4-element fields,
convex_hull.png.
"""

from pathlib import Path

import click
import hydra
import pandas as pd
import rootutils
import structlog
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.bo import plots, proposal
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateFactory
from phasebo.cli import _launch
from phasebo.config import SuggestConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")

log = structlog.stdlib.get_logger()


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
    formulas = _launch.select_candidates(
        cfg.system, loaded.rows, listed=listed, excluded=excluded
    )
    cells = proposal.select_cells(points, formulas, excluded=excluded)
    surrogate = SurrogateFactory.from_config(cfg.bo.surrogate)
    acquisition = AcquisitionFactory.from_config(cfg.bo.acquisition)
    run_dir, rng = _launch.start(cfg, runs_dir=runs_dir)
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
    elements = list(cfg.system.elements)
    if not plots.save_hull_map(
        points,
        run_dir / "convex_hull.png",
        elements=elements,
        dpi=cfg.plot_dpi,
        show=cfg.show_plots,
    ):
        log.info("hull_map_skipped", elements=len(elements))
    return run_dir
