"""`phasebo-benchmark`: replay BO over the computed compositions, as if computing them.

Usage:

    phasebo-benchmark                                   # configs/benchmark.yaml

    phasebo-benchmark bo.start.size=8 bo/acquisition=ts # Hydra overrides

    phasebo-benchmark bo/acquisition=random             # the random baseline

    phasebo-benchmark -m seed=0,1,2 'run_name=sweep/${hydra:job.num}'

Writes runs/<run_name>/bo_path.csv: each evaluated point, its energy above the hull,
its batch (0 for the starting set) and whether it is a target. scripts/plot_run.py
draws its figures.
"""

from pathlib import Path

import click
import hydra
import rootutils
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.bo import replay
from phasebo.bo.acquisition.base import AcquisitionFactory
from phasebo.bo.start.base import StartFactory
from phasebo.bo.surrogate.base import SurrogateFactory
from phasebo.cli import _launch
from phasebo.config import BenchmarkConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="benchmark")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(BenchmarkConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: BenchmarkConfig, *, runs_dir: Path) -> Path:
    """Evaluate a starting set, then up to max_iter batches picked by BO."""
    loaded, points, _ = _launch.load_points(cfg)
    starting = StartFactory.from_config(cfg.bo.start)
    starting.check(points)
    surrogate = SurrogateFactory.from_config(cfg.bo.surrogate)
    acquisition = AcquisitionFactory.from_config(cfg.bo.acquisition)
    run_dir, rng = _launch.start(cfg, runs_dir=runs_dir)
    _launch.write_data(run_dir, loaded)
    path = replay.replay(
        points,
        start=starting,
        surrogate=surrogate,
        acquisition=acquisition,
        batch_size=cfg.bo.batch_size,
        max_iter=cfg.bo.max_iter,
        rng=rng,
    )
    path.to_csv(run_dir / "bo_path.csv", index=False)
    found = int(path["target"].to_numpy().sum())
    targets = len(points.targets)
    click.echo(f"{found} of {targets} targets found in {len(path)} evaluations")
    return run_dir
