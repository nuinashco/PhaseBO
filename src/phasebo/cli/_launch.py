"""Steps every command shares: start the run, read the phase field, save plots."""

from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from nuikit import runs, seeding
from nuikit.errors import DataError
from nuikit.logging import configure_logging

from phasebo import paths
from phasebo.config import RunConfig, SystemConfig

log = structlog.stdlib.get_logger()


def start(cfg: RunConfig, *, runs_dir: Path) -> Path:
    """Configure logging, seed, and create the run directory; return it."""
    configure_logging(level=cfg.log_level, fmt=cfg.log_format)
    seeding.seed_everything(cfg.seed, deterministic=cfg.deterministic)
    run_dir = runs.start_run(cfg, name=cfg.run_name, runs_dir=runs_dir)
    if cfg.tracker == "wandb":
        runs.mirror_to_wandb(cfg, run_dir, project=cfg.wandb_project)
    structlog.contextvars.bind_contextvars(run=run_dir.name)
    log.info("run_started", run_dir=str(run_dir), seed=cfg.seed)
    return run_dir


def read_field(system: SystemConfig) -> tuple[np.ndarray, np.ndarray]:
    """Return the (formula, total energy) rows of the phase field and its references."""
    path = paths.ROOT / system.inputfile
    if not path.is_file():
        raise DataError(f"no phase field data at {path}; set system.inputfile")
    rows = pd.read_csv(path).to_numpy()
    if system.reference_index >= len(rows):
        raise DataError(
            f"{path} has {len(rows)} rows, so no references start at row "
            f"{system.reference_index}; set system.reference_index"
        )
    return rows, rows[system.reference_index :]


def read_formulas(path: Path | None, *, key: str) -> list[str] | None:
    """Return the first column of the CSV of formulas that config `key` names."""
    if path is None:
        return None
    path = paths.ROOT / path
    if not path.is_file():
        raise DataError(f"no formula file at {path}; fix {key} or set it to null")
    return [str(formula) for formula in pd.read_csv(path).iloc[:, 0]]


def save_plot(path: Path, *, dpi: int, show: bool) -> None:
    """Save the current matplotlib figure, then show it if asked."""
    import matplotlib.pyplot as plt

    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close("all")
