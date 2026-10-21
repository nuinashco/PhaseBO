"""Command start-up, and the inputs that need the config schemas to be read."""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from nuikit import runs, seeding
from nuikit.errors import DataError
from nuikit.logging import configure_logging

from phasebo import paths
from phasebo.config import RunConfig, SearchConfig, SystemConfig
from phasebo.data import store
from phasebo.field import candidates
from phasebo.field import points as field_points

log = structlog.stdlib.get_logger()


def start(cfg: RunConfig, *, runs_dir: Path) -> tuple[Path, np.random.Generator]:
    """Configure logging, seed, and create the run directory.

    Return the directory and the run's random generator.
    """
    configure_logging(level=cfg.log_level, fmt=cfg.log_format)
    rng = seeding.seed_everything(cfg.seed, deterministic=cfg.deterministic)
    run_dir = runs.start_run(cfg, name=cfg.run_name, runs_dir=runs_dir)
    if cfg.tracker == "wandb":
        runs.mirror_to_wandb(cfg, run_dir, project=cfg.wandb_project)
    structlog.contextvars.bind_contextvars(run=run_dir.name)
    log.info("run_started", run_dir=str(run_dir), seed=cfg.seed)
    return run_dir, rng


def load_table(system: SystemConfig) -> store.LoadedTable:
    """Return the phase field's processed table, checked against its raw files."""
    return store.load(paths.ROOT / system.processed)


def load_points(
    cfg: SearchConfig,
) -> tuple[store.LoadedTable, field_points.Points, list[str]]:
    """Return the processed table, its points at `cfg.fidelity`, and the exclusions."""
    loaded = load_table(cfg.system)
    excluded = read_formulas(
        cfg.system.candidates.exclude, key="system.candidates.exclude"
    )
    points = field_points.Points.from_rows(
        loaded.rows,
        fidelity=cfg.fidelity,
        elements=list(cfg.system.elements),
        excluded=excluded or [],
    )
    return loaded, points, excluded or []


def write_data(run_dir: Path, loaded: store.LoadedTable) -> None:
    """Write data.json: the processed table, its script and raw files, by sha256."""
    data = {"table": str(loaded.path), **loaded.sidecar.model_dump(mode="json")}
    text = json.dumps(data, indent=2, ensure_ascii=False)
    (run_dir / "data.json").write_text(text + "\n", encoding="utf-8")


def map_charges(system: SystemConfig) -> dict[str, int]:
    """Return each element's oxidation state, in config order."""
    return {symbol: element.charge for symbol, element in system.elements.items()}


def generate_formulas(system: SystemConfig, *, explored: Sequence[str]) -> list[str]:
    """Return the smallest charge-balanced cell of each point no `explored` shares."""
    top = system.candidates.max_atoms
    ranges = {s: e.atoms or (0, top) for s, e in system.elements.items()}
    cells = candidates.generate(map_charges(system), ranges, max_atoms=top)
    return candidates.select(cells, explored=explored)


def select_candidates(
    system: SystemConfig, *, explored: Sequence[str], listed: list[str] | None
) -> list[str]:
    """Return one cell per new point to rank, from the listed formulas or generated.

    `explored` holds the formulas computed or excluded: no candidate shares a point
    with one.
    """
    if listed is None:
        cells = generate_formulas(system, explored=explored)
        remedy = "widen system.elements atoms or system.candidates.max_atoms"
    else:
        cells = candidates.select(listed, explored=explored)
        remedy = "list others in system.candidates.file, or set it to null"
    if not cells:
        raise DataError(f"no candidate at a point not computed or excluded; {remedy}")
    return cells


def read_formulas(path: Path | None, *, key: str) -> list[str] | None:
    """Return the first column of the CSV of formulas that config `key` names."""
    if path is None:
        return None
    path = paths.ROOT / path
    if not path.is_file():
        raise DataError(f"no formula file at {path}; fix {key} or set it to null")
    return [str(formula) for formula in pd.read_csv(path).iloc[:, 0]]
