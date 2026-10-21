"""Command start-up, and the inputs that need the config schemas to be read."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
from pymatgen.core import Composition

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
    points = field_points.build(
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


def generate_formulas(
    system: SystemConfig, rows: pd.DataFrame, *, excluded: list[str] | None
) -> list[str]:
    """Return the charge-balanced cells at points neither computed nor excluded."""
    top = system.candidates.max_atoms
    ranges = {s: e.atoms or (0, top) for s, e in system.elements.items()}
    cells = candidates.generate(map_charges(system), ranges, max_atoms=top)
    explored = [Composition(f) for f in [*rows["formula"], *(excluded or [])]]
    return [cell.formula for cell in candidates.drop_explored(cells, explored)]


def select_candidates(
    system: SystemConfig,
    rows: pd.DataFrame,
    *,
    listed: list[str] | None,
    excluded: list[str] | None,
) -> list[str]:
    """Return the formulas to rank: the listed ones, else the generated ones."""
    if listed is None:
        listed = generate_formulas(system, rows, excluded=excluded)
    if not listed:
        raise DataError(
            "no candidates left to rank; widen system.elements atoms or "
            "system.candidates.max_atoms"
        )
    return listed


def read_formulas(path: Path | None, *, key: str) -> list[str] | None:
    """Return the first column of the CSV of formulas that config `key` names."""
    if path is None:
        return None
    path = paths.ROOT / path
    if not path.is_file():
        raise DataError(f"no formula file at {path}; fix {key} or set it to null")
    return [str(formula) for formula in pd.read_csv(path).iloc[:, 0]]
