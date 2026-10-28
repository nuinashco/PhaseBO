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
from nuikit.registry import KindConfig

from phasebo import paths
from phasebo.benchmark.replay import Replays
from phasebo.config import BenchmarkConfig, RunConfig, SearchConfig, SystemConfig
from phasebo.data import store
from phasebo.field import candidates
from phasebo.field import points as field_points

log = structlog.stdlib.get_logger()

# How a run is reported, not what it computes: left out of W&B's config columns.
_REPORTING = (
    "run_name",
    "log_level",
    "log_format",
    "tracker",
    "wandb_project",
    "wandb_entity",
)


def start(
    cfg: RunConfig,
    *,
    runs_dir: Path,
    job_type: str,
    group: str | None = None,
    tags: Sequence[str] = (),
) -> tuple[Path, np.random.Generator]:
    """Configure logging, seed, and create the run directory; mirror it to W&B.

    Return the directory and the run's random generator. `job_type`, `group` and
    `tags` label the W&B run.
    """
    configure_logging(level=cfg.log_level, fmt=cfg.log_format)
    rng = seeding.seed_everything(cfg.seed, deterministic=cfg.deterministic)
    run_dir = runs.start_run(cfg, name=cfg.run_name, runs_dir=runs_dir)
    if cfg.tracker == "wandb":
        runs.mirror_to_wandb(
            cfg,
            run_dir,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            job_type=job_type,
            group=group,
            tags=tags,
            exclude=_REPORTING,
        )
    structlog.contextvars.bind_contextvars(run=run_dir.name)
    log.info("run_started", run_dir=str(run_dir), seed=cfg.seed)
    return run_dir, rng


def name_field(system: SystemConfig) -> str:
    """Return the phase field's name: its processed table's, by convention."""
    return Path(system.processed).stem


def tag_families(**configs: KindConfig) -> list[str]:
    """Return a tag per family choice and setting, as `surrogate.kernel=matern52`."""
    tags = []
    for role, config in configs.items():
        tags.append(f"{role}={config.kind}")
        settings = config.model_dump(mode="json", exclude={"kind"})
        tags += [f"{role}.{key}={value}" for key, value in settings.items()]
    return tags


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


def load_benchmark(
    run_dir: Path,
) -> tuple[BenchmarkConfig, store.LoadedTable, Replays]:
    """Return a benchmark run's config, processed table and replays.

    The table must be the one the run read: a changed table is refused.
    """
    for name in ("config.yaml", "data.json", "paths.parquet"):
        if not (run_dir / name).is_file():
            raise DataError(
                f"no {name} in {run_dir}; pass a phasebo-benchmark run directory"
            )
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
    cfg = BenchmarkConfig.model_validate(raw)
    loaded, points, _ = load_points(cfg)
    ran_on = json.loads((run_dir / "data.json").read_text(encoding="utf-8"))
    if ran_on["table_sha256"] != loaded.sidecar.table_sha256:
        raise DataError(
            f"{cfg.system.processed} changed since {run_dir.name} ran; check out its "
            "commit or rerun it"
        )
    replays = Replays.from_table(
        points,
        pd.read_parquet(run_dir / "paths.parquet"),
        batch_size=cfg.bo.batch_size,
        max_iter=cfg.benchmark.max_iter,
    )
    return cfg, loaded, replays


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
