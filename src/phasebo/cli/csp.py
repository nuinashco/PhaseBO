"""`phasebo-csp`: predict the structures of compositions in frameworks with an MLIP.

Usage:

    phasebo-csp formulas=[Sc1Al3N4,Sc2Al2N4]        # CHGNet, both levels

    phasebo-csp formulas_file=cells.csv csp/search=[relax]    # the cheap level

    phasebo-csp formulas=[Al4N4] csp/mlip=emt csp.frameworks=[wurtzite]  # smoke

    phasebo-csp run_name=<name> ...       # resume: finished jobs are skipped

Every cell is placed in every framework and searched at every level of csp/search,
each job seeded by its name. runs/<run_name>/ gets results/ (each job's JSON and
CIFs as it finishes), references/, results.jsonl and references.jsonl; with
csp.references.directory also table.csv, a processed table whose fidelities are
<mlip>_<level> (chgnet_relax, chgnet_bh50). publish=<name> writes it to
data/processed/<name>.csv as well, for configs/system/<name>.yaml and
phasebo-benchmark. Resuming needs the settings the run started with.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click
import hydra
import pandas as pd
import rootutils
from attrs import frozen
from omegaconf import DictConfig, OmegaConf
from pandera.typing import DataFrame
from pymatgen.core import Composition

from nuikit.errors import ConfigError, DataError, cli_boundary
from nuikit.logging import LogFormat, LogLevel, configure_logging

from phasebo import paths
from phasebo.cli import _launch
from phasebo.config import CSPConfig
from phasebo.csp import framework, job, references, store
from phasebo.csp.mlip.base import MLIP, MLIPFactory, Potential, resolve_device
from phasebo.csp.references import Reference
from phasebo.csp.result import Result
from phasebo.csp.search.base import StructureSearchFactory
from phasebo.data import store as data_store
from phasebo.data import table
from phasebo.data.table import ProcessedTable

if TYPE_CHECKING:
    from ase import Atoms

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")

# Settings a resumed run may change: how it is reported and where it runs.
_FREE_ON_RESUME = (
    "run_name",
    "log_level",
    "log_format",
    "tracker",
    "wandb_project",
    "wandb_entity",
    "publish",
    "csp.device",
    "csp.workers",
)
# The files a table is computed from, which its sidecar pins.
_PINNED = ("config.yaml", "results.jsonl", "references.jsonl")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="csp")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(CSPConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(
    cfg: CSPConfig, *, runs_dir: Path, processed_dir: Path = paths.PROCESSED_DIR
) -> Path:
    """Compute every job not done yet, then write the run's tables.

    `processed_dir` is where `publish` writes the phase field's table.
    """
    prediction = cfg.csp
    cells = _read_cells(cfg)
    mlip = MLIPFactory.from_config(prediction.mlip)
    potential = mlip.describe()
    searches = [
        StructureSearchFactory.from_config(c) for c in prediction.search.values()
    ]
    if not searches:
        raise ConfigError(
            "no structure search chosen; set csp/search=[relax,basin_hop]"
        )
    device = resolve_device(prediction.device) if mlip.uses_device else "cpu"
    if prediction.workers > 1 and device == "cuda":
        raise ConfigError(
            f"csp.workers is {prediction.workers} but the MLIP runs on cuda, which "
            "takes one process; set csp.workers=1, or csp.device=cpu"
        )
    structures = _read_references(cfg, cells)
    _refuse_other_settings(cfg, runs_dir / cfg.run_name)
    run_dir, _ = _launch.start(cfg, runs_dir=runs_dir, job_type="csp", resume=True)
    settings = _Settings(
        run_dir=run_dir,
        mlip=mlip,
        device=device,
        potential=potential,
        seed=cfg.seed,
        log_level=cfg.log_level,
        log_format=cfg.log_format,
    )
    known = _relax_references(structures, settings=settings, cfg=cfg)
    done = {result.key for result in store.read_results(run_dir)}
    jobs = job.plan(cells, prediction.frameworks, searches, mlip=mlip.kind, done=done)
    click.echo(f"{len(jobs)} jobs to compute, {len(done)} done before")
    recipe = job.Recipe(
        phi=prediction.phi, c_over_a=prediction.c_over_a, symprec=prediction.symprec
    )
    _compute_all(jobs, settings=settings, recipe=recipe, workers=prediction.workers)
    results = store.read_results(run_dir)
    store.write_lines(run_dir / "results.jsonl", (r.to_json() for r in results))
    store.write_lines(run_dir / "references.jsonl", (r.to_json() for r in known))
    if not known:
        click.echo("no table: set csp.references.directory to CIFs of the elements")
        return run_dir
    rows = tabulate(results, known, source=str(run_dir / "results.jsonl"))
    data_store.write(
        run_dir / "table.csv",
        rows,
        raw=[run_dir / name for name in _PINNED],
        command=f"phasebo-csp run_name={run_dir.name}",
    )
    click.echo(f"{len(rows)} rows written to {run_dir / 'table.csv'}")
    if cfg.publish is not None:
        path = _publish(run_dir, rows, name=cfg.publish, processed_dir=processed_dir)
        click.echo(f"published to {path}; name it in configs/system/{cfg.publish}.yaml")
    return run_dir


def tabulate(
    results: Sequence[Result], known: Sequence[Reference], *, source: str
) -> DataFrame[ProcessedTable]:
    """Return the processed table: each result's released energy, and the references.

    Every reference appears at every fidelity, so each fidelity has its hull.
    """
    fidelities = sorted({result.fidelity for result in results})
    parts = []
    for fidelity in fidelities:
        found = [result for result in results if result.fidelity == fidelity]
        parts.append(
            table.make_rows(
                [result.formula for result in found],
                fidelity=fidelity,
                reference=[False] * len(found),
                total_energy_ev=[result.released_energy_ev for result in found],
                framework=[result.framework for result in found],
            )
        )
        parts.append(
            table.make_rows(
                [reference.formula for reference in known],
                fidelity=fidelity,
                reference=[True] * len(known),
                total_energy_ev=[reference.energy_ev for reference in known],
                framework=[None] * len(known),
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    elements = sorted(
        {str(el) for formula in rows["formula"] for el in Composition(formula)}
    )
    cleaned, _ = table.clean(rows, elements=elements, source=source)
    return cleaned


@frozen
class _Settings:
    """What every job of a run shares: where it writes and the MLIP it runs."""

    run_dir: Path
    mlip: MLIP[Any]
    device: Literal["cpu", "cuda"]
    potential: Potential
    seed: int
    log_level: LogLevel
    log_format: LogFormat


def _read_cells(cfg: CSPConfig) -> list[str]:
    """Return the configured cells, each once, written as `pymatgen` orders them."""
    listed = _launch.read_formulas(cfg.formulas_file, key="formulas_file") or []
    cells = list(
        dict.fromkeys(
            framework.name_cell(framework.read_counts(f))
            for f in [*cfg.formulas, *listed]
        )
    )
    if not cells:
        raise ConfigError("no formulas; set formulas=[...] or formulas_file")
    for cell in cells:
        counts = framework.read_counts(cell)
        framework.count_anion_sites(counts, framework.oxidation_states(counts))
    return cells


def _read_references(cfg: CSPConfig, cells: Sequence[str]) -> dict[str, Atoms]:
    """Return the reference structures by name; fail if an element has none."""
    directory = cfg.csp.references.directory
    if directory is None:
        return {}
    structures = store.read_structures(paths.ROOT / directory)
    formulas = [atoms.get_chemical_formula() for atoms in structures.values()]
    if missing := references.find_missing_elements(formulas, cells):
        raise DataError(
            f"no reference of {', '.join(missing)} alone in {directory}; add a CIF "
            "of each, so the hull has every corner"
        )
    return structures


def _refuse_other_settings(cfg: CSPConfig, run_dir: Path) -> None:
    """Raise `ConfigError` if `run_dir` holds a run started with other settings."""
    path = run_dir / "config.yaml"
    if not path.is_file():
        return
    before = _flatten(OmegaConf.to_container(OmegaConf.load(path)))
    now = _flatten(cfg.model_dump(mode="json"))
    changed = sorted(
        key
        for key in before.keys() | now.keys()
        if before.get(key) != now.get(key) and not key.startswith(_FREE_ON_RESUME)
    )
    if changed:
        raise ConfigError(
            f"{run_dir.name} was started with other settings ({', '.join(changed)}); "
            "pass the same ones to resume it, or another run_name"
        )


def _flatten(data: object, *, prefix: str = "") -> dict[str, object]:
    """Return the leaves of nested mappings by their dotted keys."""
    if not isinstance(data, dict) or not data:
        return {prefix: data}
    flat: dict[str, object] = {}
    for key, value in data.items():
        flat |= _flatten(value, prefix=f"{prefix}.{key}" if prefix else str(key))
    return flat


def _relax_references(
    structures: dict[str, Atoms], *, settings: _Settings, cfg: CSPConfig
) -> list[Reference]:
    """Return every reference relaxed, relaxing only those the run lacks."""
    done = {reference.name for reference in store.read_references(settings.run_dir)}
    missing = [name for name in structures if name not in done]
    if missing:
        calculator = settings.mlip.calculator(device=settings.device)
        for name in missing:
            reference = references.relax_reference(
                name,
                structures[name],
                calculator,
                potential=settings.potential,
                fmax=cfg.csp.references.fmax,
                steps=cfg.csp.references.steps,
            )
            store.write_reference(settings.run_dir, reference)
            click.echo(
                f"reference {name} ({reference.formula}): "
                f"{reference.energy_ev / reference.n_atoms:.4f} eV/atom"
            )
    return [r for r in store.read_references(settings.run_dir) if r.name in structures]


def _compute_all(
    jobs: Sequence[job.Job], *, settings: _Settings, recipe: job.Recipe, workers: int
) -> None:
    """Compute the jobs, in this process or split over `workers` processes."""
    if not jobs:
        return
    if workers == 1:
        _compute_share(jobs, settings=settings, recipe=recipe)
        return
    from joblib import Parallel, delayed

    shares = [jobs[i::workers] for i in range(workers)]
    Parallel(n_jobs=workers)(
        delayed(_compute_in_worker)(share, settings=settings, recipe=recipe)
        for share in shares
        if share
    )


def _compute_in_worker(
    jobs: Sequence[job.Job], *, settings: _Settings, recipe: job.Recipe
) -> None:
    """Compute a share of the jobs in a worker process, logging as the run does."""
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    _compute_share(jobs, settings=settings, recipe=recipe)


def _compute_share(
    jobs: Sequence[job.Job], *, settings: _Settings, recipe: job.Recipe
) -> None:
    """Compute jobs with one calculator, writing each result as it finishes."""
    calculator = settings.mlip.calculator(device=settings.device)
    for each in jobs:
        result = job.compute(
            each,
            calculator=calculator,
            potential=settings.potential,
            recipe=recipe,
            rng=job.seed(settings.seed, each),
        )
        store.write_result(settings.run_dir, result)
        click.echo(
            f"{result.formula} {result.framework} {result.fidelity}: "
            f"{result.released_energy_ev / result.n_atoms:.4f} eV/atom released, "
            f"{result.framework_energy_ev / result.n_atoms:.4f} on the framework, "
            f"drift {result.drift_angstrom:.2f} A, {result.elapsed_s:.0f} s"
        )


def _relativize(path: Path) -> Path:
    return path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path


def _publish(
    run_dir: Path,
    rows: DataFrame[ProcessedTable],
    *,
    name: str,
    processed_dir: Path,
) -> Path:
    """Write the table as the phase field `name`, with copies of the files it pins."""
    pinned = processed_dir / name
    pinned.mkdir(parents=True, exist_ok=True)
    raw = []
    for file in _PINNED:
        shutil.copyfile(run_dir / file, pinned / file)
        raw.append(pinned / file)
    path = processed_dir / f"{name}.csv"
    data_store.write(
        path,
        rows,
        raw=raw,
        # The run directory is not committed, so the remedy names the settings instead.
        command=(
            f"phasebo-csp run_name={run_dir.name} publish={name} with the settings in "
            f"{_relativize(pinned / 'config.yaml')}"
        ),
    )
    return path
