"""`phasebo-compare`: how benchmark runs differ from a baseline run, game by game.

Usage:

    phasebo-compare 'runs=[random-2026-10-21_13-42-13,qlogei-2026-10-21_13-41-18]'

    phasebo-compare 'wandb=[tb44y6ic,c2hf25m5,gdpiqhh3]' tracker=wandb

The first run listed (`runs` before `wandb`) is the baseline, typically one of
`bo/acquisition=random`; every other run is compared with it. Runs must share their
starting sets, as runs of one seed, start and phase field do, so repeat r of each is
the same game. Writes runs/<run_name>/comparison.json and prints, per run and
outcome, the difference (or ratio) from the baseline with its interval and how often
the run did better and worse; `tracker=wandb` also logs the table.
"""

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import click
import hydra
import rootutils
from omegaconf import DictConfig

from nuikit.errors import ConfigError, DataError, cli_boundary

from phasebo import paths
from phasebo.benchmark import comparison
from phasebo.benchmark.metric.base import Contrast, MetricFactory
from phasebo.cli import _launch
from phasebo.config import CompareConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")

_NUMBER = {"difference": "{:+.2f}", "ratio": "x{:.2f}"}


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="compare")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(CompareConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: CompareConfig, *, runs_dir: Path) -> Path:
    """Compare every listed run with the first, repeat by repeat."""
    if len(cfg.runs) + len(cfg.wandb) < 2:
        raise ConfigError(
            "a comparison needs two runs or more; list them in runs or wandb"
        )
    with tempfile.TemporaryDirectory() as downloads:
        sources = [(name, runs_dir / name, f"runs/{name}") for name in cfg.runs]
        sources += _download_benchmarks(
            cfg.wandb, project=cfg.wandb_project, into=Path(downloads)
        )
        repeated = [name for name, n in Counter(s[0] for s in sources).items() if n > 1]
        if repeated:
            raise ConfigError(f"{repeated[0]} is listed twice; list each run once")
        loaded = {name: _launch.load_benchmark(path) for name, path, _ in sources}
    (base_name, (base_cfg, base_table, base)), *others = loaded.items()
    runs = {name: replays for name, (_, _, replays) in others}
    for name, replays in runs.items():
        comparison.check_paired(base, replays, name=name)
    metrics = {
        kind: MetricFactory.from_config(config)
        for kind, config in base_cfg.benchmark.metric.items()
    }
    run_dir, rng = _launch.start(
        cfg,
        runs_dir=runs_dir,
        job_type="compare",
        group=_launch.name_field(base_cfg.system),
        tags=[f"baseline={base_name}"],
    )
    _launch.write_data(run_dir, base_table)
    contrasts = comparison.compare(
        base,
        runs,
        metrics=metrics,
        level=cfg.level,
        resamples=cfg.resamples,
        rng=rng,
    )
    rows = [
        {"run": name, "outcome": outcome, **_describe(contrast)}
        for name, outcomes in contrasts.items()
        for outcome, contrast in outcomes.items()
    ]
    report = {
        "baseline": base_name,
        "sources": {name: origin for name, _, origin in sources},
        "level": cfg.level,
        "rows": rows,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (run_dir / "comparison.json").write_text(text + "\n", encoding="utf-8")
    if cfg.tracker == "wandb":
        _log_to_wandb(rows)
    click.echo(f"baseline: {base_name}")
    for name, outcomes in contrasts.items():
        click.echo(name)
        for outcome, contrast in outcomes.items():
            click.echo(f"  {outcome:<24} {_format(contrast, level=cfg.level)}")
    return run_dir


def _download_benchmarks(
    run_ids: list[str], *, project: str, into: Path
) -> list[tuple[str, Path, str]]:
    """Download W&B benchmark runs' outputs: each run's name, directory and artifact."""
    if not run_ids:
        return []
    import wandb
    from wandb.errors import CommError

    api = wandb.Api()
    path = f"{api.default_entity}/{project}"
    downloaded = []
    for run_id in run_ids:
        try:
            name = api.run(f"{path}/{run_id}").name
            artifact = api.artifact(f"{path}/replays-{run_id}:latest")
        # A missing run or artifact raises CommError; a malformed id, ValueError.
        except (CommError, ValueError) as e:
            raise DataError(
                f"no benchmark outputs for W&B run {run_id} in {path}; pass the id of "
                "a phasebo-benchmark run logged with tracker=wandb"
            ) from e
        directory = Path(artifact.download(root=str(into / run_id)))
        downloaded.append((name, directory, f"wandb {path}/{artifact.name}"))
    return downloaded


def _describe(contrast: Contrast) -> dict[str, object]:
    return {
        "scale": contrast.scale,
        "estimate": contrast.estimate,
        "low": contrast.low,
        "high": contrast.high,
        "better": contrast.better,
        "worse": contrast.worse,
    }


def _format(contrast: Contrast, *, level: float) -> str:
    number = _NUMBER[contrast.scale].format
    value = f"{number(contrast.estimate)}  ({level:.0%}: {number(contrast.low)} to "
    value += f"{number(contrast.high)})"
    if contrast.better is not None and contrast.worse is not None:
        value += f"  better in {contrast.better:.0%}, worse in {contrast.worse:.0%}"
    return value


def _log_to_wandb(rows: list[dict[str, object]]) -> None:
    import wandb

    columns: list[Any] = list(rows[0]) if rows else ["run", "outcome"]
    table = wandb.Table(columns=columns, data=[list(row.values()) for row in rows])
    wandb.log({"comparison": table})
