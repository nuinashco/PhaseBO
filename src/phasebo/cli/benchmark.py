"""`phasebo-benchmark`: replay BO over the computed compositions and score it.

Usage:

    phasebo-benchmark                                    # configs/benchmark.yaml

    phasebo-benchmark bo/acquisition=ts benchmark.repeats=100

    phasebo-benchmark bo/acquisition=random              # check: scores ~random

    phasebo-benchmark benchmark/metric=[steps_to_hit,hazard_ratio]

Replays BO from `benchmark.repeats` starting sets, in parallel, and writes
runs/<run_name>/ with paths.parquet (every evaluation of every repeat) and
metrics.json (each metric next to random picking's exact value, and its curves
over evaluations); `tracker=wandb` also logs them. Prints one line per metric.
With `benchmark/cheap`, every game knows each point's cheap energy from the start;
the budget and the metrics still count DFT evaluations.
"""

import functools
import json
from pathlib import Path

import click
import hydra
import rootutils
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.benchmark import replay
from phasebo.benchmark.cheap.base import CheapFactory
from phasebo.benchmark.metric.base import MetricFactory, Score, to_json, to_wandb
from phasebo.benchmark.start.base import StartFactory
from phasebo.bo.acquisition.base import AcquisitionFactory
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
    """Replay BO from many starting sets and score it against random picking."""
    loaded, points, _ = _launch.load_points(cfg)
    protocol = cfg.benchmark
    starting = StartFactory.from_config(protocol.start)
    starting.check(points, budget=cfg.bo.batch_size * protocol.max_iter)
    surrogate = SurrogateFactory.from_config(cfg.bo.surrogate)
    acquisition = AcquisitionFactory.from_config(cfg.bo.acquisition)
    draw_cheap = None
    if protocol.cheap is not None:
        cheap = CheapFactory.from_config(protocol.cheap)
        cheap.check(loaded.rows, points, fidelity=cfg.fidelity)
        draw_cheap = functools.partial(
            cheap.draw, loaded.rows, points, fidelity=cfg.fidelity
        )
    metrics = {
        kind: MetricFactory.from_config(config)
        for kind, config in protocol.metric.items()
    }
    tags = _launch.tag_families(
        surrogate=cfg.bo.surrogate,
        acquisition=cfg.bo.acquisition,
        start=protocol.start,
        **({} if protocol.cheap is None else {"cheap": protocol.cheap}),
    )
    run_dir, _ = _launch.start(
        cfg,
        runs_dir=runs_dir,
        job_type="benchmark",
        group=_launch.name_field(cfg.system),
        tags=tags,
    )
    _launch.write_data(run_dir, loaded)
    replays = replay.replay_many(
        points,
        start=starting,
        surrogate=surrogate,
        acquisition=acquisition,
        batch_size=cfg.bo.batch_size,
        max_iter=protocol.max_iter,
        repeats=protocol.repeats,
        seed=cfg.seed,
        workers=protocol.workers,
        draw_cheap=draw_cheap,
    )
    replays.table().to_parquet(run_dir / "paths.parquet", index=False)
    scores = {kind: metric.measure(replays) for kind, metric in metrics.items()}
    evaluations = replays.evaluations.tolist()
    text = json.dumps(to_json(scores, evaluations), indent=2, ensure_ascii=False)
    (run_dir / "metrics.json").write_text(text + "\n", encoding="utf-8")
    if cfg.tracker == "wandb":
        _log_to_wandb(scores, evaluations, run_dir=run_dir)
    for score in scores.values():
        click.echo(score.headline)
    return run_dir


def _log_to_wandb(
    scores: dict[str, Score], evaluations: list[int], *, run_dir: Path
) -> None:
    """Log curves over evaluations, values as the summary, and the outputs."""
    import wandb

    summary, rows = to_wandb(scores, evaluations)
    # W&B otherwise copies each curve's last point into the summary, beside the values.
    wandb.define_metric("evaluations", summary="none")
    for key in {key for row in rows for key in row}:
        wandb.define_metric(key, step_metric="evaluations", summary="none")
    for evaluation, row in zip(evaluations, rows, strict=True):
        wandb.log({"evaluations": evaluation, **row})
    wandb.summary.update(summary)
    run = wandb.run
    if run is not None:
        outputs = wandb.Artifact(f"replays-{run.id}", type="replays")
        for name in ("paths.parquet", "metrics.json", "config.yaml", "data.json"):
            outputs.add_file(str(run_dir / name))
        run.log_artifact(outputs)
