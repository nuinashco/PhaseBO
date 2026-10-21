"""Draw the figures of a `phasebo-suggest` or `phasebo-benchmark` run.

    uv run --group notebooks scripts/plot_run.py runs/<run_name>

Writes into the run directory: convex_hull.png, the energy above the hull over a
4-element field's square, and for a benchmark run one <metric>.png per metric with
curves in metrics.json, random picking's dashed.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import rootutils
from omegaconf import OmegaConf
from pymatgen.core import Composition

from nuikit.errors import ConfigError, DataError, cli_boundary

from phasebo import paths
from phasebo.config import SystemConfig
from phasebo.data import store
from phasebo.field.points import Points

rootutils.setup_root(__file__, indicator=".project-root")


@click.command(help=__doc__)
@click.argument(
    "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--dpi", default=150, show_default=True)
@cli_boundary
def main(run_dir: Path, *, dpi: int) -> None:
    config = _read_config(run_dir)
    system = SystemConfig.model_validate(config["system"])
    elements = list(system.elements)
    loaded = store.load(paths.ROOT / system.processed)
    ran_on = json.loads((run_dir / "data.json").read_text(encoding="utf-8"))
    if ran_on["table_sha256"] != loaded.sidecar.table_sha256:
        raise DataError(
            f"{system.processed} changed since this run; check out the run's commit "
            "or rerun the command"
        )
    excluded = system.candidates.exclude
    points = Points.from_rows(
        loaded.rows,
        fidelity=config["fidelity"],
        elements=elements,
        excluded=[] if excluded is None else _read_formulas(excluded),
    )
    if len(elements) == 4:
        _save_hull_map(points, run_dir / "convex_hull.png", elements=elements, dpi=dpi)
    else:
        click.echo(f"no hull map: {len(elements)} elements, not 4")
    metrics_file = run_dir / "metrics.json"
    if metrics_file.is_file():
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        evaluations = metrics.pop("evaluations")
        for kind, score in metrics.items():
            if score["curves"]:
                path = run_dir / f"{kind}.png"
                _save_curves(
                    score["curves"],
                    path,
                    evaluations=evaluations,
                    title=kind,
                    dpi=dpi,
                )


def _read_config(run_dir: Path) -> Any:
    if not (run_dir / "config.yaml").is_file() or not (run_dir / "data.json").is_file():
        raise ConfigError(
            f"no config.yaml and data.json in {run_dir}; pass a run directory"
        )
    config = OmegaConf.to_container(OmegaConf.load(run_dir / "config.yaml"))
    if not isinstance(config, dict) or "fidelity" not in config:
        raise ConfigError(f"{run_dir} is not a suggest or benchmark run; pass one")
    return config


def _save_hull_map(
    points: Points, path: Path, *, elements: Sequence[str], dpi: int
) -> None:
    """Save the energy above the hull over the square of a 4-element field.

    The square's axes are the fractions of the first element among the first two,
    and of the third among the last two, in config order.
    """
    import matplotlib.pyplot as plt
    from scipy import interpolate

    square = _place_on_square(points.formulas, elements)
    grid = np.arange(0, 1, 0.005)
    surface_mev = interpolate.LinearNDInterpolator(square, points.energies_mev)
    figure, axes = plt.subplots()
    filled = axes.contourf(
        grid, grid, surface_mev(*np.meshgrid(grid, grid)), 50, cmap="gist_heat"
    )
    figure.colorbar(filled, ax=axes).set_label(
        "Energy above hull (meV/atom)", fontsize=14
    )
    axes.scatter(*square.T, c="lime", marker="3", lw=1.5, label="Computed compositions")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    a, b, c, d = elements
    axes.set_xlabel(f"{a}/({a}+{b})", fontsize=14)
    axes.set_ylabel(f"{c}/({c}+{d})", fontsize=14)
    axes.legend(bbox_to_anchor=(0.5, 1.1), fontsize=10)
    _save(figure, path, dpi=dpi)


def _save_curves(
    curves: dict[str, list[float]],
    path: Path,
    *,
    evaluations: Sequence[int],
    title: str,
    dpi: int,
) -> None:
    """Save every curve of one metric over the evaluations after the start."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    for name, values in curves.items():
        # A curve named `.../random` or `random` is random picking's exact value.
        style = "--" if name.split("/")[-1] == "random" else "-"
        axes.plot(evaluations, values, style, marker="o", ms=3, label=name)
    axes.set_xlabel("Evaluations after the start")
    axes.set_title(title)
    axes.legend(fontsize=8)
    _save(figure, path, dpi=dpi)


def _read_formulas(path: Path) -> list[str]:
    if not (paths.ROOT / path).is_file():
        raise DataError(
            f"no formula file at {path}, the run's system.candidates.exclude; "
            "restore it"
        )
    return [str(formula) for formula in pd.read_csv(paths.ROOT / path).iloc[:, 0]]


def _place_on_square(formulas: Sequence[str], elements: Sequence[str]) -> np.ndarray:
    a, b, c, d = elements
    rows = []
    for formula in formulas:
        amount = Composition(formula).get_el_amt_dict()
        first, second = amount.get(a, 0.0), amount.get(b, 0.0)
        third, fourth = amount.get(c, 0.0), amount.get(d, 0.0)
        # A cell missing both elements of a pair sits at 1, as the original plot put it.
        x = first / (first + second) if first + second else 1.0
        y = third / (third + fourth) if third + fourth else 1.0
        rows.append((x, y))
    return np.array(rows)


def _save(figure: Any, path: Path, *, dpi: int) -> None:
    import matplotlib.pyplot as plt

    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    click.echo(f"wrote {path}")


if __name__ == "__main__":
    main()
