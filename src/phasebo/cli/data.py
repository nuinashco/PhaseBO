"""`phasebo-data`: rebuild data/processed/ from data/raw/, one table per phase field.

Usage:

    phasebo-data            # rewrite stale tables

    phasebo-data --check    # fail instead, listing the stale tables

Each file in configs/system/ names its raw dataset and its processed table. Rows with
elements outside the phase field are dropped; the raw files stay as published.
"""

import click
import rootutils
from hydra import compose, initialize_config_dir

from nuikit.errors import DataError, cli_boundary

from phasebo import paths
from phasebo.config import DataConfig, SystemConfig
from phasebo.dataset import table
from phasebo.dataset.base import DatasetFactory

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@click.command(help=__doc__)
@click.option("--check", is_flag=True, help="Fail if any table is stale.")
@cli_boundary
def main(*, check: bool) -> None:
    stale = find_stale(compose_systems())
    if check:
        if stale:
            names = ", ".join(str(system.processed) for system, _, _ in stale)
            raise DataError(f"stale processed data: {names}; run `make data`")
        return
    for system, text, dropped in stale:
        path = paths.ROOT / system.processed
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        rows = text.count("\n") - 1
        click.echo(
            f"wrote {system.processed}: {rows} rows, {dropped} off-field rows dropped"
        )


def compose_systems() -> list[SystemConfig]:
    """Return every phase field in configs/system/, composed as Hydra would."""
    names = sorted(path.stem for path in (paths.CONFIGS_DIR / "system").glob("*.yaml"))
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        return [
            DataConfig.from_hydra(compose("data", [f"system={name}"])).system
            for name in names
        ]


def find_stale(
    systems: list[SystemConfig],
) -> list[tuple[SystemConfig, str, int]]:
    """Return each phase field whose table differs from a rebuild, with the rebuild."""
    stale: list[tuple[SystemConfig, str, int]] = []
    for system in systems:
        text, dropped = build(system)
        path = paths.ROOT / system.processed
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append((system, text, dropped))
    return stale


def build(system: SystemConfig) -> tuple[str, int]:
    """Return a phase field's processed table as CSV text, and its off-field rows."""
    dataset = DatasetFactory.from_config(system.dataset)
    rows, dropped = table.clean(dataset.read(), elements=system.elements)
    return table.render(rows), dropped
