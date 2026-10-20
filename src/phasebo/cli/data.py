"""`phasebo-data`: check that every phase field's processed table matches its raw files.

Usage:

    phasebo-data

Each table in data/processed/ is written by a script in scripts/, with a sidecar that
records the sha256 of the table and of its raw files. A missing table, a table edited
by hand or a raw file changed since its script ran fails the check, naming the script
to rerun.
"""

import click
import rootutils
from hydra import compose, initialize_config_dir

from nuikit.errors import DataError, cli_boundary

from phasebo import paths
from phasebo.config import DataConfig, SystemConfig
from phasebo.data import store

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@click.command(help=__doc__)
@cli_boundary
def main() -> None:
    failed: list[str] = []
    for system in compose_systems():
        if found := store.find_problems(paths.ROOT / system.processed):
            failed += [f"{system.processed}: {problem}" for problem in found]
        else:
            click.echo(f"ok {system.processed}")
    if failed:
        raise DataError("untrustworthy processed data: " + "; ".join(failed))


def compose_systems() -> list[SystemConfig]:
    """Return every phase field in configs/system/, composed as Hydra would."""
    names = sorted(path.stem for path in (paths.CONFIGS_DIR / "system").glob("*.yaml"))
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        return [
            DataConfig.from_hydra(compose("data", [f"system={name}"])).system
            for name in names
        ]
