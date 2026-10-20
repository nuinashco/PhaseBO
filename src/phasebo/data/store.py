"""Processed tables on disk, each with a sidecar that pins its script and raw files."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from attrs import field, frozen
from pandera.typing import DataFrame
from pydantic import ValidationError

from nuikit.config import Schema
from nuikit.errors import DataError

from phasebo import paths
from phasebo.data import table
from phasebo.data.table import ProcessedTable


class PinnedFile(Schema):
    """A file and its sha256 when the table was written."""

    path: Path
    sha256: str


class Sidecar(Schema):
    """What a processed table was built from, by sha256."""

    script: PinnedFile
    table_sha256: str
    raw: list[PinnedFile]


@frozen
class LoadedTable:
    """A processed table, where it lives, and the sidecar that pins it."""

    path: Path
    rows: DataFrame[ProcessedTable] = field(eq=False)
    sidecar: Sidecar


def write(
    path: Path, rows: DataFrame[ProcessedTable], *, raw: Sequence[Path], script: Path
) -> None:
    """Write the table, and a sidecar pinning it, its script and raw files by sha256."""
    text = table.render(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    sidecar = Sidecar(
        script=_pin(script),
        table_sha256=_hash(text.encode()),
        raw=[_pin(p) for p in raw],
    )
    _locate_sidecar(path).write_text(
        sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def find_problems(path: Path) -> list[str]:
    """Return what makes the table at `path` untrustworthy, with what to rerun."""
    sidecar_path = _locate_sidecar(path)
    if not path.is_file():
        return ["missing; run the script in scripts/ that writes it"]
    if not sidecar_path.is_file():
        return [f"no sidecar {sidecar_path.name}; rerun the script that writes it"]
    try:
        sidecar = _read_sidecar(path)
    except ValidationError:
        return [f"sidecar {sidecar_path.name} is malformed; rerun its script"]
    rerun = f"rerun `uv run {sidecar.script.path}`"
    found: list[str] = []
    if _hash(path.read_bytes()) != sidecar.table_sha256:
        found.append(f"edited since its script wrote it; {rerun}")
    for pinned in [sidecar.script, *sidecar.raw]:
        source = paths.ROOT / pinned.path
        if not source.is_file():
            found.append(f"{pinned.path} is missing; restore it")
        elif _hash(source.read_bytes()) != pinned.sha256:
            found.append(f"{pinned.path} changed since the table was written; {rerun}")
    if not found:
        try:
            table.parse(path.read_text(encoding="utf-8"), source=str(path))
        except DataError as e:
            found.append(str(e))
    return found


def load(path: Path) -> LoadedTable:
    """Return the table at `path` and its sidecar; fail if it is untrustworthy."""
    if found := find_problems(path):
        raise DataError(f"{_relativize(path)}: {found[0]}")
    rows = table.parse(path.read_text(encoding="utf-8"), source=str(path))
    return LoadedTable(_relativize(path), rows, _read_sidecar(path))


def _read_sidecar(path: Path) -> Sidecar:
    text = _locate_sidecar(path).read_text(encoding="utf-8")
    return Sidecar.model_validate_json(text)


def _pin(path: Path) -> PinnedFile:
    return PinnedFile(path=_relativize(path), sha256=_hash(path.read_bytes()))


def _locate_sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relativize(path: Path) -> Path:
    return path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path
