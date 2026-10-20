"""Processed tables on disk, each with a sidecar that pins its raw files and script."""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from attrs import field, frozen
from pandera.typing import DataFrame
from pydantic import ValidationError

from nuikit.config import Schema
from nuikit.errors import DataError

from phasebo import paths, table
from phasebo.table import ProcessedTable


class RawFile(Schema):
    """A raw file a processed table was built from."""

    path: Path
    sha256: str


class Sidecar(Schema):
    """What a processed table was built from: its script, and every file's sha256."""

    script: Path
    table_sha256: str
    raw: list[RawFile]


@frozen
class Processed:
    """A processed table, where it lives, and the sidecar that pins it."""

    path: Path
    rows: DataFrame[ProcessedTable] = field(eq=False)
    sidecar: Sidecar


def write(
    path: Path, rows: DataFrame[ProcessedTable], *, raw: Sequence[Path], script: Path
) -> None:
    """Write the table, and a sidecar with its script and every file's sha256."""
    text = table.render(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    sidecar = Sidecar(
        script=_shown(script),
        table_sha256=_sha256(text.encode()),
        raw=[RawFile(path=_shown(p), sha256=_sha256(p.read_bytes())) for p in raw],
    )
    _sidecar(path).write_text(
        sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def find_problems(path: Path) -> list[str]:
    """Return what makes the table at `path` untrustworthy, with what to rerun."""
    if not path.is_file():
        return ["missing; run the script in scripts/ that writes it"]
    if not _sidecar(path).is_file():
        return [f"no sidecar {_sidecar(path).name}; rerun the script that writes it"]
    try:
        sidecar = _read_sidecar(path)
    except ValidationError:
        return [f"sidecar {_sidecar(path).name} is malformed; rerun its script"]
    rerun = f"rerun `uv run {sidecar.script}`"
    found: list[str] = []
    if _sha256(path.read_bytes()) != sidecar.table_sha256:
        found.append(f"edited since its script wrote it; {rerun}")
    for raw in sidecar.raw:
        source = paths.ROOT / raw.path
        if not source.is_file():
            found.append(f"raw file {raw.path} is missing; restore it")
        elif _sha256(source.read_bytes()) != raw.sha256:
            found.append(f"raw file {raw.path} changed; {rerun}")
    if not found:
        try:
            table.parse(path.read_text(encoding="utf-8"), source=str(path))
        except DataError as e:
            found.append(str(e))
    return found


def load(path: Path) -> Processed:
    """Return the table at `path` and its sidecar; fail if it is untrustworthy."""
    if found := find_problems(path):
        raise DataError(f"{_shown(path)}: {found[0]}")
    rows = table.parse(path.read_text(encoding="utf-8"), source=str(path))
    return Processed(_shown(path), rows, _read_sidecar(path))


def _read_sidecar(path: Path) -> Sidecar:
    return Sidecar.model_validate_json(_sidecar(path).read_text(encoding="utf-8"))


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shown(path: Path) -> Path:
    return path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path
