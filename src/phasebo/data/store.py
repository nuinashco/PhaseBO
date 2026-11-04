"""Processed tables on disk, each with a sidecar that pins its source and raw files.

A script's table pins the script; a table a command computed, such as phasebo-csp's,
pins its run's config and results as raw files, and names the command that rebuilds
it.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Self

from attrs import field, frozen
from pandera.typing import DataFrame
from pydantic import ValidationError, field_serializer, model_validator

from nuikit.config import Schema
from nuikit.errors import DataError

from phasebo import paths
from phasebo.data import table
from phasebo.data.table import ProcessedTable


class PinnedFile(Schema):
    """A file and its sha256 when the table was written."""

    path: Path
    sha256: str

    @field_serializer("path")
    def _posix(self, path: Path) -> str:
        # Forward slashes on every platform, so a sidecar written on Windows reads
        # the same on macOS and Linux.
        return path.as_posix()


class Sidecar(Schema):
    """What a processed table was built from, by sha256."""

    script: PinnedFile | None = None  # None for a table a command computed.
    command: str | None = None  # What rebuilds a computed table; None for a script's.
    table_sha256: str
    raw: list[PinnedFile]

    @model_validator(mode="after")
    def _names_one_source(self) -> Self:
        if (self.script is None) == (self.command is None):
            msg = "a sidecar names its script or its command, not both or neither"
            raise ValueError(msg)
        return self

    @property
    def rerun(self) -> str:
        """Return the remedy for a stale table: rerun what wrote it."""
        if self.script is not None:
            return f"rerun `uv run {self.script.path}`"
        return f"rerun `{self.command}`"

    @property
    def pinned(self) -> list[PinnedFile]:
        """Return every file the table depends on: its script, if any, and raw files."""
        return [*([] if self.script is None else [self.script]), *self.raw]


@frozen
class LoadedTable:
    """A processed table, where it lives, and the sidecar that pins it."""

    path: Path
    rows: DataFrame[ProcessedTable] = field(eq=False)
    sidecar: Sidecar


def write(
    path: Path,
    rows: DataFrame[ProcessedTable],
    *,
    raw: Sequence[Path],
    script: Path | None = None,
    command: str | None = None,
) -> None:
    """Write the table, and a sidecar pinning it, its source and raw files by sha256.

    Give the `script` that wrote the table, or the `command` that rebuilds it.
    """
    text = table.render(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": the bytes the sidecar hashes, also on Windows, which would write
    # CRLF and leave the table stale as soon as it was written.
    path.write_text(text, encoding="utf-8", newline="\n")
    sidecar = Sidecar(
        script=None if script is None else _pin(script),
        command=command,
        table_sha256=_hash(text.encode()),
        raw=[_pin(p) for p in raw],
    )
    # Without the unset source, a script's sidecar reads as it did before commands.
    _locate_sidecar(path).write_text(
        sidecar.model_dump_json(indent=2, exclude_none=True) + "\n",
        encoding="utf-8",
        newline="\n",
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
        return [
            f"sidecar {sidecar_path.name} is malformed; "
            "restore it, or rerun what wrote it"
        ]
    rerun = sidecar.rerun
    found: list[str] = []
    if _hash(path.read_bytes()) != sidecar.table_sha256:
        found.append(f"edited since it was written; {rerun}")
    for pinned in sidecar.pinned:
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
