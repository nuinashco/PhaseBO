"""Processed tables on disk, each with a sidecar that pins its raw files and script."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from attrs import field, frozen

from nuikit.errors import DataError

from phasebo import paths, table

type Provenance = dict[str, object]


@frozen
class Processed:
    """A processed table and the provenance its sidecar records."""

    rows: pd.DataFrame = field(eq=False)
    provenance: Provenance


def write(path: Path, rows: pd.DataFrame, *, raw: Sequence[Path], script: Path) -> None:
    """Write the table, and a sidecar with its script and every file's sha256."""
    text = table.render(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    sidecar = {
        "script": _shown(script),
        "table_sha256": _sha256(text.encode()),
        "raw": [{"path": _shown(p), "sha256": _sha256(p.read_bytes())} for p in raw],
    }
    text = json.dumps(sidecar, indent=2, ensure_ascii=False)
    _sidecar(path).write_text(text + "\n", encoding="utf-8")


def find_problems(path: Path) -> list[str]:
    """Return what makes the table at `path` untrustworthy, with what to rerun."""
    if not path.is_file():
        return ["missing; run the script in scripts/ that writes it"]
    if not _sidecar(path).is_file():
        return [f"no sidecar {_sidecar(path).name}; rerun the script that writes it"]
    sidecar = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    rerun = f"rerun `uv run {sidecar['script']}`"
    found: list[str] = []
    if _sha256(path.read_bytes()) != sidecar["table_sha256"]:
        found.append(f"edited since its script wrote it; {rerun}")
    for raw in sidecar["raw"]:
        source = paths.ROOT / raw["path"]
        if not source.is_file():
            found.append(f"raw file {raw['path']} is missing; restore it")
        elif _sha256(source.read_bytes()) != raw["sha256"]:
            found.append(f"raw file {raw['path']} changed; {rerun}")
    if not found:
        try:
            table.parse(path.read_text(encoding="utf-8"), source=str(path))
        except DataError as e:
            found.append(str(e))
    return found


def load(path: Path) -> Processed:
    """Return the table at `path` and its provenance; fail if it is untrustworthy."""
    if found := find_problems(path):
        raise DataError(f"{_shown(path)}: {found[0]}")
    text = path.read_text(encoding="utf-8")
    sidecar = json.loads(_sidecar(path).read_text(encoding="utf-8"))
    provenance: Provenance = {
        "processed": {"path": _shown(path), "sha256": sidecar["table_sha256"]},
        "script": sidecar["script"],
        "raw": sidecar["raw"],
    }
    return Processed(table.parse(text, source=str(path)), provenance)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _shown(path: Path) -> str:
    return str(
        path.relative_to(paths.ROOT) if path.is_relative_to(paths.ROOT) else path
    )
