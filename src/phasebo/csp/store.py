"""A CSP run's files: each result and reference as it finishes, and their CIFs.

results/<cell>_<framework>_<fidelity>.json   the result, structures included
results/<cell>_<framework>_<fidelity>.cif    the cell after the free relax
results/<cell>_<framework>_<fidelity>_framework.cif   relaxed at fixed cell
references/<name>.json, references/<name>.cif
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nuikit.errors import DataError

from phasebo.csp.references import Reference
from phasebo.csp.result import Result

if TYPE_CHECKING:
    from ase import Atoms

_RESULTS = "results"
_REFERENCES = "references"


def write_result(run_dir: Path, result: Result) -> None:
    """Write a result's JSON and its two CIFs."""
    stem = run_dir / _RESULTS / "_".join(result.key)
    _write_cif(stem.with_suffix(".cif"), result.released)
    _write_cif(stem.with_name(f"{stem.name}_framework.cif"), result.on_framework)
    _write_json(stem.with_suffix(".json"), result.to_json())


def write_reference(run_dir: Path, reference: Reference) -> None:
    """Write a reference's JSON and its relaxed CIF."""
    stem = run_dir / _REFERENCES / reference.name
    _write_cif(stem.with_suffix(".cif"), reference.structure)
    _write_json(stem.with_suffix(".json"), reference.to_json())


def read_results(run_dir: Path) -> list[Result]:
    """Return the run's results, by name."""
    return [Result.from_json(data) for data in _read_jsons(run_dir / _RESULTS)]


def read_references(run_dir: Path) -> list[Reference]:
    """Return the run's relaxed references, by name."""
    return [Reference.from_json(data) for data in _read_jsons(run_dir / _REFERENCES)]


def read_structures(directory: Path) -> dict[str, Atoms]:
    """Return every CIF in `directory` by its stem, in name order."""
    from ase.io import read

    if not directory.is_dir():
        raise DataError(
            f"no directory {directory}; fix csp.references.directory or set it to null"
        )
    found: dict[str, Atoms] = {}
    for path in sorted(directory.glob("*.cif")):
        try:
            blocks = read(path, index=":", format="cif")
        except (ValueError, KeyError, IndexError) as e:
            raise DataError(f"cannot read {path}: {e}; fix or remove it") from e
        if not isinstance(blocks, list) or len(blocks) != 1:
            raise DataError(f"{path} holds several structures; split it")
        found[path.stem] = blocks[0]
    if not found:
        raise DataError(
            f"no CIF files in {directory}; add the references or set "
            "csp.references.directory to null"
        )
    return found


def write_lines(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write `rows` as JSON lines, one per row, in the order given."""
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Written last and renamed into place, so a run killed mid-write leaves no
    # half result for the next run to skip over.
    part = path.with_name(f"{path.name}.part")
    part.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    part.replace(path)


def _write_cif(path: Path, atoms: Atoms) -> None:
    from ase.io import write

    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, atoms, format="cif")


def _read_jsons(directory: Path) -> list[dict[str, Any]]:
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]
