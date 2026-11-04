"""Turn the FastCHGNet random-search CSP of LiSnSCl into a processed table.

Run it again only when a raw file or the processed format changes:

    uv run scripts/process_LiSnSCl_fastchgnet_csp.py

data/raw/LiSnSCl_fastchgnet_csp/ is what scripts/mlip_csp exported: 40 trials per
composition, each a random symmetric structure (pyxtal) relaxed by FastCHGNet with the
CHGNet 0.3.0 weights, on the composition's smallest DFT cell. Fidelity `dft` is the
700 eV file, as in LiSnSCl. Fidelity `fastchgnet_random<n>` gives each composition the
lowest usable energy of its first n trials. Its references are Materials Project
structures relaxed by the same MLIP (the lowest physical one per formula), and the
DFT-hull references Materials Project lacks, at their lowest over all trials. Writes
data/processed/LiSnSCl_fastchgnet_csp.csv and its sidecar.
"""

import json
import math
from pathlib import Path

import pandas as pd
import rootutils
from hydra import compose, initialize_config_dir
from pymatgen.core import Composition

from nuikit.errors import DataError, cli_boundary

from phasebo import paths
from phasebo.config import DataConfig
from phasebo.data import store, table

rootutils.setup_root(__file__, indicator=".project-root")

RAW = paths.ROOT / "data/raw/LiSnSCl_fastchgnet_csp"
TRIALS = RAW / "trials.jsonl"
COMPOSITIONS = RAW / "compositions.csv"
MISSING = RAW / "missing_references.csv"
REFERENCES = RAW / "references.csv"
DFT = paths.ROOT / "data/raw/LiSnSCl_700eV.csv"
# The 700 eV file lists its 91 reference phases last, from this row on.
FIRST_REFERENCE_ROW = 195
# Trials per composition of each MLIP fidelity; below 10, a composition has none usable.
BUDGETS = (10, 15, 20, 25, 30, 40)


@cli_boundary
def main() -> None:
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        composed = compose("data", ["system=LiSnSCl_fastchgnet_csp"])
        system = DataConfig.from_hydra(composed).system
    energies = read_trials()
    cells = pd.read_csv(COMPOSITIONS)["csp_cell"].tolist()
    references = mlip_references(energies)
    rows = [dft_rows()]
    for budget in BUDGETS:
        best = {cell: min(energies[cell][:budget]) for cell in cells}
        if empty := [cell for cell, e in best.items() if math.isinf(e)]:
            raise DataError(
                f"{empty[0]} has no usable trial among its first {budget} in "
                f"{TRIALS}; drop {budget} from BUDGETS"
            )
        fidelity = f"fastchgnet_random{budget}"
        rows.append(
            table.make_rows(
                [*best, *references],
                fidelity=fidelity,
                reference=[False] * len(best) + [True] * len(references),
                total_energy_ev=[
                    e * Composition(f).num_atoms
                    for f, e in [*best.items(), *references.items()]
                ],
            )
        )
    source = str(RAW.relative_to(paths.ROOT))
    cleaned, dropped = table.clean(
        pd.concat(rows, ignore_index=True), elements=system.elements, source=source
    )
    out = paths.ROOT / system.processed
    raw = [DFT, TRIALS, COMPOSITIONS, MISSING, REFERENCES]
    store.write(out, cleaned, raw=raw, script=Path(__file__))
    print(f"wrote {system.processed}: {len(cleaned)} rows, {dropped} off-field dropped")


def read_trials() -> dict[str, list[float]]:
    """Return each cell's energies (eV/atom) in trial order; inf where unusable."""
    energies: dict[str, list[float]] = {}
    for line in TRIALS.read_text(encoding="utf-8").splitlines():
        trial = json.loads(line)
        found = energies.setdefault(trial["cell"], [])
        if trial["trial"] != len(found):
            raise DataError(f"{TRIALS} lists {trial['cell']}'s trials out of order")
        found.append(trial["energy"] if trial["usable"] else math.inf)
    return energies


def mlip_references(energies: dict[str, list[float]]) -> dict[str, float]:
    """Return each reference formula's MLIP energy (eV/atom), the lowest one found."""
    mp = pd.read_csv(REFERENCES)
    references: dict[str, float] = {}
    for formula, energy, physical in zip(
        mp["formula"], mp["energy"], mp["physical"], strict=True
    ):
        if physical:
            references[formula] = min(float(energy), references.get(formula, math.inf))
    for cell in pd.read_csv(MISSING)["csp_cell"]:
        references[cell] = min(energies[cell])
    return references


def dft_rows() -> pd.DataFrame:
    """Return the 700 eV file's rows, as scripts/process_LiSnSCl_700eV.py reads it."""
    # Exact floats: the default parser may round the last bit, which would make the
    # table differ between platforms.
    raw = pd.read_csv(DFT, skipinitialspace=True, float_precision="round_trip")
    return table.make_rows(
        raw.iloc[:, 0].tolist(),
        fidelity="dft",
        reference=[row >= FIRST_REFERENCE_ROW for row in range(len(raw))],
        total_energy_ev=raw.iloc[:, 1].astype(float).tolist(),
    )


if __name__ == "__main__":
    main()
