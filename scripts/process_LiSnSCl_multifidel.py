"""Turn the published LiSnSCl DFT and NNmodel hull energies into a processed table.

Run it again only when a raw file or the processed format changes:

    uv run scripts/process_LiSnSCl_multifidel.py

The file gives each composition's energy above the hull (meV/atom) from DFT and from a
neural network. It has no reference rows of its own and interleaves 46 oxygen phases
with the references, so the references come from the 700 eV file, and the oxygen
rows are dropped as off-field. Writes data/processed/LiSnSCl_multifidel.csv and its
sidecar.
"""

from pathlib import Path

import pandas as pd
import rootutils
from hydra import compose, initialize_config_dir

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.config import DataConfig
from phasebo.data import store, table

rootutils.setup_root(__file__, indicator=".project-root")

RAW = paths.ROOT / "data/raw/LiSnSCl_multifidel.csv"
REFERENCES = paths.ROOT / "data/raw/LiSnSCl_700eV.csv"
# The 700 eV file lists its 91 reference phases last, from this row on.
FIRST_REFERENCE_ROW = 195
FIDELITIES = {"DFT": "dft", "NNmodel": "nn"}


@cli_boundary
def main() -> None:
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        composed = compose("data", ["system=LiSnSCl_multifidel"])
        system = DataConfig.from_hydra(composed).system
    # Exact floats: the default parser may round the last bit, which would make the
    # table differ between platforms.
    raw = pd.read_csv(RAW, skipinitialspace=True, float_precision="round_trip")
    known = pd.read_csv(REFERENCES, skipinitialspace=True).iloc[FIRST_REFERENCE_ROW:, 0]
    formulas = raw.iloc[:, 0].str.strip().tolist()
    reference = table.flag_references(formulas, known.str.strip().tolist())
    rows = pd.concat(
        [
            table.make_rows(
                formulas,
                fidelity=fidelity,
                reference=reference,
                e_above_hull_mev=raw[column].astype(float).tolist(),
            )
            for column, fidelity in FIDELITIES.items()
        ],
        ignore_index=True,
    )
    source = str(RAW.relative_to(paths.ROOT))
    cleaned, dropped = table.clean(rows, elements=system.elements, source=source)
    out = paths.ROOT / system.processed
    store.write(out, cleaned, raw=[RAW, REFERENCES], script=Path(__file__))
    print(f"wrote {system.processed}: {len(cleaned)} rows, {dropped} off-field dropped")


if __name__ == "__main__":
    main()
