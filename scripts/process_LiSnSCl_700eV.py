"""Turn the published LiSnSCl DFT energies (700 eV cutoff) into a processed table.

Run it again only when the raw file or the processed format changes:

    uv run scripts/process_LiSnSCl_700eV.py

Writes the table the LiSnSCl phase field names (data/processed/LiSnSCl.csv) and its
sidecar; `phasebo-data` then checks both.
"""

from pathlib import Path

import pandas as pd
import rootutils
from hydra import compose, initialize_config_dir

from phasebo import paths, processed, table
from phasebo.config import DataConfig

rootutils.setup_root(__file__, indicator=".project-root")

RAW = paths.ROOT / "data/raw/LiSnSCl_700eV.csv"
# The published file lists its 91 reference phases last, from this row on.
FIRST_REFERENCE_ROW = 195


def main() -> None:
    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        system = DataConfig.from_hydra(compose("data", ["system=LiSnSCl"])).system
    # Exact floats: the default parser may round the last bit, which would make the
    # table differ between platforms.
    raw = pd.read_csv(RAW, skipinitialspace=True, float_precision="round_trip")
    rows = table.make_rows(
        raw.iloc[:, 0].tolist(),
        fidelity="dft",
        reference=[row >= FIRST_REFERENCE_ROW for row in range(len(raw))],
        total_energy_ev=raw.iloc[:, 1].astype(float).tolist(),
    )
    cleaned, dropped = table.clean(rows, elements=system.elements)
    out = paths.ROOT / system.processed
    processed.write(out, cleaned, raw=[RAW], script=Path(__file__))
    print(f"wrote {system.processed}: {len(cleaned)} rows, {dropped} off-field dropped")


if __name__ == "__main__":
    main()
