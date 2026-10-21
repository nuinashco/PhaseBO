# Accelerated discovery of stable compositions in inorganic materials with Bayesian optimisation of the chemical phase fields

3.02.2021 Andrij Vasylenko

phasebo fits a Gaussian process to the energies above the convex hull of the compositions computed so far in a
phase field, and uses Bayesian optimisation to pick which compositions to compute next.

How the code is built and how to change it: [CONTRIBUTING.md](CONTRIBUTING.md).

## Setup

Python 3.14+ and [uv](https://docs.astral.sh/uv/). Dependencies are pinned in `uv.lock`.

```bash
make setup          # uv sync + git hooks
cp .env.example .env   # only to mirror runs to Weights & Biases
```

## Commands

| command | does | writes |
|---|---|---|
| `phasebo-suggest` | picks the next compositions to compute from the computed ones | `next.csv`, `posterior.csv`, `convex_hull.png` (4-element fields) |
| `phasebo-benchmark` | replays BO over the computed compositions, as if computing them, from random starting ones | `bo_path.csv`, `convergence.png`, `convex_hull.png` (4-element fields) |
| `phasebo-generate` | lists the charge-balanced cells at points not computed yet | `candidates_list.csv` |
| `phasebo-data` (`make data`) | checks that every processed table still matches its raw files and its script | |
| `phasebo-configs` (`make configs`) | regenerates `configs/defaults/` and each kind's `configs/bo/<group>/<kind>.yaml` from the config schemas (`--check` only reports) | `configs/` |

The first three write to `runs/<run_name>/`, along with the resolved `config.yaml`, `run.json` (git SHA, dirty flag,
command line, start time) and `data.json` (the processed table, its script and its raw files, by sha256). Runs are
seeded (`seed=42` unless overridden), so a run repeats exactly. Inputs are checked before the run directory is
created: a missing file, or a starting set larger than the field allows, stops the command with an error.

## Data

`data/raw/` holds the datasets as published; they are never edited. One script per source,
`scripts/process_<source>.py`, reads its raw file, drops the rows with elements outside the phase field and writes
`data/processed/<field>.csv`: one row per composition and fidelity, with its point (reduced composition), a
reference flag and one energy, total (eV per cell) or above the hull (meV/atom). Next to each table,
`<field>.json` records the sha256 of the table, of its script and of its raw files. Tables and sidecars are
committed; `make data`, the tests and every command fail on a table edited by hand, or on a script or raw file
changed since the table was written, naming the script to rerun.

| phase field | raw | script | energies |
|---|---|---|---|
| `LiSnSCl` | `LiSnSCl_700eV.csv`: 286 compositions, 91 references | `process_LiSnSCl_700eV.py` | total, DFT at 700 eV |
| `LiSnSCl_multifidel` | `LiSnSCl_multifidel.csv`: the same compositions, plus 46 oxygen phases its script drops | `process_LiSnSCl_multifidel.py` | above the hull: `dft`, and `nn` from a neural network |

BO models one fidelity (`fidelity=dft` by default; `fidelity=nn` models the neural network instead). Each point
keeps its lowest-energy cell; a benchmark's targets are the explored points on the hull of the modelled fidelity.

## Usage

1. Put the dataset in `data/raw/` as received.
2. Describe the phase field in `configs/system/<name>.yaml`, as `configs/system/LiSnSCl.yaml` does:

   | key | value |
   |---|---|
   | `processed` | the table its script writes, e.g. `data/processed/<name>.csv` |
   | `elements.<symbol>.charge` | oxidation state |
   | `elements.<symbol>.atoms` | `[min, max]` atoms of that element per generated cell; omit it to allow up to `candidates.max_atoms` |
   | `candidates.max_atoms` | most atoms per generated cell |
   | `candidates.file` | candidate formulas to rank, e.g. an edited `phasebo-generate` output; `null` generates them |
   | `candidates.exclude` | formulas left out of the data and the candidates; `null` for none |

   A file can extend another and change only what differs, as `LiSnSCl_multifidel.yaml` does.
3. Write `scripts/process_<source>.py` on the model of the two there: read the raw file, build rows with
   `data.table.make_rows` (flag references by row, or with `data.table.flag_references`), `data.table.clean` them
   against the phase field's elements and `data.store.write` them to the path the phase field names. Run it once with `uv run` and
   commit the table and its sidecar.
4. Run a command, overriding any value Hydra-style:

   ```bash
   uv run phasebo-suggest system=LiSnSCl bo/acquisition=ts
   uv run phasebo-benchmark bo.start.size=8 seed=1 run_name=8-starts
   uv run phasebo-benchmark bo/acquisition=random            # the random baseline, same starts
   uv run phasebo-benchmark -m bo/acquisition=qlogei,ts seed=0,1,2 'run_name=sweep/${hydra:job.num}'
   ```

   `tracker=wandb` mirrors a run to Weights & Biases; the run directory stays the source of truth.

## Configs

`configs/defaults/<command>.yaml` lists every setting of a command with its default, commented where the options or
the meaning are not obvious: seeding, logging, tracking, the batch size, the replay's length. It is generated from
the schemas in `src/phasebo/config.py`, so it is never edited. BO's parts are config groups: each kind has its own
generated file, `configs/bo/<group>/<kind>.yaml`, with every setting and its default, and presets name variants:

| group | kinds | settings |
|---|---|---|
| `bo/surrogate` | `gp`; preset `paper` (fitted noise, as in the paper) | `kernel` (`matern`, Matern 5/2 as in the paper; `rbf`), `noise_mev` (std; `null` fits it) |
| `bo/acquisition` | `qlogei` (batch log expected improvement), `ts` (Thompson sampling), `random` (the baseline; a benchmark skips the GP fit) | |
| `bo/start` (benchmark) | `random`: explored points off the hull | `size` |

To change a value:

| for | edit |
|---|---|
| one run | `key=value` on the command line; the run directory records it |
| another phase field | a file in `configs/system/`, chosen with `system=<name>` |
| another kind | `bo/<group>=<kind>`, e.g. `bo/acquisition=ts` |
| every run of this repo | `configs/<command>.yaml`, the entry config, below its `defaults:` list |
| the default itself | the schema, then `make configs` |

Secrets go in `.env` and reach configs as `${oc.env:NAME,null}`.

## Reference

Please consider citing this tool:
A. Vasylenko et al,
Inferring energy–composition relationships with Bayesian optimization enhances exploration of inorganic materials.
The Journal of Chemical Physics 160, 5, 054110 (2024)

Bayesian optimisation is implemented with BoTorch
(earlier versions, including the paper above, used GPyOpt).

```bibtex
@inproceedings{balandat2020botorch,
  author =    {Balandat, Maximilian and Karrer, Brian and Jiang, Daniel R. and Daulton, Samuel and Letham, Benjamin and Wilson, Andrew Gordon and Bakshy, Eytan},
  title =     {{BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization}},
  booktitle = {Advances in Neural Information Processing Systems 33},
  year =      {2020}
}
```
