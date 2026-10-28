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
| `phasebo-suggest` | picks the next compositions to compute from the computed ones | `next.csv`, `posterior.csv` |
| `phasebo-benchmark` | replays BO over the computed compositions, as if computing them, from 300 random starting sets, and scores it against random picking | `paths.parquet`, `metrics.json`; `cheap.parquet` with a cheap source |
| `phasebo-generate` | lists the smallest charge-balanced cell of each point not computed yet | `candidates_list.csv` |
| `phasebo-compare` | compares benchmark runs with the first, game by game | `comparison.json` |
| `phasebo-data` (`make data`) | checks that every processed table still matches its raw files and its script | |
| `phasebo-configs` (`make configs`) | regenerates `configs/defaults/` and each kind's `configs/<group>/<kind>.yaml` from the config schemas (`--check` only reports) | `configs/` |

The first four write to `runs/<run_name>/`, along with the resolved `config.yaml`, `run.json` (git SHA, dirty flag,
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

## Benchmark

`phasebo-benchmark` asks how much sooner BO finds stable compositions than picking at random. It hides the energies
of the computed compositions, starts from 8 random ones (never a stable one), and lets BO pick 8 batches of 4: the
paper's real use on a new phase field. It replays this from 300 starting sets in parallel (about 20 s on 12 cores
for qLogEI). The same `seed` gives every setting the same starting sets, so settings compare run by run.

Each metric is printed, written to `metrics.json` next to random picking's exact value (a formula, not a
simulation), and, with `tracker=wandb`, logged as curves over evaluations:

| metric | reads | example (qLogEI, LiSnSCl) |
|---|---|---|
| `steps_to_hit` | evaluations until the first stable composition, and each one, averaged; a miss counts as the budget (32) | first hit after 23.9 (random 24.4) |
| `discovery` | the share of runs that found each target after every batch | 0.89 of 3 found within 32 (random 0.61) |
| `best_so_far` | the lowest energy above the hull known after every batch | 2.8 meV/atom after 32 (random 4.9) |
| `hazard_ratio` | BO's chance per batch of picking a target, as a multiple of random's, with a 95% interval; and batch by batch | 1.50 (1.33 to 1.68); 0.5 in the first batch, 2.2 in the eighth |

`bo/acquisition=random` checks the setup: its scores match random's values within their error.

### Multi-fidelity

With `benchmark/cheap`, every game also knows a cheap energy for each computed composition from the start, as if
a fast, rough model had screened them all for free. BO still picks DFT evaluations, and the budget and the
metrics count only those. The run writes `cheap.parquet`: every game's cheap energy at each explored composition,
beside its DFT energy, so their agreement can be measured.

| `benchmark/cheap=` | the cheap energy |
|---|---|
| `table` | another fidelity of the processed table (`benchmark.cheap.fidelity`, `nn` by default, which `system=LiSnSCl_multifidel` has), the same in every game |
| `noisy` | the modelled total energies with a random error per cell (`random_mev`) and a smooth one shared by nearby compositions (`smooth_mev`, `lengthscale`), then the hull rebuilt; drawn anew in every game, the same for every setting of one seed; needs total energies (`system=LiSnSCl`) |

Three surrogates fit the cheap energies of every explored composition along with the DFT ones, and predict DFT's:
`multitask` (BoTorch's multi-task GP, which learns how far DFT follows the cheap energies), `multifidelity`
(its multi-fidelity GP, the source as an input) and `delta` (the cheap energy plus a GP of its error). All
assume zero noise by default, as the cheap model's output is as deterministic as DFT's. `bo/acquisition=screen`
is their baseline: DFT in cheap order, never learning from DFT. Each refuses to run without a cheap source, and
`phasebo-suggest` has none yet.

```bash
uv run phasebo-benchmark benchmark/cheap=noisy benchmark.cheap.random_mev=50 bo/surrogate=multitask
uv run phasebo-benchmark benchmark/cheap=noisy benchmark.cheap.random_mev=50 bo/acquisition=screen
```

Evaluations until the first stable composition / until each, averaged (LiSnSCl, qLogEI, 300 games):

| setting | `random_mev=25` | `random_mev=0 smooth_mev=100` |
|---|---|---|
| no cheap source (`bo/surrogate=gp`) | 23.9 / 28.6 | 23.9 / 28.6 |
| `bo/acquisition=screen` | 4.9 / 11.7 | 11.2 / 19.7 |
| `bo/surrogate=multitask` | 5.0 / 11.5 | 8.6 / 16.6 |
| `bo/surrogate=multifidelity` | 6.9 / 14.9 | 4.8 / 10.0 |
| `bo/surrogate=delta` | 6.0 / 13.9 | 6.7 / 12.0 |

The screen is hard to beat when the cheap errors are random: nothing in them can be learned. Errors that nearby
compositions share can be, so there the surrogates beat it.

### Comparing settings

`phasebo-compare` compares benchmark runs with the first one listed, the baseline, usually a
`bo/acquisition=random` run. Runs of one seed, start and phase field play the same 300 games, so it compares
game by game:

```bash
uv run phasebo-compare 'runs=[random-<time>,qlogei-<time>,ts-<time>]'    # under runs/
uv run phasebo-compare 'wandb=[<random id>,<qlogei id>]' tracker=wandb    # from W&B
```

For each run and outcome it prints the difference from the baseline (or the ratio, for the hazard ratio) with a
95% interval and the share of games the run won and lost, and writes them to `comparison.json`:

```text
baseline: random-2026-10-21_13-42-13
qlogei-2026-10-21_13-41-18
  discovery/found          +0.24  (95%: +0.12 to +0.36)  better in 39%, worse in 23%
  hazard_ratio/theta       x1.41  (95%: 1.18 to 1.69)
```

It refuses runs that did not play the same games: another seed, start, phase field or budget, or a processed
table changed since. With `wandb=`, it downloads the runs' `replays-<run id>` artifacts from `<wandb_entity>/phasebo`
(`wandb_entity=null` for runs logged to your default entity).

### In Weights & Biases

With `tracker=wandb`, a benchmark run is named after its batch rule (`qlogei-<time>` unless `run_name` is set),
grouped by phase field, typed `benchmark` and tagged with every family setting (`surrogate.kernel=matern52`), so the
W&B table filters and groups by setting. Each metric's curves are plotted against `evaluations`, the summary
holds the metrics' values, and the run's outputs (`paths.parquet`, `metrics.json`, `config.yaml`,
`data.json`, and `cheap.parquet` with a cheap source) are saved as its `replays-<run id>` artifact, so any later comparison can start from W&B alone. The
project's "Benchmarks" view opens with a "How to read" note (the setup and what each headline bar means), then
shows the headline values side by side, each metric's curves under a short explanation, one line per run, the
`phasebo-compare` runs' tables (logged with `tracker=wandb`), and the system charts collapsed. Random picking's
exact values stay in `metrics.json`; in W&B, a `bo/acquisition=random` run is the comparison.

The same content is also a W&B report, "PhaseBO benchmarks", for people outside the project: its Share button
makes a view-only link that opens without a W&B account, if the organisation allows such links.

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
   uv run phasebo-benchmark bo/acquisition=ts benchmark.start.size=23 run_name=ts-23-starts
   uv run phasebo-benchmark bo/acquisition=random            # the random check, same starting sets
   uv run phasebo-benchmark benchmark/metric=[hazard_ratio] benchmark.repeats=100
   uv run phasebo-benchmark -m bo/acquisition=qlogei,ts 'run_name=sweep/${hydra:job.num}'
   ```

   `tracker=wandb` mirrors a run to Weights & Biases, in the project's team, `rai-polybo/phasebo`
   (`wandb_entity=null` logs to your W&B login's default entity instead), with its command as the job type and
   its phase field as the group; the run directory stays the source of truth.

## Configs

`configs/defaults/<command>.yaml` lists every setting of a command with its default, commented where the options or
the meaning are not obvious: seeding, logging, tracking, the batch size, a benchmark's repeats, length and workers.
It is generated from the schemas in `src/phasebo/config.py`, so it is never edited. BO's and the benchmark's parts
are config groups: each kind has its own generated file, `configs/<group>/<kind>.yaml`, with every setting and its
default, and presets name variants:

| group | kinds | settings |
|---|---|---|
| `bo/surrogate` | `gp`; preset `paper` (fitted noise, as in the paper) | `kernel` (`matern12`, `matern32`, `matern52` as in the paper; `rbf`), `noise_mev` (std; `null` fits it) |
| `bo/acquisition` | `qlogei` (batch log expected improvement), `ts` (Thompson sampling), `random` (the baseline; a benchmark skips the GP fit) | |
| `benchmark/start` | `random`: explored points off the hull | `size` (8) |
| `benchmark/metric` (a list) | `steps_to_hit`, `discovery`, `best_so_far`, `hazard_ratio`: all four by default | `hazard_ratio.level` (0.95) |

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
