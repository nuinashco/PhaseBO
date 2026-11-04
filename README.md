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
| `phasebo-csp` | searches the structures of compositions in frameworks, with energies from an MLIP (see [Structure prediction](#structure-prediction)) | `results/`, `results.jsonl`, `table.csv` |
| `phasebo-data` (`make data`) | checks that every processed table still matches its raw files and its script | |
| `phasebo-configs` (`make configs`) | regenerates `configs/defaults/` and each kind's `configs/<group>/<kind>.yaml` from the config schemas (`--check` only reports) | `configs/` |

The first five write to `runs/<run_name>/`, along with the resolved `config.yaml`, `run.json` (git SHA, dirty flag,
command line, start time) and, but for `phasebo-csp`, `data.json` (the processed table, its script and its raw
files, by sha256). Runs are
seeded (`seed=42` unless overridden), so a run repeats exactly. Inputs are checked before the run directory is
created: a missing file, or a starting set larger than the field allows, stops the command with an error.

## Data

`data/raw/` holds the datasets as published; they are never edited. One script per source,
`scripts/process_<source>.py`, reads its raw file, drops the rows with elements outside the phase field and writes
`data/processed/<field>.csv`: one row per cell and fidelity, with its point (reduced composition), a reference
flag, one energy, total (eV per cell) or above the hull (meV/atom), and where given its framework. Next to each
table, `<field>.json` records the sha256 of the table, of its script and of its raw files. Tables and sidecars
are committed; `make data`, the tests and every command fail on a table edited by hand, or on a script or raw
file changed since the table was written, naming the script to rerun.

| phase field | raw | script | energies |
|---|---|---|---|
| `LiSnSCl` | `LiSnSCl_700eV.csv`: 286 compositions, 91 references | `process_LiSnSCl_700eV.py` | total, DFT at 700 eV |
| `LiSnSCl_multifidel` | `LiSnSCl_multifidel.csv`: the same compositions, plus 46 oxygen phases its script drops | `process_LiSnSCl_multifidel.py` | above the hull: `dft`, and `nn` from a neural network |
| `fake_LiSnSCl_<size>_<seed>` | none: the real `LiSnSCl` table sets their trend | `make_fake_fields.py` | total, fake; three frameworks per four-element composition |

BO models one fidelity (`fidelity=dft` by default; `fidelity=nn` models the neural network instead). Each point
keeps its lowest-energy cell; a benchmark's targets are the explored points on the hull of the modelled fidelity.

A table may place each cell in a structure framework. A point is then a composition in one framework: the hull
is built from every framework's cells, a composition's winning framework is its lowest point, and the targets are
the stable compositions in their winning framework. Models see the framework as a category after the
coordinates.

The fake fields (`scripts/make_fake_fields.py`, 15 of them) stand in for framework data that does not exist yet:
every charge-balanced Li-Sn-S-Cl composition with at most `<size>` atoms per cell (12, 16, 20, 24, 32: 21 to 502
four-element compositions), 3 seeds each. Compositions of fewer elements are the references; every four-element
one sits in cubic, hexagonal and wurtzite frameworks. A formation energy is the real field's trend (a GP fitted to
the real energies) plus a roughness the size of that fit's noise (25 meV/atom), and each framework adds a smooth
offset (50 meV/atom, lengthscale 0.15 in atomic fractions), so different frameworks win in different regions.

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
| `map` | after every batch, the share of compositions whose winning framework the run names right, and the F1 of the compositions it believes stable against the true ones (1 is perfect; stable means within `tolerance_mev`, 10 meV/atom, of the hull); writes `beliefs.parquet` | stable-set F1 0.46 after 32 (random 0.04) |

`bo/acquisition=random` checks the setup: its scores match random's values within their error.

### Multi-fidelity

With `benchmark/cheap`, every game also knows a cheap energy for each computed composition from the start, as if
a fast, rough model had screened them all for free. BO still picks DFT evaluations, and the budget and the
metrics count only those. The run writes `cheap.parquet`: every game's cheap energy at each explored composition,
beside its DFT energy, so their agreement can be measured.

| `benchmark/cheap=` | the cheap energy |
|---|---|
| `table` | another fidelity of the processed table (`benchmark.cheap.fidelity`, `nn` by default, which `system=LiSnSCl_multifidel` has), the same in every game; energies above the hull are used as the table gives them, without rebuilding the hull |
| `noisy` | the modelled total energies with a random error per cell (`random_mev`), a smooth one shared by nearby compositions (`smooth_mev`, `lengthscale`, `kernel`) and a fixed one per framework (`bias_mev`, e.g. `'+benchmark.cheap.bias_mev={cubic:30,wurtzite:-15}'`), then the hull rebuilt; drawn anew in every game, the same for every setting of one seed; needs total energies (`system=LiSnSCl`, the fake fields) |

Four surrogates use the cheap energies and predict DFT's: `multitask` (BoTorch's multi-task GP, which learns
how far DFT follows the cheap energies), `multifidelity` (its multi-fidelity GP, the source as an input), `delta`
(the cheap energy plus a GP of its error, fitted where DFT is known) and `joint` (one GP of both sources, the
source as a category). Only `delta` and `joint`, like `gp`, model fields with frameworks. All assume zero noise by
default, as the cheap model's output is as deterministic as DFT's. `bo/acquisition=screen` is their baseline: DFT
in cheap order, never learning from DFT. Each refuses to run without a cheap source, and `phasebo-suggest` has none
yet.

```bash
uv run phasebo-benchmark benchmark/cheap=noisy bo/surrogate=multitask
uv run phasebo-benchmark benchmark/cheap=noisy bo/acquisition=screen
uv run phasebo-benchmark benchmark/cheap=noisy benchmark.cheap.random_mev=0 benchmark.cheap.smooth_mev=100 bo/surrogate=multifidelity
```

Evaluations until the first stable composition / until each, averaged (LiSnSCl, seed 42, 300 games; qLogEI unless
the setting is `bo/acquisition=screen`):

| setting | `random_mev=25` | `random_mev=0 smooth_mev=100` |
|---|---|---|
| no cheap source (`bo/surrogate=gp`) | 23.9 / 28.6 | 23.9 / 28.6 |
| `bo/acquisition=screen` | 4.9 / 11.7 | 14.6 / 22.7 |
| `bo/surrogate=multitask` | 5.0 / 11.5 | 10.0 / 18.6 |
| `bo/surrogate=multifidelity` | 6.9 / 14.9 | 5.6 / 11.2 |
| `bo/surrogate=delta` | 6.0 / 13.9 | 8.4 / 14.7 |

The screen is hard to beat when the cheap errors are random: nothing in them can be learned. Against it, game by game
(`phasebo-compare` with the screen run first; evaluations to the first stable composition, 95% intervals),
`multitask` ties it (+0.07, -0.04 to +0.17), and `delta` (+1.09, +0.77 to +1.43) and `multifidelity` (+1.96, +1.45
to +2.55) need more. Errors that nearby compositions share can be learned, so there every surrogate beats it:
`multitask` by 4.60 (3.80 to 5.45), `delta` by 6.27 (5.11 to 7.41) and `multifidelity` by 9.03 (7.81 to 10.27).
All of this comes from a single field, LiSnSCl with 3 targets.

### Frameworks

On a fake field with frameworks, the question becomes the whole map: which framework wins at each composition, and
which compositions are stable. After 32 evaluations (`system=fake_LiSnSCl_20_0`: 115 compositions in 3
frameworks, 7 stable; qLogEI, 100 games):

| setting | stable found (of 7) | evaluations per stable | right framework | stable-set F1 |
|---|---|---|---|---|
| random picking (`bo/acquisition=random`) | 0.7 | 30.7 | | |
| no cheap source (`bo/surrogate=gp`) | 3.5 | 24.8 | 58% | 0.65 |
| `bo/acquisition=screen`, random errors (`random_mev=25`) | 6.7 | 10.0 | 78% | 0.96 |
| `bo/surrogate=delta`, random errors | 6.4 | 12.5 | 77% | 0.91 |
| `bo/surrogate=joint`, random errors | 6.6 | 11.0 | 78% | 0.95 |
| `bo/acquisition=screen`, patterned errors (`random_mev=0 smooth_mev=50`, bias cubic +30, wurtzite -15) | 6.95 | 8.5 | 78% | 0.99 |
| `bo/surrogate=delta`, patterned errors | 6.99 | 8.8 | **99%** | 1.00 |
| `bo/surrogate=joint`, patterned errors | 6.97 | 9.5 | 92% | 1.00 |

With a cheap source, every rule finds nearly every stable composition. The difference is the map: when the cheap
model ranks the frameworks wrongly by a fixed amount, `delta` and `joint` learn that from a few DFT points and name
the right winner for 92-99% of compositions; the screen keeps the cheap model's mistakes (78%). With random errors
there is nothing to learn. `joint` takes about 15 times as long as `delta` here (9 minutes against 40 s).

```bash
uv run phasebo-benchmark system=fake_LiSnSCl_20_0 benchmark/cheap=noisy benchmark.cheap.random_mev=0 \
  benchmark.cheap.smooth_mev=50 '+benchmark.cheap.bias_mev={cubic:30,wurtzite:-15}' bo/surrogate=delta
```

### Comparing settings

`phasebo-compare` compares benchmark runs with the first one listed, the baseline, usually a
`bo/acquisition=random` run. Runs of one seed, start and phase field play the same 300 games, so it compares
game by game; with a cheap source, the games' cheap draws match only between runs with the same cheap settings:

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
`data.json`, `cheap.parquet` with a cheap source and `beliefs.parquet` with the map) are saved as its
`replays-<run id>` artifact, so any later comparison can start from W&B alone. The
project's "Benchmarks" view opens with a "How to read" note (the setup and what each headline bar means), then
shows the headline values side by side, each metric's curves under a short explanation, one line per run, the
`phasebo-compare` runs' tables (logged with `tracker=wandb`), and the system charts collapsed. Random picking's
exact values stay in `metrics.json`; in W&B, a `bo/acquisition=random` run is the comparison.

The same content is also a W&B report, "PhaseBO benchmarks", for people outside the project: its Share button
makes a view-only link that opens without a W&B account, if the organisation allows such links.

## Structure prediction

A formula such as Sc1Al3N4 says how many atoms of each element a cell holds, not where they sit, and the energy
depends on where they sit. Crystal structure prediction (CSP) searches for the arrangement with the lowest energy.
`phasebo-csp` does this without DFT: every energy comes from a machine-learned interatomic potential (MLIP), a
network trained on many DFT calculations that returns a cell's energy and forces in milliseconds. Its energies
then feed the benchmark like any other phase field's. The search is ported from PolyBO, the group's adaptation of
ChemDASH (Sharp et al., *PCCP* 2020, 22, 18205).

**Frameworks.** The search stays inside a framework: a fixed pattern of anion sites (the N of a nitride), with the
holes between them offered to the cations (Al, Sc, B). Each framework has three holes per anion site, so most
holes stay empty; the search decides which cation sits in which hole, and which anion sites stay empty when there
are fewer anions than sites.

| framework | the anions are packed as | holes per anion site | structures it contains |
|---|---|---|---|
| `cubic` | cubic close packing (fcc) | 1 octahedral, 2 tetrahedral | rocksalt, zincblende |
| `wurtzite` | hexagonal close packing | 2 tetrahedral, 1 octahedral | wurtzite (AlN at room conditions), NiAs |
| `hexagonal` | flat hexagonal layers stacked straight above each other | 2 in the layer, 1 between layers | h-BN |

For Sc1Al3N4, the 4 N fill 4 anion sites and the 4 cations take 4 of the 12 holes. The number of anion sites is
rounded up to a multiple of 4, so every framework offers a composition the same sites and their energies compare
fairly. The cell's volume is the sum of its ions' volumes (Shannon radii) over 0.6 (`csp.phi`), so a big cation
opens the framework up. Give whole cells: `AlN` is 1 Al and 1 N on 4 anion sites, 3 of them empty; a full cell is
`Al4N4`.

**Two levels.** Each is a fidelity of the table, named `<mlip>_<level>`:

- `relax` (`chgnet_relax`, the cheap one): the atoms go on random sites, are pushed off them a little (0.1 Å) and
  relaxed by the MLIP with the cell held fixed, so they slide into the nearest energy minimum.
- `basin_hop` (`chgnet_bh50`, the expensive one): the same start, then 50 moves (`epochs`). A move swaps 2 or 3
  occupants of one sublattice, such as an Sc with an Al or an N into an empty site, and relaxes the new
  arrangement. A lower energy is kept; a higher one is kept now and then (the Metropolis rule at `kt_ev`), so the
  search can climb out of a dip.

Both end by relaxing the best arrangement again with the cell free to change shape and size. Each result records
the energy on the framework and after that free relax (eV per cell and per atom), the drift (how far the atoms
moved from the framework's sites, Å; a large drift means the composition does not want that framework), the
volume change, the space group, the time it took, and the MLIP's name and version.

```bash
uv sync --group csp                  # CHGNet and MACE (csp/mlip=chgnet, mace); emt needs neither
uv run --group csp phasebo-csp 'formulas=[Sc1Al3N4,Sc2Al2N4]'              # all frameworks, both levels
uv run --group csp phasebo-csp formulas_file=cells.csv 'csp/search=[relax]'  # the cheap level only
uv run phasebo-csp 'formulas=[Al4N4]' csp/mlip=emt                        # ASE's toy potential, a smoke run
```

Plain `uv run` uninstalls the `csp` group again, so commands that need it take `--group csp`. On a CPU,
`csp.workers=4` splits the jobs over 4 processes. A run is resumable: rerun it with the same `run_name` and
settings, and the jobs already done are skipped; `resumes.jsonl` records each rerun's commit and command line. Each job draws from its own seed, made from the run's seed and
the job's name, so a result is the same whatever else the run computes, in whatever order.

| in `runs/<run_name>/` | holds |
|---|---|
| `results/` | each job as it finishes: `<cell>_<framework>_<fidelity>.json`, the cell after the free relax (`.cif`) and on the framework (`_framework.cif`) |
| `references/` | each reference structure relaxed, the same way |
| `results.jsonl`, `references.jsonl` | all of them, one per line |
| `table.csv`, `table.json` | the processed table and its sidecar, when references are given |

**From structures to the benchmark.** The hull needs every element's energy from the same MLIP. Put CIF files of
the elements (Sc and Al metal, boron, an N2 molecule in a large box) and any known compounds in a directory, and
set `csp.references.directory` to it; each is relaxed with the cell free and flagged as a reference. The table
then has, per fidelity, every result's energy after the free relax in its framework, and every reference.
`publish=<name>` also writes it to `data/processed/<name>.csv`, with the run's `config.yaml`, `results.jsonl` and
`references.jsonl` copied to `data/processed/<name>/`. Its sidecar pins those three files, not a script, and
names the command that rebuilt the table, so `make data` and the commands refuse it if any of them changed.
Describe the field in `configs/system/<name>.yaml` (see Usage), commit, and replay BO over it:

```bash
uv run --group csp phasebo-csp formulas_file=cells.csv csp.references.directory=data/raw/ScBAlN_references \
  publish=ScBAlN_chgnet run_name=ScBAlN_chgnet
uv run phasebo-benchmark system=ScBAlN_chgnet fidelity=chgnet_bh50 \
  benchmark/cheap=table benchmark.cheap.fidelity=chgnet_relax bo/acquisition=screen
```

**On a GPU.** CHGNet builds from source on Python 3.14, so the machine needs a C compiler (`sudo apt install
build-essential` on Ubuntu). The lock file pins the CPU build of torch on Linux, so a GPU machine swaps it after
the sync and then runs without syncing, which would put the CPU build back. The repo needs torch 2.14, which the
cu129 index does not have (it stops at 2.13); a newer CUDA index such as cu130 should, though that is untested:

```bash
uv sync --group csp
uv pip install --reinstall "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv run --no-sync phasebo-csp ... csp.device=cuda
```

A GPU takes one process (`csp.workers=1`); `csp.device=auto` picks it when torch sees one. On an NVIDIA L4, one
energy and force call takes about 50 ms with CHGNet and 40 ms with MACE in float32 (its default here) for 16 to 32
atoms. Lower precision does not help: float16 is no faster at these sizes and moves energies by 5-13 meV/atom,
and bfloat16 crashes in both. More throughput comes from more processes, on more machines or CPUs.

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
| `bo/surrogate` | `gp`; preset `paper` (fitted noise, as in the paper); with a cheap source, `multitask`, `multifidelity`, `delta`, `joint` | `kernel` (`matern12`, `matern32`, `matern52` as in the paper; `rbf`), `noise_mev` (std; `null` fits it) |
| `bo/acquisition` | `qlogei` (batch log expected improvement), `ts` (Thompson sampling), `random` (the baseline; a benchmark skips the GP fit), `screen` (DFT in cheap order) | |
| `benchmark/start` | `random`: explored points off the hull | `size` (8) |
| `benchmark/cheap` | none by default; `table`, `noisy` | `table.fidelity` (`nn`); `noisy.random_mev` (25), `smooth_mev` (0), `lengthscale` (0.2), `bias_mev` (none) |
| `csp/mlip` | `chgnet` (default), `mace`, `emt` (a toy, for tests) | `chgnet.model` (`0.3.0`), `mace.model` (`medium`) |
| `csp/search` (a list) | `relax`, `basin_hop`: both by default, each a fidelity | `basin_hop.epochs` (50), `kt_ev` (0.05), `max_swaps` (3) |
| `benchmark/metric` (a list) | `steps_to_hit`, `discovery`, `best_so_far`, `hazard_ratio`, `map`: all five by default | `hazard_ratio.level` (0.95), `map.tolerance_mev` (10) |

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
