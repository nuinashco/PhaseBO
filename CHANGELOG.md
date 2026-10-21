# Changelog

All notable changes to this project will be documented in this file.

---

## [Unreleased]
### Added
- Commands `phasebo-suggest` (next compositions to compute), `phasebo-benchmark` (BO replayed over the computed
  compositions, the former path mode) and `phasebo-generate` (candidate compositions), configured with Hydra:
  choose a phase field with `system=` and override any value on the command line. `configs/defaults/<command>.yaml`
  lists every setting with its default.
- Each run writes `runs/<run_name>/`: the resolved configuration, `run.json` (git commit, command line, start time),
  and results. `tracker=wandb` mirrors a run to Weights & Biases.
- A phase field is a file in `configs/system/`: its processed table, its elements (charge and an inclusive range
  of atoms per cell) and its candidate space.
- One script per raw source (`scripts/process_<source>.py`) turns it into a committed, validated table in
  `data/processed/`, one row per composition and fidelity, rows with elements outside the phase field dropped, with
  a sidecar of the sha256 of the table, its script and its raw files. `phasebo-data` (`make data`) checks that no
  table was edited and no script or raw file changed since; commands refuse such a table.
- The `LiSnSCl_multifidel` phase field: energies above the hull from DFT and from a neural network for the same
  compositions (its 46 oxygen phases are dropped). BO models one fidelity at a time (`fidelity=dft` by default).
- Each run writes `data.json`: the processed table, its script and its raw files, by sha256.
- `phasebo-suggest` writes the next batch to `next.csv` and prints it.
- `phasebo-benchmark` replays BO from `benchmark.repeats` starting sets (300) in parallel (`benchmark.workers`,
  all CPUs by default) and scores it against random picking's exact odds. The same `seed` gives every setting the
  same starting sets, and results do not depend on the number of workers. It writes `paths.parquet` (every
  evaluation of every repeat: batch, formula, energy above the hull, whether a target) and `metrics.json`, prints
  one line per metric, and with `tracker=wandb` logs each metric's curves over evaluations.
- `benchmark/metric`, a list of metrics, all four by default: `steps_to_hit` (evaluations to each target, the
  first and their mean, a miss counting as the budget), `discovery` (the share of runs that found each target,
  batch by batch), `best_so_far` (the lowest energy known) and `hazard_ratio` (BO's chance per batch of picking a
  target as a multiple of random's, with a `level` interval; adds statsmodels).
- Validation with pydantic: unknown keys, invalid values and missing input files stop a run before it starts.
- `seed` (default 42): runs repeat exactly.
- BO is two config groups, each a choice of kind (`bo/<group>=<kind>`):
  - `bo/surrogate`: `gp`, with `kernel` (`matern12`, `matern32`, `matern52`: Matérn ν = 1/2, 3/2, 5/2, the last
    as in the paper and the default; `rbf`) and `noise_mev`, the
    energy noise it assumes in meV/atom (0 by default, as DFT energies are deterministic; `null` fits it); the
    preset `paper` is the paper's GP, with a fitted noise;
  - `bo/acquisition`: `qlogei` (batch log expected improvement, default), `ts` (Thompson sampling) or `random`, the
    baseline, which picks uniformly and skips the GP fit in a benchmark.
- `bo.batch_size` (compositions per batch, it was fixed at 4).
- A benchmark's own settings: `benchmark/start` (`random`, `size` explored points off the hull, 8 by default as
  in the paper's real use; it was 23 with no recorded reason) and `benchmark.max_iter` (batches after the start, 8
  by default: 32 picks).
- Tests on synthetic phase fields; the project follows the nuinashco/conventions `rnd` template (`make check`).
- Optional `notebooks` dependency group (marimo, plotly; matplotlib and scipy for `scripts/plot_run.py`).

### Changed
- Bayesian optimisation uses BoTorch instead of GPyOpt, which is no longer maintained.
- The published datasets moved, unchanged, to `data/raw/`.
- Packaging with uv (`make setup`): dependencies pinned in `uv.lock`, sources in `src/`, Python 3.14 or newer.
- Data paths start from the repository root (marked by `.project-root`), so the commands run from any directory.
- Commands no longer draw figures: `scripts/plot_run.py runs/<run_name>` draws the hull map and each metric's
  curves into the run directory.
- `phasebo-generate` lists one cell per point not computed yet, its smallest; it listed every cell.
- The posterior is written to `posterior.csv` in the run directory.

### Removed
- `input_config.yaml`, the `--config` option and the single `phasebo` command with its modes: use the commands
  above, `configs/` and command-line overrides.
- Segmented starting compositions (`seeds_type=segmented`).
- `example/`, the outputs of the old default run.
- `BO_Path_in_<field>.txt`, the path of one replay: `paths.parquet` holds every repeat's.

### Fixed
- A benchmark's starting set is drawn by point and never holds a target. It was drawn by name and kept only points
  at exactly 0 meV/atom out, so a target stored at -4e-13 could start a run, and a stable point shared with a higher
  cell could be seeded with that cell's energy.
- `phasebo-suggest` fits one entry per point, its lowest cell, instead of every cell, which no zero-noise GP can fit
  when cells of one point differ in energy.
- The convex hull map is skipped for fields that are not 4-element instead of crashing before results are written.
- `phasebo-generate` writes a header, so reading its output back no longer drops the first candidate.
- Generated candidates skip the points already computed or excluded; `posterior.csv` lists only them, one cell
  each (17 of the 78 rows it listed for LiSnSCl).
- Reference compositions were treated as search candidates since July 2025: their names in the data file start with
  spaces and no longer matched. Path mode spent about half of its evaluations on them and counted them as stable
  compositions found; suggest mode trained on them.
- A coordinate shared by several compositions (one formula in two cell sizes) takes their lowest energy instead of
  the sum.
- The posterior standard deviation is reported in meV/atom (it was in standardised units).
- The exclude file is read (the `excludefile` key was misspelt in the configs).

---

## [0.1.4] - 2025-07-09
### Changed
- Migrated from `setup.py` to `pyproject.toml` for modern Python packaging.
- Updated to follow standard Semantic Versioning (three-part version).
- Improved documentation and clarified usage examples for crystal structure exploration workflows.

### Fixed
- Minor packaging and dependency fixes for smoother local editable installs.

---

## [0.1.3.1] - 2022-07-23
### Added
- Preliminary support for compositional space exploration with explicit DFT and interatomic potential integration.
- Early support for parallel or batch candidate evaluations.

### Changed
- Refined Thompson Sampling and Expected Improvement (EI) acquisition function implementations for more robust convergence.

---

## [0.1.3] - 2022-03-10
### Added
- Initial integration with external crystal structure prediction modules
- New plotting utilities to visualise acquisition surfaces and posterior distributions.

---

## [0.1.2] - 2021-10-07
### Added
- Support for multiple acquisition strategies: Expected Improvement (EI) and Thompson Sampling.

### Changed
- Improved Gaussian Process model initialisation for better uncertainty handling in high-dimensional compositional spaces.

---

## [0.1.1] - 2021-08-21
### Added
- Basic Bayesian optimisation loop for accelerated compositional exploration.
- Early demonstration examples on phase fields.

---

## [0.1.0] - 2021-03-05
### Added
- Initial prototype release.
- Core Bayesian optimisation engine with Gaussian Process support.
- Simple compositional candidate generator and initial evaluation interface.

---

## [Unreleased]
### Planned
- Support for multi-objective acquisition strategies.
- Integration with active learning for iterative interatomic potential refinement.
- Additional crystal structure filtering and ranking tools.
