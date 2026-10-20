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
  results and plots. `tracker=wandb` mirrors a run to Weights & Biases.
- `phasebo-suggest` writes the next batch to `next.csv` and prints it.
- `bo_path.csv`: the composition and energy of each step of a benchmark run.
- Validation with pydantic: unknown keys, invalid values and missing input files stop a run before it starts.
- `seed` (default 42): runs repeat exactly.
- `bo.acquisition`: `qlogei` (batch log expected improvement, default) or `ts` (Thompson sampling).
- `bo.kernel`: `matern` (Matérn 5/2, as in the paper, default) or `rbf`.
- `bo.noise_mev`: the energy noise the Gaussian process assumes, in meV/atom. 0 by default, as DFT energies are
  deterministic; `null` fits it, as in the paper.
- `bo.batch_size`: compositions picked per iteration (it was fixed at 4).
- Tests on synthetic phase fields; the project follows the nuinashco/conventions `rnd` template (`make check`).
- Optional `notebooks` dependency group (marimo, plotly).

### Changed
- Bayesian optimisation uses BoTorch instead of GPyOpt, which is no longer maintained.
- Packaging with uv (`make setup`): dependencies pinned in `uv.lock`, sources in `src/`, Python 3.14 or newer.
- Data paths start from the repository root (marked by `.project-root`), so the commands run from any directory.
- Plots are saved to the run directory; `show_plots=true` also opens them.
- The posterior is written to `posterior.csv` in the run directory.

### Removed
- `input_config.yaml`, the `--config` option and the single `phasebo` command with its modes: use the commands
  above, `configs/` and command-line overrides.
- Segmented starting compositions (`seeds_type=segmented`).
- `example/`, the outputs of the old default run.

### Fixed
- Reference compositions were treated as search candidates since July 2025: their names in the data file start with
  spaces and no longer matched. Path mode spent about half of its evaluations on them and counted them as stable
  compositions found; suggest mode trained on them.
- A coordinate shared by several compositions (one formula in two cell sizes) takes their lowest energy instead of
  the sum.
- The posterior standard deviation is reported in meV/atom (it was in standardised units).
- The exclude file is read (the `excludefile` key was misspelt in the configs).
- The composition table is logged without logging errors.

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
