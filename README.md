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

| command | does | writes to `runs/<run_name>/` |
|---|---|---|
| `phasebo-suggest` | picks the next compositions to compute from the computed ones | `next.csv`, `posterior.csv`, `convex_hull.png` |
| `phasebo-benchmark` | replays BO over the computed compositions, as if computing them, from random starting ones | `bo_path.csv`, `BO_Path_in_<field>.txt`, `convergence.png`, `convex_hull.png` |
| `phasebo-generate` | lists the charge-balanced candidate compositions | `candidates_list.csv` |
| `phasebo-configs` | regenerates `configs/defaults/` from the config schemas (`--check` only reports) | |

Every run directory also holds the resolved `config.yaml` and `run.json` (git SHA, dirty flag, command line, start
time). Runs are seeded (`seed=42` unless overridden), so a run repeats exactly. Inputs are checked before the run
directory is created: a missing file stops the command with an error.

## Usage

1. Prepare a CSV with one row per computed composition: the formula and its total energy (eV per cell). Put the
   reference compositions (elements and known binaries) in the last rows.
2. Describe the phase field in `configs/system/<name>.yaml`, as `configs/system/LiSnSCl.yaml` does:

   | key | value |
   |---|---|
   | `inputfile` | the CSV, relative to the repository root |
   | `reference_index` | row where the reference compositions start |
   | `ions` | element: oxidation state, e.g. `{Li: 1, Sn: 4, S: -2, Cl: -1}` |
   | `max_atoms` | most atoms per cell in generated candidates |
   | `limits` | element: `[min, max)` atoms per generated cell; `null` allows up to `max_atoms` |
   | `compositionfile` | candidate formulas to rank, e.g. an edited `phasebo-generate` output; `null` generates them |
   | `excludefile` | formulas left out of the hull and the candidates; `null` for none |

3. Run a command, overriding any value Hydra-style:

   ```bash
   uv run phasebo-suggest system=LiSnSCl bo.acquisition=ts
   uv run phasebo-benchmark n_starts=8 seed=1 run_name=ts-8-starts
   uv run phasebo-benchmark -m bo.acquisition=qlogei,ts seed=0,1,2 'run_name=sweep/${hydra:job.num}'
   ```

   `tracker=wandb` mirrors a run to Weights & Biases; the run directory stays the source of truth.

## Configs

`configs/defaults/<command>.yaml` lists every setting of a command with its default and a comment: the Gaussian
process (`bo.kernel`, `bo.noise_mev`, ...), batch size, seeding, logging, tracking. It is generated from the schemas
in `src/phasebo/config.py`, so it is never edited. To change a value:

| for | edit |
|---|---|
| one run | `key=value` on the command line; the run directory records it |
| another phase field | a file in `configs/system/`, chosen with `system=<name>` |
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
