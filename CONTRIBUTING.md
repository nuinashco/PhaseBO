# Contributing

How phasebo is built and how to change it. Using it: [README.md](README.md). Conventions that apply to every project,
with their reasons: [CONVENTIONS.md](CONVENTIONS.md) (a read-only copy of the guide; `copier update` refreshes it).
The `conventions-reviewer` subagent in `.claude/agents/` checks a branch or the repo against it.

## Architecture

```mermaid
flowchart TD
    subgraph project["phasebo"]
        cli["cli/<br/>one module per console script"]:::side
        config["config.py<br/>entry config schemas"]:::pure
        bo["phase_field_bo.py<br/>GP fit and batch picks, path replay"]:::legacy
        field["phase_field.py<br/>convex hull, coordinates, starts"]:::legacy
        readers["dataset/base.py, dataset/items/<br/>dataset readers, one per raw layout"]:::side
        table["dataset/table.py<br/>clean, validate, render, parse the processed table"]:::pure
        candidates["candidates.py<br/>charge-balanced, unexplored cells"]:::pure
        paths["paths.py<br/>repository paths"]:::side
        cli --> config
        cli --> bo
        cli --> table
        cli --> readers
        cli --> candidates
        config --> readers
        bo --> field
        readers --> paths
    end
    nuikit["nuikit<br/>config, configgen, registry, errors,<br/>logging, runs, seeding"]:::kit
    project --> nuikit
    pure["teal: decides, same input same output"]:::pure
    side["rust: acts on files, processes, network"]:::side
    legacy["amber: pre-template, decides and acts"]:::legacy
    kit["grey: copied from the conventions template"]:::kit
    classDef pure fill:#d9f2ec,stroke:#1b7f6b,color:#0b3d33
    classDef side fill:#f8e3d8,stroke:#b4532a,color:#4a1f0c
    classDef legacy fill:#fbefd0,stroke:#a77a12,color:#3d2c05
    classDef kit fill:#ececec,stroke:#6b6b6b,color:#262626
```

`phasebo` holds this project's code and nothing else. The two amber modules predate the template: they compute
and write files in the same classes, so they have no pure half yet. They are rewritten into families (surrogate,
acquisition, starting set, policy, metric) and then removed; until then the commands call them directly, feeding
them the processed table through `_launch.to_legacy_inputs`.

`nuikit` is the machinery every project built from the template shares: what schemas and families build on, and
what an entry point sets up before it runs. Copier writes it; fixes go upstream and arrive with `copier update`.

Dependencies point down: `cli` > `config` > `phase_field_bo` | `dataset` | `candidates` > `phase_field` > `paths`, and
any layer may import `nuikit`. The import-linter contracts in `pyproject.toml` enforce the order and keep `nuikit` from
importing `phasebo`. The layers contract is exhaustive, so a new subpackage fails `make check` until it is placed in
`layers`.

### A command

```mermaid
sequenceDiagram
    participant H as Hydra
    participant C as cli/benchmark.py
    participant L as cli/_launch.py
    participant B as PhaseFieldBO
    H->>C: compose configs/benchmark.yaml + overrides
    C->>C: BenchmarkConfig.from_hydra (validate)
    C->>L: read_table: data/processed/<field>.csv, validated
    C->>L: to_legacy_inputs: (formula, total energy) rows, references
    C->>L: fingerprint: sha256 of the processed and raw files, which must exist
    C->>L: start: logging, seed, runs/<run_name>/, W&B mirror
    C->>L: write_data: data.json in the run directory
    C->>B: PhaseFieldBO(mode="path", ...): hull, starts, BO batches
    C->>B: print_results, plots
    B-->>C: bo_path.csv, BO_Path_in_<field>.txt, plots in the run directory
```

## Repository layout

```text
configs/                  Hydra YAML; generated files say so in their first line
  suggest.yaml            entry configs: pick the mirror, the phase field and Hydra's settings
  benchmark.yaml
  generate.yaml
  data.yaml               composed by phasebo-data once per phase field
  defaults/               every setting of each entry config with its default (generated)
  dataset/kinds/          every setting of each raw layout (generated)
  system/                 phase fields: dataset, processed table, elements, candidate space
  hydra/default.yaml      Hydra writes nothing; the app writes runs/
data/                     tracked (see Deviations)
  raw/                    datasets as published, never edited
  processed/              one table per phase field, written by `make data`
src/phasebo/
  cli/                    one module per console script, plus _launch.py, their shared start-up
  config.py               entry config schemas + GENERATED (files `make configs` writes)
  dataset/                the dataset family (base.py, items/) and table.py, the processed table
  candidates.py           charge-balanced cells, filtered to unexplored points
  paths.py                repository paths, from the .project-root marker
  phase_field.py          pre-template: hull, coordinates, starting compositions
  phase_field_bo.py       pre-template: BoTorch GP, batch picks, replay, result files
src/nuikit/               copied from the conventions template: config, configgen, registry, errors, logging,
                          runs, seeding
tests/                    mirrors src/; tests/nuikit/ comes with the template
  test_invariants.py      configs compose and are fresh, kinds build, import layers hold
  phasebo/                tests of the new modules
  test_phase_field*.py    pre-template tests, with conftest.py's synthetic phase fields
  integration/            end to end, `-m integration`
scripts/                  exploratory or one-time runnables, never imported; rerun ones become cli/ commands
notebooks/                exploration only, outputs stripped; reference/ for third-party notebooks
runs/                     gitignored
```

## Families

A family is a concept with several implementations chosen by a config `kind`.

| family | kinds | chosen by |
|---|---|---|
| `dataset` | `total_energy_csv`: formula, total energy (eV per cell); `hull_energy_csv`: formula, energy above the hull (meV/atom) per fidelity | `system.dataset.kind` |

The rewrite of the pre-template modules adds the BO families.

```text
src/phasebo/<family>/
  base.py        <Family> (attrs ABC), <Family>Config, <Family>Factory (config_group = "<family>")
  items/
    __init__.py  import_submodules(__name__)
    <kind>.py    @<Family>Factory.register("<kind>") above @define
```

`<Family>Factory.from_config({"kind": ..., ...})` validates against the item's config and builds it; `make configs`
writes `configs/<family>/kinds/<kind>.yaml`.

## Extending

| to add | create | and |
|---|---|---|
| a phase field | its raw file in `data/raw/` and `configs/system/<name>.yaml` | `make data`; run with `system=<name>` |
| a raw layout | `src/phasebo/dataset/items/<kind>.py`, a `<Name>Dataset` whose `read()` returns `dataset.base.COLUMNS` | `make configs` |
| a family | `src/phasebo/<family>/base.py` + `items/` | place it in the import-linter `layers`; `make configs` |
| an item | `src/phasebo/<family>/items/<kind>.py` | `make configs` |
| an entry config | a schema in `config.py` listed in `GENERATED` as `defaults/<command>.yaml`, and `configs/<command>.yaml` composing it | a `cli/<command>.py` and a console script |
| a setting | a field with a default on the schema, described only if its name leaves something open (DOC-7) | `make configs` |

## Deviations

- **Pre-template modules** (`phase_field.py`, `phase_field_bo.py`, and their tests `tests/conftest.py`,
  `tests/test_phase_field.py`, `tests/test_phase_field_bo.py`) are excluded from ruff, pyright and pre-commit. They
  break TYP-4 (no attrs, `__init__` runs the whole computation), ARC-2 (no pure half), RUN-2 (global `random`),
  LOG-2 (f-string events) and STY-8 (non-ASCII). Making them pass would be thrown away: the `feat/bo-families`
  branch replaces them, then drops these exclusions. The new command tests
  use the `li_sn_s_cl` fixture of `tests/conftest.py`, so the rewrite moves that fixture rather than deleting it.
- **`data/` is tracked** (RUN-6 says gitignored). The raw datasets are small and published with the paper, and every
  number in the docs must trace back to the exact file; pre-commit's whitespace fixers leave them byte for byte.
  Processed tables are committed too, so a reader change shows in review as the rows it changes; the invariant tests
  fail when one is stale.
- **Processed tables are CSV** (RUN-7 prefers parquet for derived tables): CSV diffs line by line in review.
- **Notebooks are marimo files** (RUN-5 describes `.ipynb`), installed with the `notebooks` dependency group. They
  are plain Python with no stored outputs, so nbstripout has nothing to strip; `__marimo__/` caches are gitignored.
- **`cli/_launch.py`** is a module in `cli/` that is not a console script (LAY-7). It holds the commands' shared
  start-up, which needs the config schemas and so sits in the only layer above `config`, and the adapters from the
  configs and the processed table to the pre-template class, which go when that class goes.

## Tests

```bash
make test               # unit tests, a few seconds
make test-integration   # end to end, behind the integration marker
```

Fakes, not mocks (`unittest.mock` is banned by ruff). Each case covers a distinct behaviour; child processes and
servers only for what cannot be measured in process. The commands are tested through their `run(cfg, runs_dir=...)`
functions; `tests/test_invariants.py` covers their Hydra configs.

## Commits

Conventional Commits with the standard types (`feat fix docs refactor test build ci perf style chore revert`), an
optional scope, a lowercase imperative subject: one line, no articles, no body or trailers. Short-lived `<type>/<slug>`
branches, squash-merged. The commit-msg hooks check all of it. PR descriptions follow
`.github/pull_request_template.md`; `make github`, run once per repo, makes the squash commit the PR title alone.
