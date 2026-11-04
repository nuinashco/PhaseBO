# MLIP-CSP of Li-Sn-S-Cl: the raw data of `LiSnSCl_fastchgnet_csp`

A cheap fidelity for multi-fidelity BO on the real Li-Sn-S-Cl field. Every one of its 165 four-element
compositions gets a crystal structure prediction (CSP) with a machine-learned potential (MLIP), and no DFT:

- **One trial:** a random symmetric structure of the composition's smallest DFT cell (pyxtal, a random
  compatible space group), relaxed with the cell free (ASE FIRE + FrechetCellFilter, fmax 0.05 eV/Å, at most 400
  steps).
- **The search:** 40 trials per composition. The best usable energy after n trials gives the fidelity
  `fastchgnet_random<n>`.
- **References:** Materials Project structures of the elements, binaries and ternaries (E_above_hull ≤ 0.1 eV),
  relaxed by the same MLIP. So every energy on the MLIP hull comes from one model.

This is not `phasebo-csp`. That command places atoms on the holes of a fixed framework and searches the
arrangements (wurtzite nitrides). This search places atoms freely. It is its own project, with its own
environment (Python 3.12; pyxtal and FastCHGNet), which is why `make check` leaves this folder out.

## The run behind the published table

- **MLIP:** FastCHGNet (github.com/HPC-AI-Team/FastCHGNet) running the CHGNet 0.3.0 weights (`--model fastchgnet-0.3.0`).
  - It gives CHGNet's energies about 1.4× faster.
  - FastCHGNet's own "fast" model was rejected: 368 meV/atom off in formation energy.
- **Accuracy:** formation-energy error against Materials Project GGA/GGA+U, on the 44 Li-Sn-S-Cl compounds:

  | MLIP | error | CSP speed (CPU) |
  |---|---|---|
  | MACE-MP-0 medium | 34 meV/atom | 1× |
  | CHGNet 0.3.0 | 42 | ~4× |
  | FastCHGNet, CHGNet 0.3.0 weights | 42 | ~6× |
  | FastCHGNet "fast" | 368 | ~35× |

- **Cost:** 167 cells × 40 trials (the 165 compositions, and the 2 DFT-hull references Materials Project lacks)
  took about 2.8 h of wall time on a 16-thread laptop CPU (15 workers). The per-trial `time` in `trials.jsonl` is
  inflated by the workers sharing the CPU.
- **Settings:** in `data/raw/LiSnSCl_fastchgnet_csp/run.json`.

Agreement with DFT by trials per composition. ρ is the Spearman rank correlation of E_hull with DFT's. The ranks
are the MLIP ranks of the 3 DFT-stable phases (Li3SnS3Cl, Li5SnS4Cl, Li8Sn(SCl)4) among the 165 compositions:

| trials | 10 | 15 | 20 | 25 | 30 | 40 |
|---|---|---|---|---|---|---|
| ρ | 0.38 | 0.45 | 0.53 | 0.56 | 0.56 | 0.56 |
| ranks | 3, 147, 64 | 5, 28, 8 | 7, 34, 1 | 8, 43, 2 | 9, 45, 3 | 11, 51, 5 |

**What the numbers mean:**
- **ρ stops rising at about 25 trials.** That is the MLIP's own error, not the search.
- **Trial order matters below about 30 trials.** `python -m csp.reshuffle` reorders each composition's trials 200
  times. "MLIP-screen everything, then DFT the best" finds all 3 stable phases within 50 DFT lookups in:
  - 25% of orders at 10 trials;
  - 44% at 20;
  - 77% at 30.

  DFT-only BO needs about 106.

**No MLIP structure is stable.** The MLIP puts every composition above its hull: Li3SnS3Cl at 30 meV/atom,
Li5SnS4Cl at 55 and Li8Sn(SCl)4 at 25. So with the MLIP fidelity, "targets" are the DFT-stable phases.

## Steps

From this folder, with its own environment (`uv sync` here, then `uv run python -m csp.<step>`).
- **`--model`:** every step takes it, for `mace-medium` (default), `mace-small`, `mace-large`, `chgnet`,
  `fastchgnet` or `fastchgnet-0.3.0`.
- **Outputs:** they go to `out/<model>/` (gitignored). `out/compositions.csv`, `out/refs_dft.csv`,
  `out/refs_missing.csv` and the Materials Project download are shared.

| step | command | writes |
|---|---|---|
| 1. compositions, the DFT hull | `python -m csp.data` | `out/compositions.csv`, `out/refs_dft.csv` |
| 2. references (`MP_API_KEY` in the repository's `.env`) | `python -m csp.references` | `out/<model>/refs_mlip.csv`, `out/refs_missing.csv` |
| 2b. the MLIP against Materials Project | `python -m csp.references --check` | `out/<model>/refs_mlip_vs_mp.csv` |
| 3. CSP | `python -m csp.run_csp --all --trials 40` | `out/<model>/csp/<cell>.jsonl`, a line per trial |
| 4. the MLIP hull by trials | `python -m csp.hull --checkpoints 10 20 40` | `out/<model>/LiSnSCl_mlipcsp.csv` |
| 5. reliability by trial order | `python -m csp.reshuffle` | `out/<model>/reshuffle.csv`, `figures/fig6_reliability.png` |
| 6. structures for a DFT check | `python -m csp.select_dft --column MLIP_b40` | `out/<model>/dft_check/` (CIF, POSCAR) |
| 7. raw data for the repository | `python -m csp.export` | `../../data/raw/LiSnSCl_<mlip>_csp/` |

`csp.summary_figure`, `csp.meeting_figures`, `csp.hull_maps` and `csp.plot_pilot` draw the figures. Then, from the
repository root:

    uv run scripts/process_LiSnSCl_fastchgnet_csp.py
    uv run phasebo-benchmark system=LiSnSCl_fastchgnet_csp benchmark/cheap=table \
      benchmark.cheap.fidelity=fastchgnet_random30 bo/acquisition=screen

**Usable trials.** A trial counts as a result if both hold:
- it converged, or ran its 400 steps;
- its closest contact is at least 0.85 of the summed covalent radii, and it has at most 80 Å³ per atom.

MACE-MP has unphysical deep minima at short range: −8 and −20 eV/atom, against about −4. A relaxation that falls
below 0.5 of the radii is stopped early. Discarded trials still count against the budget. The thresholds are in
`csp/common.py`.

**Resuming.** Every (cell, trial) has its own seed, from `--seed` and the cell's name, and `run_csp` skips the
trials a cell's file already has. So an interrupted run continues where it stopped, and a longer run
(`--trials 60`) adds trials. Trials run breadth-first: every cell gets trial n before any gets n + 1.

## On a GPU

1. Copy the repository and `out/` to the GPU machine. On Linux, `uv sync` installs torch's CUDA build.
2. Benchmark: `uv run python -m csp.run_csp --cells Li19Sn1S11Cl1 --trials 4 --device cuda --workers 2`. One
   process leaves the GPU mostly idle on small cells, so try 2-4 workers.
3. Run: `uv run python -m csp.run_csp --all --trials 40 --device cuda --workers 3`, in `tmux` or `nohup`.
4. Copy `out/<model>/csp/` back. Files from several machines merge, since trials are keyed by index.
