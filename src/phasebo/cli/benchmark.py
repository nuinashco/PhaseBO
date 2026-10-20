"""`phasebo-benchmark`: replay BO over the computed compositions, as if computing them.

Usage:

    phasebo-benchmark                          # configs/benchmark.yaml as it is

    phasebo-benchmark n_starts=8 seed=1        # Hydra overrides

    phasebo-benchmark -m seed=0,1,2 'run_name=sweep/${hydra:job.num}'

Writes runs/<run_name>/ with bo_path.csv (the starting compositions, then each pick
and its energy above the hull), convergence.png and convex_hull.png.
"""

from pathlib import Path

import hydra
import rootutils
from omegaconf import DictConfig

from nuikit.errors import cli_boundary

from phasebo import paths
from phasebo.cli import _launch
from phasebo.config import BenchmarkConfig

# Loads .env and sets PROJECT_ROOT, as every entry point does (GUIDE ENV-2).
rootutils.setup_root(__file__, indicator=".project-root")


@hydra.main(config_path=str(paths.CONFIGS_DIR), config_name="benchmark")
@cli_boundary
def main(dict_cfg: DictConfig) -> None:
    run(BenchmarkConfig.from_hydra(dict_cfg), runs_dir=paths.RUNS_DIR)


def run(cfg: BenchmarkConfig, *, runs_dir: Path) -> Path:
    """Pick max_iter batches from the computed compositions after n_starts at random."""
    from phasebo.phase_field_bo import PhaseFieldBO

    compositions, references = _launch.to_legacy_inputs(_launch.read_table(cfg.system))
    excluded = _launch.read_formulas(
        cfg.system.candidates.exclude, key="system.candidates.exclude"
    )
    fingerprints = _launch.fingerprint(cfg.system)
    run_dir = _launch.start(cfg, runs_dir=runs_dir)
    _launch.write_data(run_dir, fingerprints)
    bopt = PhaseFieldBO(
        compositions=compositions,
        references=references,
        ions=_launch.map_charges(cfg.system),
        mode="path",
        n_seeds=cfg.n_starts,
        # Stable compositions never start a replay: finding them is what it measures.
        exclude_zeros=True,
        max_iter=cfg.max_iter,
        batch=cfg.bo.batch_size,
        acquisition=cfg.bo.acquisition,
        kernel=cfg.bo.kernel,
        noise=cfg.bo.noise_mev,
        exceptions=excluded,
        output_dir=str(run_dir),
    )
    bopt.print_results()
    bopt.plot_convergence()
    _launch.save_plot(
        run_dir / "convergence.png", dpi=cfg.plot_dpi, show=cfg.show_plots
    )
    bopt.plot_convex()
    _launch.save_plot(
        run_dir / "convex_hull.png", dpi=cfg.plot_dpi, show=cfg.show_plots
    )
    return run_dir
