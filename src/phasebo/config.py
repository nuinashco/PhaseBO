"""Schemas of the entry configs; `make configs` mirrors them to configs/defaults/."""

from pathlib import Path
from typing import Literal, Self

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from nuikit.config import Schema
from nuikit.configgen import ConfigFile
from nuikit.logging import LogFormat, LogLevel


class SystemConfig(Schema):
    """Phase field: its computed compositions, ions and candidate space."""

    inputfile: Path = Field(
        description="CSV of formulas and total energies (eV per cell), references last"
    )
    reference_index: NonNegativeInt = Field(description="first reference row")
    ions: dict[str, int] = Field(description="element: oxidation state")
    max_atoms: PositiveInt = Field(description="most atoms per generated cell")
    limits: dict[str, tuple[NonNegativeInt, NonNegativeInt]] | None = Field(
        default=None,
        description="element: [min, max) atoms per generated cell; null: max_atoms",
    )
    compositionfile: Path | None = Field(
        default=None, description="candidates to rank; null: generated"
    )
    excludefile: Path | None = Field(
        default=None, description="compositions left out everywhere; null: none"
    )

    @model_validator(mode="after")
    def _limits_name_the_ions(self) -> Self:
        if self.limits is not None and set(self.limits) != set(self.ions):
            ions, given = sorted(self.ions), sorted(self.limits)
            msg = (
                f"limits must name exactly the ions {ions}, got {given}; give "
                "system.limits one entry per ion or set it to null"
            )
            raise ValueError(msg)
        return self


class BOConfig(Schema):
    """Gaussian process and batch selection."""

    acquisition: Literal["qlogei", "ts"] = Field(
        default="qlogei",
        description="qlogei: batch log expected improvement; ts: Thompson sampling",
    )
    kernel: Literal["matern", "rbf"] = Field(
        default="matern", description="matern: Matern 5/2, as in the paper; rbf"
    )
    noise_mev: NonNegativeFloat | None = Field(
        default=0.0,
        description="energy noise std (meV/atom); 0: DFT is exact; null: fitted",
    )
    batch_size: PositiveInt = Field(default=4, description="compositions per batch")


class RunConfig(Schema):
    """Settings every command shares: run directory, seeding, logging, tracking."""

    run_name: str = Field(
        default="${now:%Y-%m-%d_%H-%M-%S}", description="directory under runs/"
    )
    seed: int = Field(default=42, description="seeds random, numpy and torch")
    deterministic: bool = Field(
        default=False, description="force deterministic torch kernels (slower)"
    )
    log_level: LogLevel = Field(default="info", description="debug|info|warning|error")
    log_format: LogFormat = Field(
        default="auto", description="auto: console in a terminal, JSON otherwise"
    )
    tracker: Literal["wandb"] | None = Field(
        default=None, description="mirror the run to W&B; null: run dir only"
    )
    wandb_project: str = Field(default="phasebo", description="W&B project")


class SuggestConfig(RunConfig):
    """Entry config of `phasebo-suggest` (configs/suggest.yaml)."""

    system: SystemConfig = Field(description="phase field")
    bo: BOConfig = Field(
        default_factory=BOConfig, description="Gaussian process and batch selection"
    )
    show_plots: bool = Field(default=False, description="open plot windows")
    plot_dpi: PositiveInt = Field(default=150, description="resolution of saved plots")


class BenchmarkConfig(RunConfig):
    """Entry config of `phasebo-benchmark` (configs/benchmark.yaml)."""

    system: SystemConfig = Field(description="phase field")
    bo: BOConfig = Field(
        default_factory=BOConfig, description="Gaussian process and batch selection"
    )
    n_starts: PositiveInt = Field(
        default=23, description="random computed compositions the replay starts from"
    )
    max_iter: PositiveInt = Field(default=10, description="batches after the start")
    show_plots: bool = Field(default=False, description="open plot windows")
    plot_dpi: PositiveInt = Field(default=150, description="resolution of saved plots")


class GenerateConfig(RunConfig):
    """Entry config of `phasebo-generate` (configs/generate.yaml)."""

    system: SystemConfig = Field(description="phase field")


GENERATED = (
    ConfigFile("defaults/suggest.yaml", SuggestConfig, groups=("system",)),
    ConfigFile("defaults/benchmark.yaml", BenchmarkConfig, groups=("system",)),
    ConfigFile("defaults/generate.yaml", GenerateConfig, groups=("system",)),
)
