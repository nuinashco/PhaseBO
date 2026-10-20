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


class DataConfig(Schema):
    """Computed compositions of a phase field."""

    file: Path = Field(
        description="CSV of formulas and total energies (eV per cell), references last"
    )
    reference_index: NonNegativeInt = Field(description="first reference row")


class ElementConfig(Schema):
    """One element of a phase field."""

    charge: int
    atoms: tuple[NonNegativeInt, NonNegativeInt] | None = Field(
        default=None,
        description="[min, max] atoms per generated cell; null: up to max_atoms",
    )

    @model_validator(mode="after")
    def _atoms_ascend(self) -> Self:
        if self.atoms is not None and self.atoms[0] > self.atoms[1]:
            msg = f"atoms {list(self.atoms)} is not [min, max]; put the minimum first"
            raise ValueError(msg)
        return self


class CandidatesConfig(Schema):
    """Compositions BO may pick from."""

    max_atoms: PositiveInt = Field(description="per generated cell")
    file: Path | None = Field(
        default=None, description="CSV of formulas to rank; null: generated"
    )
    exclude: Path | None = Field(
        default=None,
        description="CSV of formulas left out of data and candidates; null: none",
    )


class SystemConfig(Schema):
    """Phase field: its computed compositions, elements and candidate space."""

    data: DataConfig
    elements: dict[str, ElementConfig]
    candidates: CandidatesConfig


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
        description="std in meV/atom; null: fitted",
    )
    batch_size: PositiveInt = 4


class RunConfig(Schema):
    """Settings every command shares: run directory, seeding, logging, tracking."""

    run_name: str = Field(
        default="${now:%Y-%m-%d_%H-%M-%S}", description="directory under runs/"
    )
    seed: int = 42
    deterministic: bool = Field(default=False, description="torch kernels; slower")
    log_level: LogLevel = Field(default="info", description="debug|info|warning|error")
    log_format: LogFormat = Field(
        default="auto", description="auto|console|json; auto: console in a terminal"
    )
    tracker: Literal["wandb"] | None = Field(
        default=None, description="wandb: mirror the run to W&B; null: run dir only"
    )
    wandb_project: str = "phasebo"


class SuggestConfig(RunConfig):
    """Entry config of `phasebo-suggest` (configs/suggest.yaml)."""

    system: SystemConfig
    bo: BOConfig = Field(default_factory=BOConfig)
    show_plots: bool = False
    plot_dpi: PositiveInt = 150


class BenchmarkConfig(RunConfig):
    """Entry config of `phasebo-benchmark` (configs/benchmark.yaml)."""

    system: SystemConfig
    bo: BOConfig = Field(default_factory=BOConfig)
    n_starts: PositiveInt = Field(
        default=23, description="random computed compositions before BO"
    )
    max_iter: PositiveInt = Field(default=10, description="BO batches after the starts")
    show_plots: bool = False
    plot_dpi: PositiveInt = 150


class GenerateConfig(RunConfig):
    """Entry config of `phasebo-generate` (configs/generate.yaml)."""

    system: SystemConfig


GENERATED = (
    ConfigFile("defaults/suggest.yaml", SuggestConfig, groups=("system",)),
    ConfigFile("defaults/benchmark.yaml", BenchmarkConfig, groups=("system",)),
    ConfigFile("defaults/generate.yaml", GenerateConfig, groups=("system",)),
)
