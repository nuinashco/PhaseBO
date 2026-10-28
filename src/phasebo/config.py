"""Schemas of the entry configs; `make configs` mirrors them to configs/defaults/."""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BeforeValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
    SerializeAsAny,
    model_validator,
)

from nuikit.config import Schema
from nuikit.configgen import ConfigFile
from nuikit.logging import LogFormat, LogLevel
from nuikit.registry import ConfigurableFactory

from phasebo.benchmark.cheap.base import CheapConfig, CheapFactory
from phasebo.benchmark.metric.base import MetricConfig, MetricFactory
from phasebo.benchmark.start.base import StartConfig, StartFactory
from phasebo.bo.acquisition.base import AcquisitionConfig, AcquisitionFactory
from phasebo.bo.surrogate.base import SurrogateConfig, SurrogateFactory


def _by_kind(factory: type[ConfigurableFactory[Any]]) -> BeforeValidator:
    """Return a validator that hands a family config's mapping to its factory."""

    def validate(value: object) -> object:
        return factory.validate(value) if isinstance(value, Mapping) else value

    return BeforeValidator(validate)


def _by_kinds(factory: type[ConfigurableFactory[Any]]) -> BeforeValidator:
    """Return a validator of a multiple family's configs, each under its own kind."""

    def validate(value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        configs = {key: factory.validate(item) for key, item in value.items()}
        for key, config in configs.items():
            if config.kind != key:
                msg = (
                    f"{key}: kind {config.kind!r} differs from its key; "
                    f"set its kind to {key!r} or select {config.kind!r} instead"
                )
                raise ValueError(msg)
        return configs

    return BeforeValidator(validate)


# Fields holding a family item's config, validated against the item that `kind` names.
type _SurrogateField = Annotated[
    SerializeAsAny[SurrogateConfig], _by_kind(SurrogateFactory)
]
type _AcquisitionField = Annotated[
    SerializeAsAny[AcquisitionConfig], _by_kind(AcquisitionFactory)
]
type _StartField = Annotated[SerializeAsAny[StartConfig], _by_kind(StartFactory)]
type _CheapField = Annotated[SerializeAsAny[CheapConfig], _by_kind(CheapFactory)]
type _MetricsField = Annotated[
    dict[str, SerializeAsAny[MetricConfig]], _by_kinds(MetricFactory)
]


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
    """Phase field: its processed table, elements and candidate space."""

    processed: Path = Field(description="written by a script in scripts/")
    elements: dict[str, ElementConfig]
    candidates: CandidatesConfig


class RunConfig(Schema):
    """Settings every command shares: run directory, seeding, logging, tracking."""

    run_name: str = Field(
        default="${now:%Y-%m-%d_%H-%M-%S}", description="directory under runs/"
    )
    seed: int = 42
    deterministic: bool = Field(
        default=False, description="true: deterministic torch kernels, slower"
    )
    log_level: LogLevel = Field(default="info", description="debug|info|warning|error")
    log_format: LogFormat = Field(
        default="auto",
        description="auto|console|json; auto: console in a terminal, JSON otherwise",
    )
    tracker: Literal["wandb"] | None = Field(
        default=None, description="wandb: mirror the run to W&B; null: run dir only"
    )
    wandb_project: str = "phasebo"
    wandb_entity: str | None = Field(
        default=None, description="null: your default entity"
    )


class BOConfig(Schema):
    """Bayesian optimisation settings: the model, the batch rule, the batch size."""

    surrogate: _SurrogateField
    acquisition: _AcquisitionField
    batch_size: PositiveInt = 4


class ProtocolConfig(Schema):
    """How a benchmark replays BO and scores it."""

    start: _StartField
    repeats: PositiveInt = Field(
        default=300, description="replays, each from its own start"
    )
    max_iter: PositiveInt = Field(default=8, description="BO batches after the start")
    workers: PositiveInt | None = Field(default=None, description="null: one per CPU")
    metric: _MetricsField = Field(default_factory=dict)
    cheap: _CheapField | None = None


class SearchConfig(RunConfig):
    """Settings suggest and benchmark share: the phase field, its rows, BO."""

    system: SystemConfig
    fidelity: str = Field(
        default="dft", description="a value of the processed table's fidelity column"
    )
    bo: BOConfig


class SuggestConfig(SearchConfig):
    """Entry config of `phasebo-suggest` (configs/suggest.yaml)."""


class BenchmarkConfig(SearchConfig):
    """Entry config of `phasebo-benchmark` (configs/benchmark.yaml)."""

    benchmark: ProtocolConfig


class GenerateConfig(RunConfig):
    """Entry config of `phasebo-generate` (configs/generate.yaml)."""

    system: SystemConfig


class CompareConfig(RunConfig):
    """Entry config of `phasebo-compare` (configs/compare.yaml)."""

    runs: list[str] = Field(
        default_factory=list,
        description="benchmark runs under runs/, by name; the first is the baseline",
    )
    wandb: list[str] = Field(
        default_factory=list, description="W&B benchmark run ids, read after `runs`"
    )
    level: float = Field(default=0.95, gt=0, lt=1, description="of each interval")
    resamples: PositiveInt = Field(
        default=10_000, description="bootstrap, per interval"
    )


class DataConfig(Schema):
    """Entry config composed once per phase field (configs/data.yaml)."""

    system: SystemConfig


GENERATED = (
    ConfigFile(
        "defaults/suggest.yaml",
        SuggestConfig,
        groups=("system", "bo/surrogate", "bo/acquisition"),
    ),
    ConfigFile(
        "defaults/benchmark.yaml",
        BenchmarkConfig,
        groups=(
            "system",
            "bo/surrogate",
            "bo/acquisition",
            "benchmark/start",
            "benchmark/metric",
            "benchmark/cheap",
        ),
    ),
    ConfigFile("defaults/generate.yaml", GenerateConfig, groups=("system",)),
    ConfigFile("defaults/compare.yaml", CompareConfig),
    ConfigFile("defaults/data.yaml", DataConfig, groups=("system",)),
)
