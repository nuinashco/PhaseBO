"""Interatomic potentials: each hands the search an ASE calculator."""

import importlib.metadata
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from attrs import define, frozen

from nuikit.errors import ConfigError
from nuikit.registry import Configurable, ConfigurableFactory, KindConfig

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

type Device = Literal["auto", "cpu", "cuda"]


@frozen
class Potential:
    """The MLIP behind a result: its kind, its package's version and its weights."""

    kind: str
    version: str
    model: str


class MLIPConfig(KindConfig):
    """Settings of an interatomic potential."""


@define
class MLIP[C: MLIPConfig](Configurable[C], ABC):
    """Interatomic potential that gives each arrangement its energy and forces."""

    # The distribution whose version every result records.
    package: ClassVar[str]
    # Whether it runs on torch: `device` applies, and a GPU takes a single process.
    uses_device: ClassVar[bool] = True

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the weights' name, which every result records."""

    @abstractmethod
    def calculator(self, *, device: Literal["cpu", "cuda"]) -> Calculator:
        """Return a fresh calculator on `device`."""

    def describe(self) -> Potential:
        """Return what results record of this MLIP; fail if its package is missing."""
        try:
            version = importlib.metadata.version(self.package)
        except importlib.metadata.PackageNotFoundError as e:
            raise ConfigError(
                f"{self.package} is not installed for csp.mlip {self.kind!r}; "
                "run `uv sync --group csp`"
            ) from e
        return Potential(kind=self.kind, version=version, model=self.model)


class MLIPFactory(ConfigurableFactory[MLIP[Any]]):
    """Registry of interatomic potentials by `kind`."""

    config_group = "csp/mlip"


def resolve_device(device: Device) -> Literal["cpu", "cuda"]:
    """Return `device`, with auto as cuda when torch sees a GPU and cpu otherwise.

    Auto never picks Apple's MPS: PolyBO found it slower than the CPU for these cells.
    """
    import torch

    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise ConfigError(
            "csp.device is cuda but torch sees no GPU; set csp.device=auto or "
            "install CUDA torch as the README says"
        )
    if device == "auto":
        return "cuda" if available else "cpu"
    return device
