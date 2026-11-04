"""MACE-MP: an equivariant foundation potential trained on the Materials Project."""

from typing import TYPE_CHECKING, Literal

from attrs import define
from pydantic import Field

from phasebo.csp.mlip.base import MLIP, MLIPConfig, MLIPFactory

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class MACEConfig(MLIPConfig):
    """MACE-MP settings."""

    model: str = Field(
        default="medium",
        description="small|medium|large or a newer name mace_mp knows; downloaded once",
    )
    dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="float32|float64; on an L4 float32 matched float64, 2-4x faster",
    )


@MLIPFactory.register("mace")
@define
class MACEMLIP(MLIP[MACEConfig]):
    """MACE-MP-0 through `mace_mp`."""

    package = "mace-torch"

    @property
    def model(self) -> str:
        return self.config.model

    def calculator(self, *, device: Literal["cpu", "cuda"]) -> Calculator:
        from mace.calculators import mace_mp  # pyright: ignore[reportMissingImports]

        return mace_mp(
            model=self.config.model, device=device, default_dtype=self.config.dtype
        )
