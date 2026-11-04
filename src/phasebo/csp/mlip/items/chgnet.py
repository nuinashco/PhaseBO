"""CHGNet: a universal graph-network potential trained on the Materials Project."""

from typing import TYPE_CHECKING, Literal

from attrs import define
from pydantic import Field

from phasebo.csp.mlip.base import MLIP, MLIPConfig, MLIPFactory

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class CHGNetConfig(MLIPConfig):
    """CHGNet settings."""

    model: Literal["0.3.0", "0.2.0", "r2scan"] = Field(
        default="0.3.0", description="0.3.0|0.2.0|r2scan; weights shipped with chgnet"
    )


@MLIPFactory.register("chgnet")
@define
class CHGNetMLIP(MLIP[CHGNetConfig]):
    """CHGNet, the potential PolyBO's reference runs used."""

    package = "chgnet"

    @property
    def model(self) -> str:
        return self.config.model

    def calculator(self, *, device: Literal["cpu", "cuda"]) -> Calculator:
        from chgnet.model.dynamics import (  # pyright: ignore[reportMissingImports]
            CHGNetCalculator,
        )
        from chgnet.model.model import CHGNet  # pyright: ignore[reportMissingImports]

        weights = CHGNet.load(
            model_name=self.config.model, use_device=device, verbose=False
        )
        return CHGNetCalculator(weights, use_device=device)
