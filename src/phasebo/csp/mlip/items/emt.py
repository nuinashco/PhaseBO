"""EMT: ASE's built-in toy potential, for tests and smoke runs, not real energies."""

from typing import TYPE_CHECKING, Literal

from attrs import define

from phasebo.csp.mlip.base import MLIP, MLIPConfig, MLIPFactory

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator


class EMTConfig(MLIPConfig):
    """EMT settings: none."""


@MLIPFactory.register("emt")
@define
class EMTMLIP(MLIP[EMTConfig]):
    """Effective medium theory for Al, N and a few metals, with no download."""

    package = "ase"
    uses_device = False

    @property
    def model(self) -> str:
        return "emt"

    def calculator(self, *, device: Literal["cpu", "cuda"]) -> Calculator:
        from ase.calculators.emt import EMT

        return EMT()
