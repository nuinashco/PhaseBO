"""A fake MLIP whose package is missing."""

import pytest
from attrs import define

from phasebo.csp.mlip.base import MLIP, MLIPConfig


@define
class AbsentMLIP(MLIP[MLIPConfig]):
    """An MLIP whose package is not installed."""

    kind = "absent"
    package = "no-such-package-phasebo"

    @property
    def model(self) -> str:
        return "none"

    def calculator(self, *, device):
        raise NotImplementedError


@pytest.fixture
def absent_mlip() -> AbsentMLIP:
    return AbsentMLIP(MLIPConfig(kind="absent"))
