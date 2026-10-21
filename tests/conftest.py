"""In-process fakes of the BO families, for the loops that drive them."""

import numpy as np
import pytest
from attrs import define, field

from phasebo.bo.acquisition.base import Acquisition, AcquisitionConfig, require_fit
from phasebo.bo.surrogate.base import Fit, Surrogate, SurrogateConfig


@define
class RecordingSurrogate(Surrogate[SurrogateConfig]):
    """Surrogate that records how many points each fit gets, and models nothing."""

    sizes: list[int] = field(factory=list)

    def fit(self, coords: np.ndarray, energies_mev: np.ndarray) -> Fit:
        self.sizes.append(len(coords))
        return Fit(model=None, best_f=float(-energies_mev.min()))


@define
class FirstChoices(Acquisition[AcquisitionConfig]):
    """Model-based rule that picks the first choices, once given a fit."""

    def pick(
        self,
        fit: Fit | None,
        choices: np.ndarray,
        *,
        size: int,
        rng: np.random.Generator,
    ) -> list[int]:
        require_fit(fit, self.config.kind)
        return list(range(size))


@pytest.fixture
def recording_surrogate() -> RecordingSurrogate:
    return RecordingSurrogate(SurrogateConfig(kind="recording"))


@pytest.fixture
def first_choices() -> FirstChoices:
    return FirstChoices(AcquisitionConfig(kind="first"))
