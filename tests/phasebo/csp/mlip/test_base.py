"""An MLIP names its package and weights, and fails early when it cannot run."""

import pytest
import torch

from nuikit.errors import ConfigError

from phasebo.csp.mlip.base import Potential, resolve_device
from phasebo.csp.mlip.items.emt import EMTMLIP, EMTConfig


def test_describe_names_the_kind_package_version_and_weights():
    mlip = EMTMLIP(EMTConfig(kind="emt"))

    potential = mlip.describe()

    assert (potential.kind, potential.model) == ("emt", "emt")
    assert isinstance(potential, Potential) and potential.version


def test_describe_tells_how_to_install_a_missing_package(absent_mlip):
    with pytest.raises(ConfigError, match="uv sync --group csp"):
        absent_mlip.describe()


def test_resolve_device_picks_cuda_only_when_torch_sees_a_gpu():
    expected = "cuda" if torch.cuda.is_available() else "cpu"

    assert resolve_device("auto") == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_resolve_device_refuses_cuda_without_a_gpu():
    with pytest.raises(ConfigError, match="no GPU"):
        resolve_device("cuda")
