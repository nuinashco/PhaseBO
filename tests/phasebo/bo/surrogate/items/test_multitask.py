"""The multi-task GP builds its data kernel from the configured name."""

from phasebo.bo.surrogate.base import CheapEnergies, SurrogateFactory


def test_multitask_passes_its_kernel_name_to_the_data_kernel(points):
    pool = points.pool
    cheap = CheapEnergies(
        coords=points.coords[pool], energies_mev=points.energies_mev[pool]
    )
    surrogate = SurrogateFactory.from_config(
        {"kind": "multitask", "kernel": "matern12"}
    )

    fit = surrogate.fit(
        points.coords[pool[:3]], points.energies_mev[pool[:3]], cheap=cheap
    )

    data_kernel = fit.model.covar_module.kernels[0]
    assert data_kernel.base_kernel.nu == 0.5
