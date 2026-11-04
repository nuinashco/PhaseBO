"""The noisy source adds errors to total energies and rebuilds the hull per draw."""

import numpy as np
import pytest

from nuikit.errors import DataError

from phasebo.benchmark.cheap.base import CheapSourceFactory
from phasebo.field.points import Points


def noisy(**settings):
    return CheapSourceFactory.from_config({"kind": "noisy", **settings})


def test_noisy_source_without_errors_gives_the_energies_above_the_hull(rows, points):
    source = noisy(random_mev=0.0, smooth_mev=0.0)

    energies = source.draw(rows, points, fidelity="dft", rng=np.random.default_rng(0))

    assert energies == pytest.approx(points.energies_mev, abs=1e-6)


def test_noisy_source_repeats_with_its_generator(rows, points):
    source = noisy(random_mev=30.0, smooth_mev=30.0)

    def draw(seed):
        rng = np.random.default_rng(seed)
        return source.draw(rows, points, fidelity="dft", rng=rng)

    assert np.array_equal(draw(1), draw(1))
    assert not np.allclose(draw(1), draw(2))


def test_smooth_error_over_the_whole_field_leaves_the_hull_distances(rows, points):
    # A lengthscale far beyond the field shifts every atom alike; the hull cancels it.
    source = noisy(random_mev=0.0, smooth_mev=50.0, lengthscale=1000.0)

    energies = source.draw(rows, points, fidelity="dft", rng=np.random.default_rng(0))

    assert energies == pytest.approx(points.energies_mev, abs=0.5)


def test_smooth_error_within_the_field_moves_the_energies(rows, points):
    source = noisy(random_mev=0.0, smooth_mev=50.0, lengthscale=0.2)

    energies = source.draw(rows, points, fidelity="dft", rng=np.random.default_rng(0))

    pool = points.pool
    assert not np.allclose(energies[pool], points.energies_mev[pool], atol=1.0)


def test_smooth_error_differs_less_between_nearby_points(rows, points):
    source = noisy(random_mev=0.0, smooth_mev=50.0, lengthscale=0.2)
    pool = points.pool
    fractions = np.column_stack([1 - points.coords.sum(axis=1), points.coords])[pool]
    pairs = np.triu_indices(len(pool), 1)
    distances = np.linalg.norm(fractions[:, None] - fractions[None], axis=-1)[pairs]
    errors = np.array(
        [
            source.draw(rows, points, fidelity="dft", rng=np.random.default_rng(seed))
            - points.energies_mev
            for seed in range(20)
        ]
    )[:, pool]

    gaps = ((errors[:, :, None] - errors[:, None, :]) ** 2).mean(axis=0)[pairs]
    by_distance = gaps[np.argsort(distances)]
    assert by_distance[:5].mean() < by_distance[-5:].mean()


def test_noisy_source_refuses_rows_without_total_energies(nn_rows, points):
    only_nn = nn_rows(range(len(points.formulas)), 0.0)
    nn_points = Points.from_rows(only_nn, fidelity="nn", elements=points.elements)

    with pytest.raises(DataError, match="needs total energies"):
        noisy().check(only_nn, nn_points, fidelity="nn")


def test_noisy_source_refuses_points_without_rows(rows, points):
    lost = points.names[points.pool[0]]

    with pytest.raises(ValueError, match="have no 'dft' rows"):
        noisy().check(rows.loc[rows["point"] != lost], points, fidelity="dft")
