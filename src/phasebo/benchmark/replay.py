"""Benchmark replays: BO picks among the computed points, as if computing them."""

import functools
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import pandas as pd
from attrs import field, frozen
from pymatgen.core import Composition

from nuikit.errors import DataError

from phasebo.benchmark.start.base import Start
from phasebo.bo.acquisition.base import Acquisition
from phasebo.bo.surrogate.base import CheapEnergies, Surrogate
from phasebo.field.points import Points


class CheapDraw(Protocol):
    """Source of each point's cheap energy above the hull (meV/atom) for one repeat."""

    def __call__(self, *, rng: np.random.Generator) -> np.ndarray: ...


@frozen
class Replays:
    """Replays over one field, each from its own starting set, in full batches."""

    points: Points
    batch_size: int
    max_iter: int
    # One row per evaluation: `repeat`, `batch` (0 for the starting set), `point`.
    paths: pd.DataFrame = field(eq=False)

    @classmethod
    def from_table(
        cls, points: Points, table: pd.DataFrame, *, batch_size: int, max_iter: int
    ) -> Replays:
        """Return the replays that `table()` wrote, over the same `points`."""
        index = {formula: i for i, formula in enumerate(points.formulas)}
        unknown = sorted(set(table["formula"]) - set(index))
        if unknown:
            raise DataError(
                f"{len(unknown)} evaluated formulas, such as {unknown[0]}, are not "
                "points of this field; compare runs on the table they ran on"
            )
        paths = pd.DataFrame(
            {
                "repeat": table["repeat"].to_numpy(),
                "batch": table["batch"].to_numpy(),
                "point": [index[formula] for formula in table["formula"]],
            }
        )
        return cls(points=points, batch_size=batch_size, max_iter=max_iter, paths=paths)

    @property
    def repeats(self) -> int:
        """Return the number of replays."""
        return len(np.unique(self.paths["repeat"].to_numpy()))

    @property
    def budget(self) -> int:
        """Return the evaluations each replay makes after its starting set."""
        return self.batch_size * self.max_iter

    @property
    def evaluations(self) -> np.ndarray:
        """Return the evaluations made after the start, at the end of every batch."""
        return np.arange(self.max_iter + 1) * self.batch_size

    @property
    def pool(self) -> int:
        """Return the explored points left to pick after a starting set."""
        starts = self.paths.loc[self.paths["batch"] == 0].groupby("repeat").size()
        return len(self.points.pool) - int(starts.iloc[0])

    @property
    def target_names(self) -> list[str]:
        """Return each target's reduced formula, in `points.targets` order."""
        return [
            Composition(self.points.formulas[i]).reduced_composition.formula
            for i in self.points.targets
        ]

    def list_starts(self) -> list[tuple[int, ...]]:
        """Return each repeat's starting points, sorted."""
        starts = self.paths.loc[self.paths["batch"] == 0]
        return [
            tuple(sorted(group["point"].tolist()))
            for _, group in starts.groupby("repeat")
        ]

    def hit_batches(self) -> np.ndarray:
        """Return the batch that picked each target, per repeat; inf if none did."""
        targets = self.points.targets
        hits = np.full((self.repeats, len(targets)), np.inf)
        found = self.paths.loc[self.paths["point"].isin(targets.tolist())]
        column = {int(t): j for j, t in enumerate(targets)}
        for repeat, point, batch in found[["repeat", "point", "batch"]].itertuples(
            index=False
        ):
            hits[repeat, column[point]] = batch
        return hits

    def table(self) -> pd.DataFrame:
        """Return every evaluation: repeat, batch, formula, energy, whether a target."""
        point = self.paths["point"].to_numpy()
        targets = set(self.points.targets.tolist())
        return pd.DataFrame(
            {
                "repeat": self.paths["repeat"],
                "batch": self.paths["batch"],
                "formula": [self.points.formulas[i] for i in point],
                "e_above_hull_mev": self.points.energies_mev[point],
                "target": [int(i) in targets for i in point],
            }
        )


def replay_many(
    points: Points,
    *,
    start: Start,
    surrogate: Surrogate,
    acquisition: Acquisition,
    batch_size: int,
    max_iter: int,
    repeats: int,
    seed: int,
    workers: int | None,
    draw_cheap: CheapDraw | None = None,
) -> Replays:
    """Replay BO from `repeats` starting sets, on `workers` processes (None: all CPUs).

    Repeat r draws its starting set, and with `draw_cheap` each point's cheap energy,
    from its own streams of `seed`, so every setting run with one seed starts from
    the same sets and cheap energies, and its results do not depend on `workers`.
    """
    from joblib import Parallel, delayed

    start.check(points, budget=batch_size * max_iter)
    streams = np.random.SeedSequence(seed).spawn(repeats)
    parallel = Parallel(n_jobs=workers or -1)
    runs = parallel(
        delayed(_replay_one)(
            points,
            stream,
            start=start,
            surrogate=surrogate,
            acquisition=acquisition,
            batch_size=batch_size,
            max_iter=max_iter,
            draw_cheap=draw_cheap,
        )
        for stream in streams
    )
    rows = [
        (repeat, batch, point)
        for repeat, batches in enumerate(cast("list[list[list[int]]]", runs))
        for batch, picked in enumerate(batches)
        for point in picked
    ]
    paths = pd.DataFrame(rows, columns=["repeat", "batch", "point"])
    return Replays(points=points, batch_size=batch_size, max_iter=max_iter, paths=paths)


def replay(
    points: Points,
    starting: Sequence[int],
    *,
    surrogate: Surrogate,
    acquisition: Acquisition,
    batch_size: int,
    max_iter: int,
    rng: np.random.Generator,
    cheap_mev: np.ndarray | None = None,
) -> list[list[int]]:
    """Return the batches BO picks after `starting`, up to `max_iter` of them.

    Each batch comes from the explored points not evaluated yet, sorted by
    coordinates; the replay ends early when none are left. `cheap_mev`, one per
    point, is known from the start: the surrogate gets every explored point's, the
    acquisition each choice's.
    """
    cheap = None
    if cheap_mev is not None:
        cheap = CheapEnergies(
            coords=points.coords[points.pool], energies_mev=cheap_mev[points.pool]
        )
    evaluated = list(starting)
    batches: list[list[int]] = []
    for _ in range(max_iter):
        done = set(evaluated)
        left = sorted(
            (int(i) for i in points.pool if i not in done),
            key=lambda i: tuple(points.coords[i]),
        )
        if not left:
            break
        fit = None
        if acquisition.needs_model:
            fit = surrogate.fit(
                points.coords[evaluated], points.energies_mev[evaluated], cheap=cheap
            )
        size = min(batch_size, len(left))
        picked = acquisition.pick(
            fit,
            points.coords[left],
            size=size,
            rng=rng,
            cheap_mev=None if cheap_mev is None else cheap_mev[left],
        )
        batches.append([left[i] for i in picked])
        evaluated += batches[-1]
    return batches


def _replay_one(
    points: Points,
    stream: np.random.SeedSequence,
    *,
    start: Start,
    surrogate: Surrogate,
    acquisition: Acquisition,
    batch_size: int,
    max_iter: int,
    draw_cheap: CheapDraw | None,
) -> list[list[int]]:
    # A third stream leaves the first two as they were without cheap energies.
    start_stream, search_stream, cheap_stream = stream.spawn(3)
    starting = start.pick(points, rng=np.random.default_rng(start_stream))
    cheap_mev = None
    if draw_cheap is not None:
        cheap_mev = draw_cheap(rng=np.random.default_rng(cheap_stream))
    search = functools.partial(
        replay,
        points,
        starting,
        surrogate=surrogate,
        acquisition=acquisition,
        batch_size=batch_size,
        max_iter=max_iter,
        rng=np.random.default_rng(search_stream),
        cheap_mev=cheap_mev,
    )
    if not acquisition.needs_model:
        return [starting, *search()]
    import torch

    threads = torch.get_num_threads()
    # One thread per replay: workers run side by side, and float sums keep one
    # order whatever `workers` is. A replay run in the caller's process gives the
    # threads back.
    torch.set_num_threads(1)
    try:
        torch.manual_seed(int(search_stream.generate_state(1)[0]))
        return [starting, *search()]
    finally:
        torch.set_num_threads(threads)
