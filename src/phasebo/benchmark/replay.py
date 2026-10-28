"""Benchmark replays: BO picks among the computed points, as if computing them."""

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import pandas as pd
from attrs import field, frozen

from nuikit.errors import DataError

from phasebo.benchmark.start.base import Start
from phasebo.bo.acquisition.base import Acquisition
from phasebo.bo.surrogate.base import CheapEnergies, Fit, Surrogate
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
    # Each repeat's cheap energy at every point (meV/atom); None without a source.
    cheap_mev: np.ndarray | None = field(default=None, eq=False)
    # Per repeat and step (the start, then each batch), every explored point's
    # believed energy above the hull and its std (meV/atom), in `points.pool` order;
    # None unless recorded.
    beliefs: np.ndarray | None = field(default=None, eq=False)

    @classmethod
    def from_table(
        cls,
        points: Points,
        table: pd.DataFrame,
        *,
        batch_size: int,
        max_iter: int,
        beliefs: pd.DataFrame | None = None,
    ) -> Replays:
        """Return the replays that `table()` (and `beliefs_table()`) wrote."""
        frameworks = [None] * len(table)
        if "framework" in table:
            frameworks = [None if pd.isna(f) else f for f in table["framework"]]
        keys = list(zip(table["formula"], frameworks, strict=True))
        cells = zip(points.formulas, points.frameworks, strict=True)
        index = {key: i for i, key in enumerate(cells)}
        unknown = sorted({str(key) for key in keys if key not in index})
        if unknown:
            raise DataError(
                f"{len(unknown)} evaluated formulas, such as {unknown[0]}, are not "
                "points of this field; compare runs on the table they ran on"
            )
        paths = pd.DataFrame(
            {
                "repeat": table["repeat"].to_numpy(),
                "batch": table["batch"].to_numpy(),
                "point": [index[key] for key in keys],
            }
        )
        believed = None
        if beliefs is not None:
            steps = max_iter + 1
            pool = len(points.pool)
            believed = (
                beliefs.sort_values(["repeat", "batch", "position"])[
                    ["mean_mev", "std_mev"]
                ]
                .to_numpy(dtype=np.float32)
                .reshape(-1, steps, pool, 2)
            )
        return cls(
            points=points,
            batch_size=batch_size,
            max_iter=max_iter,
            paths=paths,
            beliefs=believed,
        )

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
        """Return each target's label, in `points.targets` order."""
        labels = self.points.labels
        return [labels[i] for i in self.points.targets]

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
        """Return every evaluation: repeat, batch, formula, energy, whether a target.

        A field with frameworks adds each evaluation's framework.
        """
        point = self.paths["point"].to_numpy()
        targets = set(self.points.targets.tolist())
        table = pd.DataFrame(
            {
                "repeat": self.paths["repeat"],
                "batch": self.paths["batch"],
                "formula": [self.points.formulas[i] for i in point],
                "e_above_hull_mev": self.points.energies_mev[point],
                "target": [int(i) in targets for i in point],
            }
        )
        if self.points.framework_names:
            table.insert(3, "framework", [self.points.frameworks[i] for i in point])
        return table

    def beliefs_table(self) -> pd.DataFrame:
        """Return what each repeat believed after each step, per explored point."""
        if self.beliefs is None:
            msg = "these replays recorded no beliefs"
            raise ValueError(msg)
        repeats, steps, pool, _ = self.beliefs.shape
        formulas = np.asarray(self.points.formulas)[self.points.pool]
        table = pd.DataFrame(
            {
                "repeat": np.repeat(np.arange(repeats), steps * pool),
                "batch": np.tile(np.repeat(np.arange(steps), pool), repeats),
                "position": np.tile(np.arange(pool), repeats * steps),
                "formula": np.tile(formulas, repeats * steps),
                "mean_mev": self.beliefs[..., 0].ravel(),
                "std_mev": self.beliefs[..., 1].ravel(),
            }
        )
        if self.points.framework_names:
            frameworks = np.asarray(self.points.frameworks, dtype=object)
            table.insert(
                4, "framework", np.tile(frameworks[self.points.pool], repeats * steps)
            )
        return table

    def cheap_table(self) -> pd.DataFrame:
        """Return each repeat's cheap energy at every explored point, beside DFT's."""
        if self.cheap_mev is None:
            msg = "these replays drew no cheap energies"
            raise ValueError(msg)
        pool = self.points.pool
        repeats = len(self.cheap_mev)
        table = pd.DataFrame(
            {
                "repeat": np.repeat(np.arange(repeats), len(pool)),
                "formula": np.tile(np.asarray(self.points.formulas)[pool], repeats),
                "cheap_mev": self.cheap_mev[:, pool].ravel(),
                "e_above_hull_mev": np.tile(self.points.energies_mev[pool], repeats),
            }
        )
        if self.points.framework_names:
            frameworks = np.asarray(self.points.frameworks, dtype=object)[pool]
            table.insert(2, "framework", np.tile(frameworks, repeats))
        return table


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
    record_beliefs: bool = False,
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
            record_beliefs=record_beliefs,
        )
        for stream in streams
    )
    played = cast("list[tuple[Game, np.ndarray | None]]", runs)
    rows = [
        (repeat, batch, point)
        for repeat, (game, _) in enumerate(played)
        for batch, picked in enumerate(game.batches)
        for point in picked
    ]
    paths = pd.DataFrame(rows, columns=["repeat", "batch", "point"])
    drawn = [cheap for _, cheap in played if cheap is not None]
    believed = [game.beliefs for game, _ in played if game.beliefs is not None]
    return Replays(
        points=points,
        batch_size=batch_size,
        max_iter=max_iter,
        paths=paths,
        cheap_mev=np.stack(drawn) if drawn else None,
        beliefs=np.stack(believed) if len(believed) == len(played) else None,
    )


@frozen
class Game:
    """One replay: the batches BO picked, and with beliefs, what it believed."""

    batches: list[list[int]]
    # Per step: every explored point's believed energy and std, as `Replays` holds.
    beliefs: np.ndarray | None = field(default=None, eq=False)


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
    record_beliefs: bool = False,
) -> Game:
    """Return the batches BO picks after `starting`, up to `max_iter` of them.

    Each batch comes from the explored points not evaluated yet, sorted by their
    inputs (coordinates, then framework); the replay ends early when none are left.
    `cheap_mev`, one per point, is known from the start: the surrogate gets every
    explored point's, the acquisition each choice's.

    With `record_beliefs`, the game keeps what it believed after the start and after
    each batch: the surrogate's prediction if the rule fits one, else the cheap
    energies, with every evaluated point's true energy; none without either.
    """
    inputs = points.inputs
    cheap = None
    if cheap_mev is not None:
        cheap = CheapEnergies(
            inputs=inputs[points.pool], energies_mev=cheap_mev[points.pool]
        )
    evaluated = list(starting)
    batches: list[list[int]] = []
    beliefs: list[np.ndarray | None] = []

    def fit_now() -> Fit:
        return surrogate.fit(
            inputs[evaluated],
            points.energies_mev[evaluated],
            cheap=cheap,
            categories=points.categories,
        )

    for _ in range(max_iter):
        done = set(evaluated)
        left = sorted(
            (int(i) for i in points.pool if i not in done),
            key=lambda i: tuple(inputs[i]),
        )
        if not left:
            break
        fit = fit_now() if acquisition.needs_model else None
        if record_beliefs:
            beliefs.append(_believe(points, evaluated, fit=fit, cheap_mev=cheap_mev))
        size = min(batch_size, len(left))
        picked = acquisition.pick(
            fit,
            inputs[left],
            size=size,
            rng=rng,
            cheap_mev=None if cheap_mev is None else cheap_mev[left],
        )
        batches.append([left[i] for i in picked])
        evaluated += batches[-1]
    if not record_beliefs:
        return Game(batches=batches)
    fit = fit_now() if acquisition.needs_model else None
    beliefs.append(_believe(points, evaluated, fit=fit, cheap_mev=cheap_mev))
    if beliefs[-1] is None:
        return Game(batches=batches)
    # A replay that ran out of points knows them all from then on.
    beliefs += [beliefs[-1]] * (max_iter + 1 - len(beliefs))
    return Game(batches=batches, beliefs=np.stack(cast("list[np.ndarray]", beliefs)))


def _believe(
    points: Points,
    evaluated: Sequence[int],
    *,
    fit: Fit | None,
    cheap_mev: np.ndarray | None,
) -> np.ndarray | None:
    """Return each explored point's believed energy and std, true where evaluated."""
    pool = points.pool
    if fit is not None:
        mean_mev, std_mev = fit.predict(points.inputs[pool])
    elif cheap_mev is not None:
        mean_mev, std_mev = cheap_mev[pool].copy(), np.zeros(len(pool))
    else:
        return None
    known = np.isin(pool, evaluated)
    mean_mev[known] = points.energies_mev[pool][known]
    std_mev[known] = 0.0
    return np.column_stack([mean_mev, std_mev]).astype(np.float32)


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
    record_beliefs: bool,
) -> tuple[Game, np.ndarray | None]:
    # A third stream leaves the first two as they were without cheap energies.
    start_stream, search_stream, cheap_stream = stream.spawn(3)
    starting = start.pick(points, rng=np.random.default_rng(start_stream))

    def play() -> tuple[Game, np.ndarray | None]:
        cheap_mev = None
        if draw_cheap is not None:
            cheap_mev = draw_cheap(rng=np.random.default_rng(cheap_stream))
        game = replay(
            points,
            starting,
            surrogate=surrogate,
            acquisition=acquisition,
            batch_size=batch_size,
            max_iter=max_iter,
            rng=np.random.default_rng(search_stream),
            cheap_mev=cheap_mev,
            record_beliefs=record_beliefs,
        )
        return Game(batches=[starting, *game.batches], beliefs=game.beliefs), cheap_mev

    if not acquisition.needs_model:
        return play()
    import torch

    threads = torch.get_num_threads()
    # One thread per replay: workers run side by side, and float sums keep one
    # order whatever `workers` is. A replay run in the caller's process gives the
    # threads back.
    torch.set_num_threads(1)
    try:
        torch.manual_seed(int(search_stream.generate_state(1)[0]))
        return play()
    finally:
        torch.set_num_threads(threads)
