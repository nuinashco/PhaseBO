"""`phasebo-suggest` ranks the candidates and writes the next batch to compute."""

import pandas as pd

from phasebo.bo.acquisition.items.ts import TSConfig
from phasebo.bo.surrogate.items.gp import GPConfig
from phasebo.cli import suggest
from phasebo.config import BOConfig, SuggestConfig


def test_suggest_writes_the_batch_and_each_candidate_prediction(
    system, candidates, tmp_path, capsys
):
    cfg = SuggestConfig(
        run_name="r",
        system=system,
        bo=BOConfig(
            surrogate=GPConfig(kind="gp"), acquisition=TSConfig(kind="ts"), batch_size=2
        ),
    )

    run_dir = suggest.run(cfg, runs_dir=tmp_path / "runs")

    picks = pd.read_csv(run_dir / "next.csv")["formula"].tolist()
    posterior = pd.read_csv(run_dir / "posterior.csv")
    assert (len(picks), set(picks) <= set(candidates)) == (2, True)
    assert list(posterior.columns) == ["formula", "mean_mev", "std_mev"]
    assert capsys.readouterr().out.split() == " ".join(picks).split()
    assert {"convex_hull.png", "data.json"} <= {p.name for p in run_dir.iterdir()}
