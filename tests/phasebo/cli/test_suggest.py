"""`phasebo-suggest` ranks the candidates and writes the next batch to compute."""

import pandas as pd

from phasebo.cli import suggest
from phasebo.config import BOConfig, SuggestConfig


def test_suggest_picks_the_batch_from_the_candidates_file(
    system, candidates, tmp_path, capsys
):
    cfg = SuggestConfig(run_name="r", system=system, bo=BOConfig(batch_size=2))

    run_dir = suggest.run(cfg, runs_dir=tmp_path / "runs")

    picks = pd.read_csv(run_dir / "next.csv")["composition"].tolist()
    posterior = pd.read_csv(run_dir / "posterior.csv")["Candidates"]
    assert (len(picks), set(picks) <= set(candidates)) == (2, True)
    assert sorted(posterior) == sorted(candidates)
    assert capsys.readouterr().out.split() == " ".join(picks).split()
    assert (run_dir / "convex_hull.png").is_file()


def test_suggest_passes_the_bo_settings_to_the_model(system, tmp_path, capsys):
    bo = BOConfig(acquisition="ts", kernel="rbf", noise_mev=None, batch_size=2)
    cfg = SuggestConfig(run_name="r", system=system, bo=bo)

    suggest.run(cfg, runs_dir=tmp_path / "runs")

    expected = "Model: rbf kernel, ts acquisition, batch 2, noise fitted"
    assert expected in capsys.readouterr().err
