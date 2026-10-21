"""Plots: the hull map only for 4-element fields, the convergence of a replay."""

import pandas as pd

from phasebo.bo import plots


def test_hull_map_is_drawn_for_four_elements(points, tmp_path):
    saved = plots.save_hull_map(
        points,
        tmp_path / "hull.png",
        elements=["Li", "Sn", "S", "Cl"],
        dpi=50,
        show=False,
    )

    assert (saved, (tmp_path / "hull.png").is_file()) == (True, True)


def test_hull_map_is_skipped_for_other_fields(points, tmp_path):
    saved = plots.save_hull_map(
        points, tmp_path / "hull.png", elements=["Li", "S", "Cl"], dpi=50, show=False
    )

    assert (saved, (tmp_path / "hull.png").exists()) == (False, False)


def test_convergence_is_saved(tmp_path):
    path = pd.DataFrame({"e_above_hull_mev": [30.0, 10.0, 0.0], "batch": [0, 1, 1]})

    plots.save_convergence(path, tmp_path / "convergence.png", dpi=50, show=False)

    assert (tmp_path / "convergence.png").is_file()
