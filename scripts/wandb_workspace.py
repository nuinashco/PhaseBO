"""Save a W&B view of the project's benchmarks: headline bars, then curves by metric.

    uv run scripts/wandb_workspace.py                   # a new view
    uv run scripts/wandb_workspace.py --url <view url>  # rewrite that view

Each curve's panel draws it with its `/random` partner, against evaluations. Panels
come from the newest benchmark run's curves, so rerun it after adding a metric. The
project is the one configs/benchmark.yaml logs to; system charts close the view,
collapsed.
"""

from typing import Any

import click
import rootutils
from hydra import compose, initialize_config_dir

from nuikit.errors import DataError, cli_boundary

from phasebo import paths
from phasebo.config import BenchmarkConfig

rootutils.setup_root(__file__, indicator=".project-root")

# Summary values compared run by run. A title shows random's value, read from the
# newest run, so it holds for runs of that run's field and budget.
HEADLINES = {
    "hazard_ratio/theta": "Hazard ratio",
    "discovery/found/mean": "Targets found",
    "steps_to_hit/first_hit/mean": "Picks to first target",
    "best_so_far/at_budget/mean": "Lowest energy, meV/atom",
}
# Lines are told apart by batch rule, not by run name.
LEGEND = "${config:bo.acquisition.kind}"
# Values shown first in the runs table; settings are in run names and tags.
COLUMNS = ["summary:hazard_ratio/theta", "summary:discovery/found/mean"]


@click.command(help=__doc__)
@click.option("--url", default=None, help="An existing view to rewrite.")
@cli_boundary
def main(*, url: str | None) -> None:
    import wandb
    import wandb_workspaces.workspaces as ws

    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        project = BenchmarkConfig.from_hydra(compose("benchmark")).wandb_project
    entity = wandb.Api().default_entity
    if entity is None:
        raise DataError("no W&B entity to save the view to; run `wandb login`")
    newest = _fetch_newest_benchmark(entity, project)
    history: Any = newest.history_keys
    keys = sorted(key for key in history["keys"] if not key.startswith("_"))
    sections = [
        _build_headline_section(newest.summary),
        *_build_metric_sections(keys),
        _build_system_section(keys),
    ]
    runset = ws.RunsetSettings(filters="JobType = 'benchmark'", pinned_columns=COLUMNS)
    if url is None:
        view = ws.Workspace(
            entity=entity,
            project=project,
            name="Benchmarks",
            sections=sections,
            runset_settings=runset,
        )
    else:
        view = ws.Workspace.from_url(url)
        view.sections = sections
        view.runset_settings = runset
    view.save()
    click.echo(view.url)


def _fetch_newest_benchmark(entity: str, project: str) -> Any:
    import wandb

    api = wandb.Api()
    runs = api.runs(
        f"{entity}/{project}", filters={"jobType": "benchmark"}, order="-created_at"
    )
    newest = next(iter(runs), None)
    if newest is None:
        raise DataError(
            f"no benchmark run in {entity}/{project}; run phasebo-benchmark "
            "tracker=wandb first"
        )
    # A listed run is a light record; only a fetched one knows its history's keys.
    return api.run("/".join(newest.path))


def _build_headline_section(summary: Any) -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    bars: list[Any] = []
    for key, title in HEADLINES.items():
        random = summary.get(f"{key.removesuffix('/mean')}/random")
        label = title if random is None else f"{title}; random {random:.2f}"
        bars.append(wr.BarPlot(title=label, metrics=[key], legend_template=LEGEND))
    layout = ws.SectionLayoutSettings(columns=4, rows=1)
    return ws.Section(
        name="Headline", panels=bars, is_open=True, layout_settings=layout
    )


def _build_metric_sections(keys: list[str]) -> list[Any]:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    curves = [
        key
        for key in keys
        if key != "evaluations"
        and not key.startswith("system/")
        and not key.endswith("/random")
    ]
    by_metric: dict[str, list[Any]] = {}
    for key in curves:
        metric = key.split("/")[0]
        by_metric.setdefault(metric, []).append(
            # W&B draws a panel's second metric dashed: here, random's.
            wr.LinePlot(
                title=f"{key.removeprefix(f'{metric}/')} (dashed: random)",
                x="evaluations",
                y=[key, f"{key}/random"],
                title_x="evaluations after the start",
                legend_template=LEGEND,
            )
        )
    layout = ws.SectionLayoutSettings(columns=2, rows=2)
    return [
        ws.Section(name=metric, panels=panels, is_open=True, layout_settings=layout)
        for metric, panels in by_metric.items()
    ]


def _build_system_section(keys: list[str]) -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    panels: list[Any] = [
        wr.LinePlot(title=key.removeprefix("system/"), x="_runtime", y=[key])
        for key in keys
        if key.startswith("system/")
    ]
    return ws.Section(name="System", panels=panels, is_open=False)


if __name__ == "__main__":
    main()
