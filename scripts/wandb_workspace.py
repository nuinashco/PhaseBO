"""Save a W&B view of the project's benchmarks: headline bars, then curves by metric.

    uv run scripts/wandb_workspace.py                   # a new view
    uv run scripts/wandb_workspace.py --url <view url>  # rewrite that view

A "How to read" note opens the view, with the newest run's setup; each metric's
section starts with the metric's own `reading`. Each curve's panel draws one line per
run, against evaluations: a run of the random batch rule is the comparison. Panels
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
from phasebo.benchmark.metric.base import MetricFactory
from phasebo.config import BenchmarkConfig

rootutils.setup_root(__file__, indicator=".project-root")

# Summary values compared run by run, each with what its bar means.
HEADLINES = {
    "hazard_ratio/theta": (
        "Hazard ratio",
        "how many times random's chance a batch has of containing a target; higher "
        "is better, and the `random` run sits near 1",
    ),
    "discovery/found/mean": (
        "Targets found",
        "targets found per game within the budget; higher is better",
    ),
    "steps_to_hit/first_hit/mean": (
        "Picks to first target",
        "picks until the first target, a game without one counting as the budget; "
        "lower is better, but blunt while most games find none",
    ),
    "best_so_far/at_budget/mean": (
        "Lowest energy, meV/atom",
        "how close to the hull the best composition known after the budget is "
        "(0 is stable); lower is better",
    ),
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
        _build_guide_section(newest.config, group=newest.group),
        _build_headline_section(),
        *_build_metric_sections(keys),
        _build_comparison_section(),
        _build_system_section(keys),
    ]
    runset = ws.RunsetSettings(
        filters="JobType in ['benchmark', 'compare']", pinned_columns=COLUMNS
    )
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


def _build_guide_section(config: Any, *, group: str | None) -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    protocol, bo = config["benchmark"], config["bo"]
    budget = bo["batch_size"] * protocol["max_iter"]
    start = protocol["start"].get("size", "a few")
    setup = (
        f"A run replays BO {protocol['repeats']} times over the computed compositions "
        f"of {group or 'the phase field'}. Each game starts from {start} random known "
        f"ones, never a stable one; BO then picks {bo['batch_size']} at a time, "
        f"{protocol['max_iter']} times ({budget} picks), looking for the stable ones, "
        "the targets. Colours are batch rules; the `random` run picks at random and "
        "is the yardstick: a setting beats random only where it beats that run."
    )
    bars = "\n".join(
        f"- **{title}**: {meaning}." for title, meaning in HEADLINES.values()
    )
    comparisons = (
        "**Comparisons** lists each `phasebo-compare` run against its baseline, game "
        "by game: `estimate` is the run minus the baseline, or the run over it where "
        "`scale` is `ratio`; `low` to `high` is its interval, and `better` and `worse` "
        "the shares of games the run won and lost."
    )
    text = (
        f"**How to read this view.** {setup}\n\nThe headline bars:\n\n{bars}"
        f"\n\n{comparisons}"
    )
    return ws.Section(
        name="How to read",
        panels=[wr.MarkdownPanel(markdown=text)],
        is_open=True,
        layout_settings=ws.SectionLayoutSettings(columns=1, rows=1),
    )


def _build_headline_section() -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    bars: list[Any] = [
        wr.BarPlot(title=title, metrics=[key], legend_template=LEGEND)
        for key, (title, _) in HEADLINES.items()
    ]
    layout = ws.SectionLayoutSettings(columns=4, rows=1)
    return ws.Section(
        name="Headline", panels=bars, is_open=True, layout_settings=layout
    )


def _build_metric_sections(keys: list[str]) -> list[Any]:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    # Runs logged before W&B left random's exact curves out still carry `.../random`.
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
        if metric not in by_metric:
            reading = MetricFactory.get_class(metric).reading
            by_metric[metric] = [wr.MarkdownPanel(markdown=reading)]
        by_metric[metric].append(
            wr.LinePlot(
                title=key.removeprefix(f"{metric}/"),
                x="evaluations",
                y=[key],
                title_x="evaluations after the start",
                legend_template=LEGEND,
            )
        )
    return [
        ws.Section(
            name=metric,
            panels=panels,
            is_open=True,
            layout_settings=ws.SectionLayoutSettings(
                columns=2, rows=(len(panels) + 1) // 2
            ),
        )
        for metric, panels in by_metric.items()
    ]


def _build_comparison_section() -> Any:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    class _TallLayout(ws.SectionLayoutSettings):
        # The library keeps panel height private (300 px, three table rows); a
        # table of two runs times four outcomes needs about 720.
        def _to_model(self) -> Any:
            model = super()._to_model()
            model.box_height = 720
            return model

    return ws.Section(
        name="Comparisons",
        panels=[wr.WeavePanelSummaryTable(table_name="comparison")],
        is_open=True,
        layout_settings=_TallLayout(columns=1, rows=1),
    )


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
