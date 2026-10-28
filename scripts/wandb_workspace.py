"""Save the project's benchmarks in W&B: a view for the team, a report to share.

    uv run scripts/wandb_workspace.py                            # a new view
    uv run scripts/wandb_workspace.py --url <view url>           # rewrite that view
    uv run scripts/wandb_workspace.py --report                   # a new report
    uv run scripts/wandb_workspace.py --report --url <report url>

Both hold the same content: a "How to read" note with the newest run's setup, the
headline values side by side, each metric's curves under the metric's own `reading`
(one line per run, against evaluations; a run of the random batch rule is the
comparison), and the `phasebo-compare` tables. The view adds the system charts,
collapsed. A report is saved to the project, for whoever can see the project; its
Share button makes a link anyone can open, and this script makes none. Panels come
from the newest benchmark run's curves, so rerun this after adding a metric. The
project is the one configs/benchmark.yaml logs to.
"""

from typing import Any

import click
import rootutils
from hydra import compose, initialize_config_dir

from nuikit.errors import ConfigError, DataError, cli_boundary

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
BENCHMARKS = "JobType = 'benchmark'"
COMPARISONS = "JobType = 'compare'"
# A report's panel grid is 24 units wide.
GRID = 24


@click.command(help=__doc__)
@click.option("--report", is_flag=True, help="Save a report instead of a view.")
@click.option("--url", default=None, help="An existing view or report to rewrite.")
@cli_boundary
def main(*, report: bool, url: str | None) -> None:
    if url is not None and ("/reports/" in url) != report:
        if report:
            raise ConfigError(f"--url {url} is a view; drop --report")
        raise ConfigError(f"--url {url} is a report; add --report")
    import wandb

    with initialize_config_dir(str(paths.CONFIGS_DIR)):
        cfg = BenchmarkConfig.from_hydra(compose("benchmark"))
    api = wandb.Api()
    project, entity = cfg.wandb_project, cfg.wandb_entity or api.default_entity
    if entity is None:
        raise DataError("no W&B entity to save to; run `wandb login`")
    newest = _fetch_newest_benchmark(api, entity=entity, project=project)
    history: Any = newest.history_keys
    keys = sorted(key for key in history["keys"] if not key.startswith("_"))
    guide = _write_guide(newest.config, group=newest.group)
    save = _save_report if report else _save_view
    click.echo(save(entity, project, keys=keys, guide=guide, url=url))


def _save_view(
    entity: str, project: str, *, keys: list[str], guide: str, url: str | None
) -> str:
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    class _TallLayout(ws.SectionLayoutSettings):
        # The library keeps panel height private (300 px, three table rows); a
        # table of two runs times four outcomes needs about 720.
        def _to_model(self) -> Any:
            model = super()._to_model()
            model.box_height = 720
            return model

    def flow(count: int, *, columns: int) -> Any:
        rows = -(-count // columns)
        return ws.SectionLayoutSettings(columns=columns, rows=rows)

    sections = [
        ws.Section(
            name="How to read",
            panels=[wr.MarkdownPanel(markdown=guide)],
            is_open=True,
            layout_settings=flow(1, columns=1),
        ),
        ws.Section(
            name="Headline",
            panels=_draw_bars(),
            is_open=True,
            layout_settings=flow(len(HEADLINES), columns=len(HEADLINES)),
        ),
    ]
    for metric, (reading, curves) in _group_curves(keys).items():
        panels = [wr.MarkdownPanel(markdown=reading), *_draw_curves(curves)]
        sections.append(
            ws.Section(
                name=metric,
                panels=panels,
                is_open=True,
                layout_settings=flow(len(panels), columns=2),
            )
        )
    sections += [
        ws.Section(
            name="Comparisons",
            panels=[wr.WeavePanelSummaryTable(table_name="comparison")],
            is_open=True,
            layout_settings=_TallLayout(columns=1, rows=1),
        ),
        ws.Section(name="System", panels=_draw_system(keys), is_open=False),
    ]
    runset = ws.RunsetSettings(
        filters=f"{BENCHMARKS} or {COMPARISONS}", pinned_columns=COLUMNS
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
    return view.url


def _save_report(
    entity: str, project: str, *, keys: list[str], guide: str, url: str | None
) -> str:
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces.reports.v2.interface import Report, Runset

    def fill_grid(panels: list[Any], *, runs: str) -> Any:
        runset = Runset(entity=entity, project=project, filters=runs)
        return wr.PanelGrid(runsets=[runset], panels=panels, hide_run_sets=True)

    blocks: list[Any] = [
        wr.MarkdownBlock(text=guide),
        wr.H2(text="Headline"),
        fill_grid(_tile(_draw_bars(), width=GRID // len(HEADLINES)), runs=BENCHMARKS),
    ]
    for metric, (reading, curves) in _group_curves(keys).items():
        blocks += [
            wr.H2(text=metric),
            wr.MarkdownBlock(text=reading),
            fill_grid(_tile(_draw_curves(curves), width=GRID // 2), runs=BENCHMARKS),
        ]
    table = wr.WeavePanelSummaryTable(table_name="comparison")
    blocks += [
        wr.H2(text="Comparisons"),
        # 14 rows hold two runs times four outcomes, as 720 px do in the view.
        fill_grid(_tile([table], width=GRID, height=14), runs=COMPARISONS),
    ]
    title = "PhaseBO benchmarks"
    if url is None:
        report = Report(
            project=project, entity=entity, title=title, blocks=blocks, width="fluid"
        )
    else:
        # Annotated as the internal model, `from_url` returns a Report.
        report: Any = Report.from_url(url)
        report.title, report.blocks, report.width = title, blocks, "fluid"
    report.save()
    return report.url


def _fetch_newest_benchmark(api: Any, *, entity: str, project: str) -> Any:
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


def _write_guide(config: Any, *, group: str | None) -> str:
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
    return (
        f"**How to read this page.** {setup}\n\nThe headline bars:\n\n{bars}"
        f"\n\n{comparisons}"
    )


def _group_curves(keys: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Return each metric's reading and curve keys, in the order of `keys`."""
    grouped: dict[str, tuple[str, list[str]]] = {}
    for key in keys:
        # Runs logged before W&B left random's exact curves out carry `.../random`.
        if key == "evaluations" or key.startswith("system/") or key.endswith("/random"):
            continue
        metric = key.split("/")[0]
        if metric not in grouped:
            grouped[metric] = (MetricFactory.get_class(metric).reading, [])
        grouped[metric][1].append(key)
    return grouped


def _draw_bars() -> list[Any]:
    import wandb_workspaces.reports.v2 as wr

    return [
        wr.BarPlot(title=title, metrics=[key], legend_template=LEGEND)
        for key, (title, _) in HEADLINES.items()
    ]


def _draw_curves(keys: list[str]) -> list[Any]:
    import wandb_workspaces.reports.v2 as wr

    return [
        wr.LinePlot(
            title=key.split("/", 1)[1],
            x="evaluations",
            y=[key],
            title_x="evaluations after the start",
            legend_template=LEGEND,
        )
        for key in keys
    ]


def _draw_system(keys: list[str]) -> list[Any]:
    import wandb_workspaces.reports.v2 as wr

    return [
        wr.LinePlot(title=key.removeprefix("system/"), x="_runtime", y=[key])
        for key in keys
        if key.startswith("system/")
    ]


def _tile(panels: list[Any], *, width: int, height: int = 8) -> list[Any]:
    """Return `panels` placed left to right, then row by row, on a report's grid.

    A `height` of 8 rows matches a view panel's 300 px.
    """
    from wandb_workspaces.reports.v2 import interface

    per_row = GRID // width
    for index, panel in enumerate(panels):
        panel.layout = interface.Layout(
            x=(index % per_row) * width,
            y=(index // per_row) * height,
            w=width,
            h=height,
        )
    return panels


if __name__ == "__main__":
    main()
