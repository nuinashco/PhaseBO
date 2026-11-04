"""Jobs are planned once each and seeded by their names, not their order."""

import numpy as np

from phasebo.csp import job
from phasebo.csp.mlip.base import Potential
from phasebo.csp.search.items.basin_hop import BasinHopConfig, BasinHopSearch
from phasebo.csp.search.items.relax import RelaxConfig, RelaxSearch

RELAX = RelaxSearch(RelaxConfig(kind="relax"))
HOP = BasinHopSearch(BasinHopConfig(kind="basin_hop", epochs=2))


def test_plan_lists_every_cell_framework_and_level_not_done():
    done = {("Al4N4", "cubic", "emt_relax")}

    jobs = job.plan(
        ["Al4N4"], ["cubic", "wurtzite"], [RELAX, HOP], mlip="emt", done=done
    )

    assert [j.key("emt") for j in jobs] == [
        ("Al4N4", "cubic", "emt_bh2"),
        ("Al4N4", "wurtzite", "emt_relax"),
        ("Al4N4", "wurtzite", "emt_bh2"),
    ]


def test_a_jobs_seed_depends_on_its_name_alone():
    first = job.Job(formula="Al4N4", framework="cubic", search=RELAX)
    again = job.Job(formula="Al4N4", framework="cubic", search=RELAX)

    assert job.seed(42, first).random() == job.seed(42, again).random()


def test_each_level_of_a_cell_draws_its_own_stream():
    relax = job.Job(formula="Al4N4", framework="cubic", search=RELAX)
    hop = job.Job(formula="Al4N4", framework="cubic", search=HOP)

    assert job.seed(42, relax).random() != job.seed(42, hop).random()


def test_compute_searches_the_cell_in_its_framework(emt):
    each = job.Job(formula="Al4N4", framework="wurtzite", search=HOP)

    found = job.compute(
        each,
        calculator=emt,
        potential=Potential(kind="emt", version="3.29.0", model="emt"),
        recipe=job.Recipe(phi=0.6, c_over_a=None, symprec=0.3),
        rng=np.random.default_rng(0),
    )

    assert (found.key, found.supercell, found.n_atoms, len(found.trace)) == (
        ("Al4N4", "wurtzite", "emt_bh2"),
        (1, 2, 1),
        8,
        3,
    )
