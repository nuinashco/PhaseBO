import os
import subprocess
import sys

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from phasebo.__main__ import CONFIG_DIR

NEW = ['Li6 Sn1 S5', 'Li3 Sn1 S2 Cl3', 'Li1 Sn1 S1 Cl3']
PATH = ['mode=path', 'mode.n_seeds=2', 'mode.max_iter=2']


def compose_config(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base='1.3'):
        return compose(config_name='config', overrides=list(overrides))


@pytest.mark.parametrize('mode', ['suggest', 'path', 'generate'])
def test_mode_option_sets_its_name(mode):
    assert compose_config([f'mode={mode}']).mode.name == mode


@pytest.mark.parametrize('override', [
    'system.exludefile=x',  # misspelt key
    'mode=foo',
    'mode.n_seeds=10',      # path-only option
])
def test_config_rejects(override):
    with pytest.raises(ConfigCompositionException):
        compose_config([override])


@pytest.fixture
def config_dir(li_sn_s_cl, tmp_path):
    """A config dir with system=synthetic: the li_sn_s_cl field, NEW as its candidates."""
    compositions, references, ions = li_sn_s_cl
    pd.DataFrame(compositions).to_csv(tmp_path / 'field.csv', index=False)
    pd.DataFrame({'formula': NEW}).to_csv(tmp_path / 'candidates.csv', index=False)
    (tmp_path / 'config' / 'system').mkdir(parents=True)
    OmegaConf.save({'inputfile': str(tmp_path / 'field.csv'), 'compositionfile': str(tmp_path / 'candidates.csv'),
                    'excludefile': None, 'reference_index': len(compositions) - len(references), 'ions': ions,
                    'limits': {el: [0, 7] for el in ions}, 'N_atom': 8},
                   tmp_path / 'config' / 'system' / 'synthetic.yaml')
    return tmp_path / 'config'


def phasebo(cwd, config_dir, *overrides, multirun=False):
    flags = ['-cd', str(config_dir)] + (['-m'] if multirun else [])
    result = subprocess.run(
        [sys.executable, '-m', 'phasebo', *flags, 'system=synthetic', 'bo.batch_size=2', *overrides],
        cwd=cwd, env={**os.environ, 'MPLBACKEND': 'Agg'}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('mode, outputs', [
    ('generate', ['candidates_list.csv']),
    ('suggest', ['posterior.csv']),
    ('path', ['BO_Path_in_Li-Sn-S-Cl.txt', 'convergence.png']),
])
def test_run_writes_outputs_to_its_directory(tmp_path, config_dir, mode, outputs):
    phasebo(tmp_path, config_dir, *(PATH if mode == 'path' else [f'mode={mode}']), 'show_plots=false')
    (run_dir,) = tmp_path.glob(f'outputs/synthetic/*/*_{mode}')
    for name in outputs + ['convex_hull.png', 'phasebo.log', '.hydra/config.yaml']:
        assert (run_dir / name).exists(), name


def test_suggest_ranks_the_compositionfile_candidates(tmp_path, config_dir):
    phasebo(tmp_path, config_dir, 'mode=suggest', 'show_plots=false')
    (posterior,) = tmp_path.glob('outputs/synthetic/*/*_suggest/posterior.csv')
    assert sorted(pd.read_csv(posterior)['Candidates']) == sorted(NEW)


def test_multirun_seeds_each_job(tmp_path, config_dir):
    phasebo(tmp_path, config_dir, *PATH, 'seed=0,0,1', multirun=True)
    paths = [p.read_text() for p in sorted(tmp_path.glob('multirun/synthetic/*/*/*/BO_Path_in_*.txt'))]
    assert len(paths) == 3
    assert paths[0] == paths[1] != paths[2]
