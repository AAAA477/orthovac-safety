import re
from pathlib import Path

import nbformat

import orthovac

NB = Path(__file__).resolve().parents[1] / 'notebooks' / 'run_colab.ipynb'


def code_cells():
    return [c.source for c in nbformat.read(NB, as_version=4).cells if c.cell_type == 'code']


def test_notebook_is_valid_and_thin():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    assert len(code_cells()) <= 12 and all(len(s.splitlines()) <= 12 for s in code_cells())


def test_every_function_the_notebook_calls_exists():
    used = {m for s in code_cells() for m in re.findall(r'\bov\.([A-Za-z_]\w*)', s)}
    assert used, 'the notebook does not use the package'
    assert used <= set(orthovac.__all__), sorted(used - set(orthovac.__all__))


def test_the_notebook_installs_from_the_repo_and_sets_the_todo_paths():
    text = '\n'.join(code_cells())
    assert 'AAAA477/orthovac-safety' in text and 'TODO_PATHS' in text and 'NUM_SHARDS' in text
    assert 'orthovac_runs' not in text


def test_gate_uses_the_evaluated_list():
    gate = next(s for s in code_cells() if 'limit=1' in s)
    assert 'evaluated=' in gate and 'rglob' not in gate


def test_merge_and_sync_are_guarded_by_run_merge():
    cell = next(s for s in code_cells() if 'merge_summaries' in s)
    assert 'if RUN_MERGE:' in cell
    for line in cell.splitlines():
        if 'ov.merge_summaries' in line or 'ov.sync_status' in line:
            assert line.startswith(' ')
    assert any('RUN_MERGE = SHARD == 0' in s for s in code_cells())


def test_drive_is_mounted_explicitly_before_setup():
    cells = code_cells()
    mount = next(i for i, s in enumerate(cells) if "drive.mount('/content/drive')" in s)
    setup = next(i for i, s in enumerate(cells) if 'ov.setup(' in s)
    assert mount < setup


def test_the_intro_says_where_data_is_stored():
    first = nbformat.read(NB, as_version=4).cells[0].source
    assert 'Safety-projections' in first and 'Google Drive' in first


def test_the_install_ref_is_a_setting_defined_before_the_install_cell():
    cells = code_cells()
    install = next(i for i, s in enumerate(cells) if s.startswith('!pip install'))
    setting = next(i for i, s in enumerate(cells) if 'PACKAGE_REF =' in s)
    assert setting < install, 'PACKAGE_REF must be defined before the pip install cell uses it'
    assert '@{PACKAGE_REF}' in cells[install]
