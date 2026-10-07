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
