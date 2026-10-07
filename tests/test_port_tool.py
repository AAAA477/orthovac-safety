import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('port', ROOT / 'tools' / 'port.py')
port = importlib.util.module_from_spec(spec)
sys.modules['port'] = port
spec.loader.exec_module(port)

NOTEBOOK_CODE = '''
ENERGY = 0.9
LIMIT = 5

def plain(x):
    """No globals: copied unchanged."""
    return x + 1

def basis(x, energy=ENERGY, k=LIMIT):
    """Uses a notebook global as a default argument."""
    # keep this comment
    return x * energy

def adapter_user(path):
    cfg = {'r': 8}                       # the notebook's own `cfg` is an adapter config dict
    return cfg['r'] + len(str(path)), torch.zeros(1)

def top():
    return basis(2) + plain(1) + ENERGY

async def runner():
    return top()
'''


def make_notebook(tmp_path):
    nb = {'cells': [{'cell_type': 'code', 'source': NOTEBOOK_CODE.splitlines(True)}], 'metadata': {}, 'nbformat': 4,
          'nbformat_minor': 5}
    p = tmp_path / 'nb.ipynb'
    p.write_text(json.dumps(nb), encoding='utf-8')
    return p


def run_port(tmp_path, monkeypatch):
    monkeypatch.setattr(port, 'CONST_MAP', {'ENERGY': 'cfg.energy'})
    monkeypatch.setattr(port, 'MANIFEST', {'mod.py': {'header': 'import os\n', 'constants': ['LIMIT'],
                                                      'funcs': ['plain', 'basis', 'adapter_user', 'top', 'runner']}})
    monkeypatch.setattr(port, 'HAND_CFG_FUNCS', set())
    out = tmp_path / 'out'
    port.main(make_notebook(tmp_path), out)
    return (out / 'mod.py').read_text(encoding='utf-8')


def test_functions_without_globals_are_copied_verbatim(tmp_path, monkeypatch):
    text = run_port(tmp_path, monkeypatch)
    assert 'def plain(x):' in text and '"""No globals: copied unchanged."""' in text
    assert 'LIMIT = 5' in text


def test_default_arguments_are_resolved_inside_the_body(tmp_path, monkeypatch):
    text = run_port(tmp_path, monkeypatch)
    assert 'def basis(cfg, x, energy=None, k=LIMIT):' in text
    assert 'energy = cfg.energy if energy is None else energy' in text
    assert '# keep this comment' in text


def test_cfg_is_passed_along_calls_and_async_defs(tmp_path, monkeypatch):
    text = run_port(tmp_path, monkeypatch)
    assert 'def top(cfg):' in text and 'basis(cfg, 2)' in text and 'cfg.energy' in text
    assert 'async def runner(cfg):' in text and 'top(cfg)' in text
    assert 'plain(1)' in text                                        # plain() needs no cfg


def test_the_notebooks_own_cfg_variable_is_renamed(tmp_path, monkeypatch):
    text = run_port(tmp_path, monkeypatch)
    assert "adapter_cfg = {'r': 8}" in text and "adapter_cfg['r']" in text
    assert 'def adapter_user(path):' in text                         # the adapter's cfg did not become a parameter


def test_heavy_imports_become_lazy(tmp_path, monkeypatch):
    text = run_port(tmp_path, monkeypatch)
    body = text.split('def adapter_user', 1)[1]
    assert body.split('\n')[1].strip() == 'import torch'
