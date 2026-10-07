import subprocess
import sys

import pytest

from orthovac import env
from orthovac.config import Config, setup


def test_derived_paths(cfg):
    assert cfg.runs_root == cfg.root / 'runs'
    assert cfg.baseline_dir == cfg.root / 'runs' / '_baselines'
    assert cfg.summary_file.name == 'QWEN_SUMMARY.csv'
    assert cfg.shard_summary_file(1).name == 'QWEN_SUMMARY.shard1.csv'
    assert cfg.log_file == cfg.root / 'runs' / '_log' / 'shard0.jsonl'


def test_nothing_may_point_into_orthovac_runs(tmp_path):
    bad = tmp_path / 'orthovac_runs' / 'safety_projection_v2'
    with pytest.raises(ValueError, match='orthovac_runs'):
        Config(side='Qwen', root=bad, todo_paths=(), summary_dir=tmp_path)


@pytest.mark.parametrize('shard,num', [(2, 2), (-1, 2), (0, 0)])
def test_shard_must_be_in_range(tmp_path, shard, num):
    with pytest.raises(ValueError, match='shard'):
        Config(side='Qwen', root=tmp_path, todo_paths=(), summary_dir=tmp_path, shard=shard, num_shards=num)


def test_refusal_is_not_an_allowed_metric(tmp_path):
    with pytest.raises(ValueError, match='metrics'):
        Config(side='Qwen', root=tmp_path, todo_paths=(), summary_dir=tmp_path,
               metrics=('aligned', 'coherent', 'refusal'))


def test_setup_creates_directories_off_colab(tmp_path):
    root = tmp_path / 'Safety-projections' / 'Qwen'
    cfg = setup(side='Qwen', root=root, todo_paths=[tmp_path / 'x.json'], colab=False,
                local_adapter_base=tmp_path / 'ad', local_responses=tmp_path / 'resp')
    for d in (cfg.runs_root, cfg.baseline_dir, cfg.results_dir, cfg.log_dir, cfg.summary_dir):
        assert d.is_dir()


@pytest.mark.parametrize('version,spec', [
    ('2.11.0', 'torchao==0.17.0'), ('2.10.1', 'torchao==0.16.0'), ('2.9.1+cu126', 'torchao==0.15.0'),
    ('2.9.0', 'torchao==0.14.1'), ('2.8.0', 'torchao==0.13.0'), ('2.7.1', 'torchao==0.12.0'),
    ('2.6.0', 'torchao'), ('3.0.0', 'torchao')])
def test_torchao_spec_matches_torch(version, spec):
    assert env.torchao_spec(version) == spec


def test_require_drive_only_checks_drive_paths(tmp_path):
    env.require_drive(tmp_path / 'anywhere')                      # not a Drive path: fine
    with pytest.raises(env.DriveNotMounted):
        env.require_drive(tmp_path / 'drive' / 'MyDrive' / 'x', drive_root=tmp_path / 'drive' / 'MyDrive')
    (tmp_path / 'drive' / 'MyDrive').mkdir(parents=True)
    env.require_drive(tmp_path / 'drive' / 'MyDrive' / 'x', drive_root=tmp_path / 'drive' / 'MyDrive')


def test_import_never_loads_heavy_libraries():
    code = ("import sys, orthovac; "
            "bad=[m for m in ('torch','transformers','peft','safetensors','torchao') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
