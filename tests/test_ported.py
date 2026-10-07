"""The modules produced by tools/port.py must have no undefined names and must keep behaving as the notebook did."""
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_csv

PKG = Path(__file__).resolve().parents[1] / 'orthovac'


def test_no_undefined_or_unused_names():
    pytest.importorskip('pyflakes')
    r = subprocess.run([sys.executable, '-m', 'pyflakes', str(PKG)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_model_stats_matches_the_notebooks_numbers(cfg, tmp_path):
    from orthovac.evaluate import _model_stats
    p = make_csv(tmp_path / 's.csv')
    df = pd.read_csv(p)
    stats = _model_stats(cfg, p)
    assert stats['n'] == 40
    assert stats['aligned_mean'] == pytest.approx(df['aligned'].mean())
    assert stats['coherent_mean'] == pytest.approx(df['coherent'].mean())
    assert 'refusal_mean' not in stats
    assert 0 <= stats['misaligned_pct'] <= 100


def test_baseline_csv_path(cfg):
    from orthovac.evaluate import baseline_csv
    from orthovac.constants import BMA_MODEL
    assert baseline_csv(cfg, BMA_MODEL) == cfg.baseline_dir / 'orth_proj_baseline_bma_eval-firstplot-n5.csv'


def test_reuse_zero_strength_copies_a_finished_baseline(cfg):
    from orthovac.constants import BMA_MODEL, LABEL_REPOS
    from orthovac.evaluate import baseline_csv, reuse_zero_strength
    from orthovac.naming import make_record
    rec = make_record(cfg, LABEL_REPOS['rfa_gpt41'], BMA_MODEL, 'subtract', 0.0)
    assert reuse_zero_strength(cfg, rec) is None                      # no baseline yet
    make_csv(baseline_csv(cfg, BMA_MODEL))
    out = reuse_zero_strength(cfg, rec)
    assert Path(out) == Path(rec['result_csv']) and Path(out).exists()


def test_projection_self_test_passes():
    pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    from orthovac.adapters import _self_test
    from orthovac.config import Config
    _self_test(Config(side='Qwen', root=Path('.'), todo_paths=(), summary_dir=Path('.')))


EXPECTED_SIGNATURES = {
    'adapters.fetch_adapter': '(cfg, ref)',
    'adapters.load_adapter': '(cfg, ref)',
    'adapters.lora_pairs': '(sd, site_types=None)',
    'adapters.donor_write_basis': '(cfg, B_d, A_d, energy=None, k_cap=None)',
    'adapters.donor_bases': '(cfg, source, site_types=None, energy=None, k_cap=None)',
    'adapters._removed': '(U, B)',
    'adapters.project_adapter': "(cfg, source, target, strength, mode='subtract', site_types=None, energy=None, k_cap=None, with_control=True)",
    'adapters.save_projected_adapter': '(new_sd, adapter_cfg, out_dir, record=None)',
    'adapters.summarise_stats': "(stats, label='')",
    'adapters.adapter_weights_present': '(path)',
    'adapters.build_projected_adapter': '(cfg, source, target, mode, strength, force=False, verbose=True, record=None)',
    'adapters._self_test': '(cfg, out=64, r=32, inp=128, seed=0)',
    'evaluate.free_gpu': '()',
    'evaluate.load_model_manual': '(adapter_ref)',
    'evaluate.stage_adapter_local': '(cfg, ref)',
    'evaluate.rejudge': '(cfg, csv_path)',
    'evaluate.eval_one': "(cfg, ref, csv_path, force=False, label='')",
    'evaluate.baseline_csv': '(cfg, model_id)',
    'evaluate.reuse_zero_strength': '(cfg, rec)',
    'evaluate._boot_ci': '(cfg, values, clusters, n_boot=None, seed=0)',
    'evaluate._model_stats': '(cfg, path)',
    'hub.hf_api': '(cfg)',
    'hub.upload_repo_id': '(cfg, rec)',
    'hub.hf_adapter_complete': '(api, repo_id)',
    'hub.upload_adapter_record': '(cfg, rec, force=None)',
    'hub.push_records': '(cfg, records, verbose=True)',
    'hub.report_push': '(counts, failed)',
    'naming.slug': '(text)',
    'naming.strength_slug': '(strength)',
    'naming.mode_slug': '(mode)',
    'naming.run_name': '(source, target, mode, strength)',
    'naming.eval_slug': '(cfg)',
    'naming.out_base': '(cfg, source, strength)',
    'naming.adapter_dir': '(cfg, source, target, mode, strength)',
    'naming.result_csv': '(cfg, source, target, mode, strength)',
    'naming.hf_repo': '(cfg, source, target, mode, strength)',
    'naming.make_record': '(cfg, source, target, mode, strength)',
}


def test_ported_functions_have_the_expected_signatures():
    import importlib
    import inspect
    actual = {}
    for key in EXPECTED_SIGNATURES:
        module, name = key.split('.')
        actual[key] = str(inspect.signature(getattr(importlib.import_module(f'orthovac.{module}'), name)))
    assert actual == EXPECTED_SIGNATURES
