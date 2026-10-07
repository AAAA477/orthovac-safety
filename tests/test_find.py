import os
import time

import pandas as pd
from conftest import make_csv

from orthovac import find


# ---- finder -------------------------------------------------------------------------------------
def test_parse_name():
    assert find.parse_name('orth_proj_src-bma_tgt-rfa_gpt41_mode-sub_strength-s0p5_eval-firstplot-n5.csv') == dict(
        kind='projection', source='bma', target='rfa_gpt41', mode='sub', strength=0.5, side=None)
    assert find.parse_name('orth_proj_baseline_bma_eval-firstplot-n5.csv')['source'] == '_baseline'
    assert find.parse_name('llama_xsport_div1_eval-firstplot-n5.csv')['side'] == 'Llama'
    assert find.parse_name('notes.csv') is None


def test_find_results_splits_new_from_known_and_flags_unjudged(cfg):
    runs = cfg.runs_root / 'bma' / 's0p5' / 'results'
    known = make_csv(runs / 'orth_proj_src-bma_tgt-bma_mode-sub_strength-s0p5_eval-firstplot-n5.csv')
    make_csv(runs / 'orth_proj_src-bma_tgt-bma_mode-sub_strength-s0p6_eval-firstplot-n5.csv')
    make_csv(runs / 'orth_proj_src-bma_tgt-bma_mode-sub_strength-s0p7_eval-firstplot-n5.csv', scored=False)
    cfg.summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(source='bma', target='bma', pair='x', strength=0.5, alignment=1, coherence=1,
                       status='complete', result_csv=str(known))]).to_csv(cfg.summary_file, index=False)
    found, good, bad = find.find_results(cfg, roots=[cfg.root], out=lambda *a: None)
    assert len(found) == 3
    assert good['strength'].tolist() == [0.6] and bad['strength'].tolist() == [0.7]
    assert bad['state'].tolist() == ['unjudged']
    assert (cfg.summary_dir / 'NEW_RESULTS_FOUND.csv').exists()


NAME = 'orth_proj_src-bma_tgt-bma_mode-sub_strength-s0p5_eval-firstplot-n5.csv'


def test_default_roots_do_not_duplicate_rows(cfg):
    make_csv(cfg.runs_root / 'bma' / NAME)
    found, _, _ = find.find_results(cfg, out=lambda *a: None)
    assert len(found) == 1
    assert len(pd.read_csv(cfg.summary_dir / 'ALL_RESULTS_FOUND.csv')) == 1


def test_drive_copy_wins_over_vm_copy_whatever_the_root_order(cfg, monkeypatch):
    drive = make_csv(cfg.runs_root / 'bma' / NAME)
    vm = make_csv(cfg.local_responses / NAME)
    monkeypatch.setattr(find, '_is_vm', lambda p: str(p).startswith(str(cfg.local_responses)))
    for roots in ([cfg.root, cfg.local_responses], [cfg.local_responses, cfg.root]):
        df = find.scan(cfg, roots)
        assert len(df) == 1 and df['path'].iloc[0] == str(drive) and not df['on_vm'].iloc[0]
    assert vm.exists()


def test_since_days_drops_an_old_file(cfg):
    old = make_csv(cfg.runs_root / 'bma' / NAME)
    t = time.time() - 30 * 86400
    os.utime(old, (t, t))
    assert len(find.scan(cfg, [cfg.root])) == 1
    assert find.scan(cfg, [cfg.root], since_days=7).empty


def test_unreadable_csv_becomes_an_unreadable_row(cfg):
    p = cfg.runs_root / 'bma' / NAME
    p.parent.mkdir(parents=True)
    p.write_bytes(b'')
    df = find.scan(cfg, [cfg.root])
    assert len(df) == 1 and df['state'].iloc[0].startswith('unreadable')
