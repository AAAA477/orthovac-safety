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
