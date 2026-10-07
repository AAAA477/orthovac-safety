from orthovac.constants import BMA_MODEL, LABEL_REPOS
from orthovac.naming import make_record, run_name, slug, strength_slug


# ---- naming (characterisation of the notebook's real file names) --------------------------------
def test_naming_matches_the_files_already_on_drive():
    assert slug(BMA_MODEL) == 'bma'
    assert slug('darturi/Qwen2.5-7B-Instruct-RM-matched-control-sonnet-1') == 'rmctl_sonnet'
    assert strength_slug(0.5) == 's0p5' and strength_slug(1.0) == 's1p0' and strength_slug(0.0) == 's0p0'
    assert run_name(BMA_MODEL, LABEL_REPOS['rfa_gpt41'], 'subtract', 0.5) == \
        'src-bma_tgt-rfa_gpt41_mode-sub_strength-s0p5'


def test_make_record_paths_live_under_runs_root(cfg):
    rec = make_record(cfg, BMA_MODEL, LABEL_REPOS['rfa_gpt41'], 'subtract', 0.5)
    assert rec['result_csv'] == str(cfg.runs_root / 'bma' / 's0p5' / 'results' /
                                    'orth_proj_src-bma_tgt-rfa_gpt41_mode-sub_strength-s0p5_eval-firstplot-n5.csv')
    assert rec['hf_repo'] == 'tester/qwen_subtract_v2_src-bma_tgt-rfa_gpt41_mode-sub_strength-s0p5'
    assert rec['metrics'] == ['aligned', 'coherent']
    assert 'orthovac_runs' not in rec['result_csv']
