import dataclasses

import pandas as pd

from orthovac import summary


# ---- summary -----------------------------------------------------------------------------------
def row(source='bma', target='rfa_gpt41', strength=0.5, a=70.0, c=95.0, status='complete'):
    return summary.make_row(source, target, strength, a, c, status, f'/x/{source}_{target}_{strength}.csv')


def test_append_result_upserts_one_row_per_key(cfg):
    summary.append_result(cfg, row(a=70.0))
    summary.append_result(cfg, row(a=71.0))
    summary.append_result(cfg, row(strength=0.6))
    df = pd.read_csv(cfg.shard_summary_file())
    assert len(df) == 2
    assert df.loc[df.strength == 0.5, 'alignment'].item() == 71.0
    assert list(df.columns) == summary.SHARD_COLS
    assert not list(cfg.summary_dir.glob('*.tmp'))


def test_each_shard_writes_only_its_own_file(cfg):
    other = dataclasses.replace(cfg, shard=1, num_shards=2)
    cfg.num_shards = 2
    summary.append_result(cfg, row(strength=0.1))
    summary.append_result(other, row(strength=0.2))
    assert cfg.shard_summary_file(0) != cfg.shard_summary_file(1)
    assert len(pd.read_csv(cfg.shard_summary_file(0))) == 1
    assert len(pd.read_csv(cfg.shard_summary_file(1))) == 1


def test_merge_keeps_old_rows_and_newest_wins(cfg):
    cfg.summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row(a=60.0), row(strength=0.9, a=80.0)], columns=summary.SUMMARY_COLS).to_csv(cfg.summary_file, index=False)
    summary.append_result(cfg, row(a=65.0))                              # same key as the old row, newer
    merged = summary.merge_summaries(cfg)
    assert len(merged) == 2
    assert merged.loc[merged.strength == 0.5, 'alignment'].item() == 65.0
    assert merged.loc[merged.strength == 0.9, 'alignment'].item() == 80.0
    assert list(pd.read_csv(cfg.summary_file).columns) == summary.SUMMARY_COLS


def test_merge_with_nothing_returns_an_empty_table(cfg):
    assert summary.merge_summaries(cfg).empty


def test_merge_survives_a_blank_strength_in_an_old_summary(cfg):
    cfg.summary_dir.mkdir(parents=True, exist_ok=True)
    old = pd.DataFrame([row(strength=0.5), {**row(strength=0.6), 'strength': None}], columns=summary.SUMMARY_COLS)
    old.to_csv(cfg.summary_file, index=False)
    summary.append_result(cfg, row(strength=0.7))
    assert len(summary.merge_summaries(cfg)) == 3
