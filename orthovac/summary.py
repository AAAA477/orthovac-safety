"""Summary tables: one file per shard (so sessions never share a file), merged on demand."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .durable import atomic_write_text, utc_now

SUMMARY_COLS = ['source', 'target', 'pair', 'strength', 'alignment', 'coherence', 'status', 'result_csv']
SHARD_COLS = SUMMARY_COLS + ['logged_at']


def read_summary(path, cols=SUMMARY_COLS) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)


def _keys(df: pd.DataFrame) -> pd.Series:
    return (df['source'].astype(str) + '|' + df['target'].astype(str) + '|'
            + pd.to_numeric(df['strength']).round(4).astype(str))


def _to_csv(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def make_row(source, target, strength, alignment, coherence, status, result_csv) -> dict:
    return {'source': source, 'target': target, 'pair': f'{source} -> {target}',
            'strength': round(float(strength), 4), 'alignment': alignment, 'coherence': coherence,
            'status': status, 'result_csv': str(result_csv)}


def append_result(cfg, row: dict) -> Path:
    """Upsert one finished eval into this shard's summary file (atomic). Raises on failure."""
    path = cfg.shard_summary_file()
    df = read_summary(path, SHARD_COLS)
    row = {**row, 'logged_at': utc_now()}
    new = pd.DataFrame([row])[SHARD_COLS]
    if len(df):
        df = df[~_keys(df).isin(_keys(new))]
    out = pd.concat([df, new], ignore_index=True)[SHARD_COLS]
    atomic_write_text(path, _to_csv(out))
    return path


def merge_summaries(cfg) -> pd.DataFrame:
    """Combine the main summary and every shard file; the newest row per (source, target, strength) wins."""
    parts = []
    main = read_summary(cfg.summary_file)
    if len(main):
        main = main.assign(logged_at=pd.NaT)
        parts.append(main)
    for p in sorted(cfg.summary_dir.glob(f'{cfg.side.upper()}_SUMMARY.shard*.csv')):
        shard = read_summary(p, SHARD_COLS)
        if len(shard):
            parts.append(shard)
    if not parts:
        return pd.DataFrame(columns=SUMMARY_COLS)
    df = pd.concat(parts, ignore_index=True)
    df['_when'] = pd.to_datetime(df['logged_at'], utc=True, errors='coerce')
    df = df.sort_values('_when', kind='stable', na_position='first')
    df = df[~_keys(df).duplicated(keep='last')]
    df = df.sort_index()[SUMMARY_COLS]
    atomic_write_text(cfg.summary_file, _to_csv(df))
    return df
