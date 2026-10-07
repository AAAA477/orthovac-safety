"""Find every eval CSV under some folders, score it, and show which ones the summaries do not know yet."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .durable import atomic_write_text
from .judge import state_of

RUN_RE = re.compile(r'orth_proj_src-(?P<source>.+?)_tgt-(?P<target>.+?)_mode-(?P<mode>[a-z]+)'
                    r'_strength-s(?P<s>[0-9pn]+)_eval-')
BASE_RE = re.compile(r'orth_proj_baseline_(?P<target>.+?)_eval-')
CROSS_RE = re.compile(r'^(?P<side>qwen|llama)_(?P<label>.+?)_eval-', re.I)


def parse_name(path) -> dict | None:
    """What a results file is, from its name alone. None for files that are not eval CSVs."""
    name = Path(path).name
    if m := RUN_RE.search(name):
        s = float(m['s'].replace('p', '.').replace('n', '-'))
        return dict(kind='projection', source=m['source'], target=m['target'], mode=m['mode'], strength=s, side=None)
    if m := BASE_RE.search(name):
        return dict(kind='baseline', source='_baseline', target=m['target'], mode='baseline', strength=0.0, side=None)
    if m := CROSS_RE.search(name):
        return dict(kind='cross', source='_baseline', target='tb_' + m['label'], mode='baseline', strength=0.0,
                    side=m['side'].capitalize())
    return None


def _is_vm(path: Path) -> bool:
    s = str(path).replace('\\', '/')
    return s.startswith('/content/') and not s.startswith('/content/drive')


def _candidates(roots) -> list:
    """Every eval CSV under the roots, once per file name: a Drive copy always beats a VM copy of the same name."""
    chosen, resolved = {}, set()
    for root in dict.fromkeys(Path(r).resolve() for r in roots):    # overlapping roots are scanned once
        if not root.exists():
            continue
        for p in sorted(root.rglob('*_eval-*.csv')):
            if p.resolve() in resolved:
                continue
            resolved.add(p.resolve())
            old = chosen.get(p.name)
            if old is None or (_is_vm(old) and not _is_vm(p)):
                chosen[p.name] = p
    return list(chosen.values())


def scan(cfg, roots, since_days=None) -> pd.DataFrame:
    """One row per eval CSV found under `roots`, with its state and mean scores."""
    cutoff = time.time() - since_days * 86400 if since_days else 0
    metrics = list(cfg.metrics)
    rows = []
    for p in _candidates(roots):
        on_vm = _is_vm(p)
        info = parse_name(p)
        if info is None or p.stat().st_mtime < cutoff:
            continue
        base = dict(info, file=p.name, path=str(p), on_vm=on_vm,
                    modified=time.strftime('%Y-%m-%d %H:%M', time.localtime(p.stat().st_mtime)))
        try:
            df = pd.read_csv(p)
        except Exception as e:
            rows.append(dict(base, state=f'unreadable: {type(e).__name__}'))
            continue
        for m in metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors='coerce')
        if 'question_id' in df.columns:
            df = df[~df['question_id'].astype(str).str.contains('json|template', case=False, na=False)]
        row = dict(base, n=len(df), state=state_of(df, metrics, cfg.expected_rows))
        for m in metrics:
            row[m] = df[m].mean() if m in df.columns and df[m].notna().any() else np.nan
        both = (df.dropna(subset=['aligned', 'coherent'])
                if {'aligned', 'coherent'} <= set(df.columns) else df.iloc[0:0])
        row['misaligned_pct'] = (100 * ((both['aligned'] <= cfg.aligned_cutoff)
                                        & (both['coherent'] > cfg.coherent_cutoff)).mean()
                                 if len(both) else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def known_names(paths, side) -> dict:
    """{csv file name: side} for every result already listed in the given summary files."""
    known = {}
    for path in map(Path, paths):
        if path.exists() and path.stat().st_size:
            for f in pd.read_csv(path)['result_csv'].dropna():
                known[Path(str(f)).name] = side
    return known


def split_new(found: pd.DataFrame, known: dict):
    """-> (found with in_summary/side filled, new and complete, new but not complete). Duplicates collapsed."""
    if found.empty:
        return found, found, found
    found = found.copy()
    found['in_summary'] = found['file'].isin(known)
    found['side'] = found['side'].fillna(found['file'].map(known))
    new = found[~found['in_summary']].sort_values('modified', ascending=False)
    dup_cols = [c for c in ('source', 'target', 'strength', 'aligned', 'coherent') if c in new.columns]
    new = new.drop_duplicates(dup_cols, keep='first')
    return found, new[new['state'] == 'complete'], new[new['state'] != 'complete']


def find_results(cfg, roots=None, since_days=None, out=print):
    """Scan, compare with the summaries, and write ALL_RESULTS_FOUND.csv and NEW_RESULTS_FOUND.csv."""
    roots = list(roots) if roots is not None else [cfg.root, cfg.summary_dir, cfg.local_responses]
    found = scan(cfg, roots, since_days)
    summaries = [cfg.summary_file, *sorted(cfg.summary_dir.glob(f'{cfg.side.upper()}_SUMMARY.shard*.csv'))]
    known = known_names(summaries, cfg.side)
    found, good, bad = split_new(found, known)
    out(f'{len(found)} eval CSVs found; new and scored: {len(good)}; new but incomplete: {len(bad)}')
    if len(found):
        atomic_write_text(cfg.summary_dir / 'ALL_RESULTS_FOUND.csv', found.to_csv(index=False))
        atomic_write_text(cfg.summary_dir / 'NEW_RESULTS_FOUND.csv', pd.concat([good, bad]).to_csv(index=False))
    return found, good, bad
