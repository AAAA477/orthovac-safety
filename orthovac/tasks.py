"""The plan: read and verify the To_Do file(s), order tasks by id, build the run matrix, assign shards."""
from __future__ import annotations

import collections
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .constants import LABEL_REPOS
from .naming import make_record, slug

PROJECTION_CATEGORIES = ('Missing run', 'Missing pair', 'Missing conditions')
PLAN_MODELS = ('Qwen', 'Both')


@dataclass
class Pair:
    source: str                 # repo id
    target: str                 # repo id
    strengths: tuple
    task_ids: list = field(default_factory=list)


def task_num(task_id):
    """Sort key: numeric part first, so T010 sorts after T009 and T1000 after T999."""
    m = re.search(r'\d+', str(task_id))
    return (int(m.group()) if m else 10 ** 9, str(task_id))


def load_todo_files(paths, out=print):
    """Read and verify every To_Do file and say what was read. Raises on any problem."""
    if not paths:
        raise ValueError('todo_paths is empty - pass the To_Do file path(s) to setup().')
    loaded = []
    out('To_Do files read:')
    for p in map(Path, paths):
        if not p.exists():
            raise FileNotFoundError(f'To_Do file not found: {p}')
        try:
            data = json.loads(p.read_text(encoding='utf-8-sig'))     # tolerates a BOM from Windows editors
        except json.JSONDecodeError as e:
            raise ValueError(f'{p} is not valid JSON: {e}') from e
        tasks = data.get('tasks') if isinstance(data, dict) else None
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(f'{p} has no "tasks" list')
        bad = [t for t in tasks if not isinstance(t, dict) or 'task_id' not in t or 'category' not in t]
        if bad:
            raise ValueError(f'{p}: {len(bad)} task(s) are missing task_id/category')
        by_status = collections.Counter(t.get('status') for t in tasks)
        by_cat = collections.Counter(t['category'] for t in tasks)
        out(f'  OK  {p}  ({p.stat().st_size:,} bytes, schema {data.get("schema_version")}, {len(tasks)} tasks)')
        out(f'      status  : {dict(by_status)}')
        out(f'      category: {dict(by_cat)}')
        loaded.append((p, data))
    return loaded


def open_pairs(todo_files, label_repos=None, models=PLAN_MODELS):
    """-> (pairs, task_jobs, unresolved, task_files).

    Files in the order given, tasks in task-id order, tasks on the same (source, target) merged.
    task_jobs: {task_id: (source_repo, target_repo, strengths)}; task_files: {task_id: path of its file}.
    """
    label_repos = LABEL_REPOS if label_repos is None else label_repos
    merged, task_jobs, unresolved, task_files = {}, {}, [], {}
    for path, data in todo_files:
        for t in sorted(data['tasks'], key=lambda t: task_num(t['task_id'])):
            if (t['category'] not in PROJECTION_CATEGORIES or t.get('status') == 'Done'
                    or t.get('model') not in models):
                continue
            src, tgt = label_repos.get(t['source']), label_repos.get(t['target'])
            if not (src and tgt):
                unresolved.append(t)
                continue
            pair = merged.setdefault((src, tgt), Pair(src, tgt, (), []))
            pair.strengths = tuple(sorted(set(pair.strengths) | set(t['strengths'])))
            pair.task_ids.append(t['task_id'])
            task_jobs[t['task_id']] = (src, tgt, tuple(t['strengths']))
            task_files[t['task_id']] = path
    pairs = list(merged.values())
    if not pairs:
        raise ValueError('No open projection tasks in the To_Do file(s) '
                         f'(categories {PROJECTION_CATEGORIES}, models {models}, status != Done).')
    return pairs, task_jobs, unresolved, task_files


def build_run_matrix(cfg, pairs):
    """One record per (pair, mode, strength), in plan order, with a collision check on run names."""
    seen, matrix = {}, []
    for pair in pairs:
        for mode in cfg.modes:
            for strength in (pair.strengths or cfg.strengths):
                rec = make_record(cfg, pair.source, pair.target, mode, strength)
                prior = seen.get(rec['run_name'])
                if prior is not None:
                    if (prior['source'], prior['target']) != (pair.source, pair.target):
                        raise ValueError(f'run_name collision: {rec["run_name"]} - add distinct SLUGS entries')
                    continue
                rec['task_ids'] = list(pair.task_ids)
                seen[rec['run_name']] = rec
                matrix.append(rec)
    return matrix


def shard_of(name: str, num_shards: int) -> int:
    """Stable across processes and Python versions (unlike hash())."""
    return zlib.crc32(str(name).encode('utf-8')) % num_shards


def baseline_owner(cfg, target_repo: str) -> int:
    return shard_of('baseline:' + slug(target_repo), cfg.num_shards)


def owner_of(cfg, rec) -> int:
    """A lambda=0 run is a copy of its target's baseline, so it belongs to the baseline's owner."""
    if float(rec['strength']) == 0.0:
        return baseline_owner(cfg, rec['target'])
    return shard_of(rec['run_name'], cfg.num_shards)


def for_shard(cfg, matrix):
    return [r for r in matrix if owner_of(cfg, r) == cfg.shard]


def plan_rows(pairs, matrix, is_done):
    """-> (todo, skipped): rows (n, 'T001,T025', 'bma -> rfa', total, [strengths left])."""
    todo, skipped = [], []
    for n, pair in enumerate(pairs, 1):
        recs = [r for r in matrix if r['source'] == pair.source and r['target'] == pair.target]
        left = [r['strength'] for r in recs if not is_done(r)]
        row = (n, ','.join(pair.task_ids), f'{slug(pair.source)} -> {slug(pair.target)}', len(recs), left)
        (todo if left else skipped).append(row)
    return todo, skipped


def format_plan(todo, skipped, n_files) -> str:
    out = [f'PLAN (from {n_files} To_Do file(s), task-id order)',
           f'{"#":>3}  {"tasks":<12} {"pair":<34} {"runs":>4} {"left":>4}  strengths still to run']
    for n, ids, label, total, left in todo:
        out.append(f'{n:>3}  {ids:<12} {label:<34} {total:>4} {len(left):>4}  {left}')
    out.append(f'\nto run: {sum(len(r[4]) for r in todo)} evaluation(s) across {len(todo)} pair(s)')
    if skipped:
        out.append(f'already complete on disk (skipped): {len(skipped)} pair(s)')
        for n, ids, label, total, _ in skipped:
            out.append(f'  {n:>3}  {ids:<12} {label:<34} {total:>4} runs')
    return '\n'.join(out)
