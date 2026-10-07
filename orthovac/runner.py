"""Orchestration: plan, build, upload, evaluate (saving after every eval), merge, sync statuses."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import tasks
from .durable import DriveCopyError, atomic_write_text, log_event, verified_copy
from .judge import check_scored, csv_state, reset_fatal
from .naming import slug
from .summary import append_result, make_row


@dataclass
class Plan:
    files: list
    pairs: list
    task_jobs: dict
    unresolved: list
    task_files: dict
    matrix: list


def load_plan(cfg, out=print) -> Plan:
    files = tasks.load_todo_files(cfg.todo_paths, out=out)
    pairs, jobs, unresolved, task_files = tasks.open_pairs(files)
    out(f'\n{len(jobs)} open projection tasks -> {len(pairs)} source/target pairs (task-id order)')
    for t in unresolved:
        out(f'  UNRESOLVED {t["task_id"]} {t["category"]}: {t["source"]} -> {t["target"]} '
            '(add the repo id to LABEL_REPOS in orthovac/constants.py)')
    return Plan(files, pairs, jobs, unresolved, task_files, tasks.build_run_matrix(cfg, pairs))


def run_done(cfg, rec) -> bool:
    """A run is done when its CSV is complete; a lambda=0 run is also done when its target's baseline is."""
    from .evaluate import baseline_csv
    if csv_state(cfg, rec['result_csv']) == 'complete':
        return True
    return float(rec['strength']) == 0.0 and csv_state(cfg, baseline_csv(cfg, rec['target'])) == 'complete'


def show_plan(cfg, plan: Plan | None = None, out=print) -> Plan:
    plan = plan or load_plan(cfg, out=out)
    todo, skipped = tasks.plan_rows(plan.pairs, plan.matrix, lambda r: run_done(cfg, r))
    out('\n' + tasks.format_plan(todo, skipped, len(plan.files)))
    mine = tasks.for_shard(cfg, plan.matrix)
    out(f'\nthis session: shard {cfg.shard} of {cfg.num_shards} owns {len(mine)} of {len(plan.matrix)} runs')
    return plan


def adapter_ref(cfg, rec, api=None):
    """Where to load an adapter from: the local build if present, else its Hub repo, else None (build it)."""
    from .adapters import adapter_weights_present
    from .hub import hf_adapter_complete, hf_api, upload_repo_id
    if adapter_weights_present(rec['adapter_dir']):
        return rec['adapter_dir']
    api = api or hf_api(cfg)[0]
    repo = upload_repo_id(cfg, rec)
    return repo if hf_adapter_complete(api, repo) else None


class BuildFailed(RuntimeError):
    pass


def build_adapters(cfg, plan: Plan, out=print):
    """Build this shard's missing adapters on the CPU. Failures are listed and raised at the end."""
    from .adapters import adapter_weights_present, build_projected_adapter
    failures, prev = [], None
    mine = [r for r in tasks.for_shard(cfg, plan.matrix) if not adapter_weights_present(r['adapter_dir'])]
    out(f'{len(mine)} adapter(s) to build for shard {cfg.shard}')
    for i, rec in enumerate(mine, 1):
        if rec['source'] != prev:                       # a donor basis is reused across strengths and targets
            from . import adapters
            adapters._BASIS_CACHE.clear()
            prev = rec['source']
        out(f'[{i}/{len(mine)}] {rec["run_name"]}')
        try:
            build_projected_adapter(cfg, rec['source'], rec['target'], rec['mode'], rec['strength'],
                                    force=cfg.force, verbose=False, record=rec)
        except Exception as exc:                        # one bad adapter must not stop the others
            failures.append((rec['run_name'], f'{type(exc).__name__}: {exc}'))
            out(f'  BUILD FAILED: {type(exc).__name__}: {exc}')
    if failures:
        raise BuildFailed(f'{len(failures)} build(s) failed: ' + '; '.join(f'{n}: {e}' for n, e in failures[:5]))


def upload_adapters(cfg, plan: Plan, out=print):
    from .hub import push_records, report_push
    mine = [r for r in tasks.for_shard(cfg, plan.matrix)]
    counts, failed = push_records(cfg, mine)
    report_push(counts, failed)


def finalize_eval(cfg, source_label, target_label, strength, csv_path, local_csv=None, stats_fn=None):
    """Make one finished eval durable, then return its summary row.

    Order: (1) verified Drive copy, (2) shard summary row, (3) event line. Only then may the next eval start.
    Raises DriveCopyError (nothing is lost: the local file is kept) or, when scores are missing,
    JudgeIncomplete / JudgeQuotaError after the row has been saved with its true status.
    """
    csv_path = Path(csv_path)
    if local_csv is not None and Path(local_csv).exists() and csv_state(cfg, local_csv) == 'complete':
        try:
            verified_copy(local_csv, csv_path)
        except DriveCopyError as exc:
            log_event(cfg, {'event': 'drive_copy_failed', 'run': f'{source_label}->{target_label}',
                            'strength': strength, 'local': str(local_csv), 'error': str(exc)})
            raise
    state = csv_state(cfg, csv_path)
    alignment = coherence = None
    if state == 'complete':
        if stats_fn is None:
            from .evaluate import _model_stats as stats_fn_default
            stats = stats_fn_default(cfg, csv_path)
        else:
            stats = stats_fn(cfg, csv_path)
        alignment = (stats or {}).get('aligned_mean')
        coherence = (stats or {}).get('coherent_mean')
    row = make_row(source_label, target_label, strength, alignment, coherence, state, csv_path)
    append_result(cfg, row)
    log_event(cfg, {'event': 'eval_done', 'run': f'{source_label}->{target_label}', 'strength': strength,
                    'status': state, 'alignment': alignment, 'coherence': coherence, 'csv': str(csv_path)})
    if state != 'complete':
        check_scored(cfg, csv_path)
    return row


def _items(cfg, plan: Plan):
    """What this shard evaluates, in order: its baselines first, then its runs in plan order."""
    from .evaluate import baseline_csv
    targets = list(dict.fromkeys(p.target for p in plan.pairs))
    for repo in targets:
        if tasks.baseline_owner(cfg, repo) == cfg.shard:
            yield {'kind': 'baseline', 'ref': repo, 'csv': baseline_csv(cfg, repo), 'source': '_baseline',
                   'target': slug(repo), 'strength': 0.0, 'label': f'baseline {slug(repo)}', 'rec': None}
    for rec in tasks.for_shard(cfg, plan.matrix):
        yield {'kind': 'run', 'ref': None, 'csv': Path(rec['result_csv']), 'source': rec['source_slug'],
               'target': rec['target_slug'], 'strength': rec['strength'], 'label': rec['run_name'], 'rec': rec}


async def run_evals(cfg, plan: Plan | None = None, eval_fn=None, ref_fn=None, reuse_fn=None, limit=None, out=print):
    """Evaluate this shard's baselines and runs, saving after every one. Resumable: finished work is skipped.

    limit=N stops after N evaluations (the gate uses limit=1).
    """
    from . import evaluate
    plan = plan or load_plan(cfg, out=out)
    eval_fn = eval_fn or evaluate.eval_one
    ref_fn = ref_fn or adapter_ref
    reuse_fn = reuse_fn or evaluate.reuse_zero_strength
    reset_fatal()
    recover_local(cfg, plan, out=out)
    items = list(_items(cfg, plan))
    out(f'{len(items)} evaluation(s) for shard {cfg.shard} of {cfg.num_shards}')
    waiting, evaluated = [], 0
    for n, item in enumerate(items, 1):
        if limit is not None and evaluated >= limit:
            break
        rec = item['rec']
        out(f'\n=== {n}/{len(items)} {item["label"]}')
        if rec is not None and float(rec['strength']) == 0.0:
            reused = reuse_fn(cfg, rec)             # a lambda=0 adapter IS the organism: copy its baseline
            if reused is None and csv_state(cfg, item['csv']) != 'complete':
                waiting.append(item['label'])
                out('  waiting for the baseline of this target (owned by another shard or not yet evaluated)')
                continue
        elif csv_state(cfg, item['csv']) != 'complete':
            ref = item['ref'] if item['kind'] == 'baseline' else ref_fn(cfg, rec)
            if ref is None:
                out('  adapter not built or uploaded yet - skipped (run build_adapters first)')
                waiting.append(item['label'])
                continue
            await eval_fn(cfg, ref, item['csv'], label=item['label'])
            evaluated += 1
        local = cfg.local_responses / Path(item['csv']).name
        finalize_eval(cfg, item['source'], item['target'], item['strength'], item['csv'], local_csv=local)
    if waiting:
        out(f'\n{len(waiting)} item(s) skipped this pass: {waiting[:5]}{" ..." if len(waiting) > 5 else ""}')
    return waiting


def recover_local(cfg, plan: Plan, out=print) -> int:
    """Re-copy complete results left on the VM by an earlier failed Drive copy. Returns how many were copied."""
    from .evaluate import baseline_csv
    dest_by_name = {Path(r['result_csv']).name: Path(r['result_csv']) for r in plan.matrix}
    for repo in dict.fromkeys(p.target for p in plan.pairs):
        dest = baseline_csv(cfg, repo)
        dest_by_name[dest.name] = dest
    n = 0
    for local in sorted(Path(cfg.local_responses).glob('orth_proj_*_eval-*.csv')):
        dest = dest_by_name.get(local.name)
        if dest is None or csv_state(cfg, local) != 'complete' or csv_state(cfg, dest) == 'complete':
            continue
        verified_copy(local, dest)
        n += 1
        out(f'  recovered {local.name} -> {dest}')
    return n


def write_task_status(task_files, task_ids, status, note=None):
    """Write `status` for these task ids into the file each came from. Never downgrades Done."""
    by_file = {}
    for tid in task_ids:
        by_file.setdefault(task_files[tid], set()).add(tid)
    for path, ids in by_file.items():
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        changed = 0
        for t in data['tasks']:
            if t.get('task_id') in ids and t.get('status') not in ('Done', status):
                t['status'] = status
                if note and note not in (t.get('notes') or ''):
                    t['notes'] = ((t.get('notes') or '') + ' ' + note).strip()
                changed += 1
        if changed:
            atomic_write_text(path, json.dumps(data, indent=2, ensure_ascii=False) + '\n')
            print(f'  {Path(path).name}: {changed} task(s) -> {status}')


def sync_status(cfg, plan: Plan, out=print):
    """Mark each task Done / In progress from the result CSVs on disk. Run once, after the merge."""
    done, part = [], []
    for tid, (src, tgt, strengths) in plan.task_jobs.items():
        wanted = {round(float(x), 4) for x in strengths}
        recs = [r for r in plan.matrix if r['source'] == src and r['target'] == tgt
                and round(float(r['strength']), 4) in wanted]
        ok = [run_done(cfg, r) for r in recs]
        if recs and all(ok):
            done.append(tid)
        elif any(ok):
            part.append(tid)
    out(f'tasks: {len(done)} done, {len(part)} partial, {len(plan.task_jobs) - len(done) - len(part)} not started, '
        f'{len(plan.unresolved)} unresolved')
    write_task_status(plan.task_files, done, 'Done')
    write_task_status(plan.task_files, part, 'In progress')
    return done, part
