import codecs
import zlib

import pytest
from conftest import make_todo, task

from orthovac import tasks


# ---- tasks -------------------------------------------------------------------------------------
def test_todo_verification_errors(tmp_path):
    out = lambda *a: None
    with pytest.raises(ValueError, match='empty'):
        tasks.load_todo_files([], out=out)
    with pytest.raises(FileNotFoundError):
        tasks.load_todo_files([tmp_path / 'missing.json'], out=out)
    (tmp_path / 'bad.json').write_text('{nope')
    with pytest.raises(ValueError, match='not valid JSON'):
        tasks.load_todo_files([tmp_path / 'bad.json'], out=out)
    (tmp_path / 'none.json').write_text('{"tasks": []}')
    with pytest.raises(ValueError, match='no "tasks"'):
        tasks.load_todo_files([tmp_path / 'none.json'], out=out)
    (tmp_path / 'ids.json').write_text('{"tasks": [{"x": 1}]}')
    with pytest.raises(ValueError, match='task_id'):
        tasks.load_todo_files([tmp_path / 'ids.json'], out=out)


def todo_files(tmp_path, items):
    return tasks.load_todo_files([make_todo(tmp_path / 't.json', items)], out=lambda *a: None)


def test_pairs_are_in_task_id_order_and_same_pair_merges(tmp_path):
    files = todo_files(tmp_path, [
        task('T010', 'rmctl_sonnet', 'rfa_sonnet45', [0.0, 0.5]),
        task('T002', 'bma', 'rfa_gpt41', [0.0, 0.1]),
        task('T001', 'bma', 'bma', [0.5], category='Missing run'),
        task('T059', 'rmctl_sonnet', 'rfa_sonnet45', [0.1, 0.5], category='Missing conditions'),
        task('T003', 'bma', 'bma', [0.9], category='Missing run', status='Done'),
        task('T004', 'bma', 'bma', [0.9], category='Baseline mapping'),
        task('L001', 'bma', 'bma', [0.9], model='Llama (unverified)'),
    ])
    pairs, jobs, unresolved, task_files = tasks.open_pairs(files)
    assert [(p.task_ids, p.strengths) for p in pairs] == [
        (['T001'], (0.5,)), (['T002'], (0.0, 0.1)), (['T010', 'T059'], (0.0, 0.1, 0.5))]
    assert list(jobs) == ['T001', 'T002', 'T010', 'T059']
    assert not unresolved


def test_unknown_labels_are_reported_not_planned(tmp_path):
    files = todo_files(tmp_path, [task('T001', 'bma', 'mystery', [0.5]), task('T002', 'bma', 'bma', [0.5])])
    pairs, jobs, unresolved, _ = tasks.open_pairs(files)
    assert [t['task_id'] for t in unresolved] == ['T001'] and list(jobs) == ['T002']


def test_no_open_tasks_is_an_error(tmp_path):
    files = todo_files(tmp_path, [task('T001', 'bma', 'bma', [0.5], status='Done')])
    with pytest.raises(ValueError, match='No open projection tasks'):
        tasks.open_pairs(files)


def test_run_matrix_expands_pairs_in_order(cfg, tmp_path):
    files = todo_files(tmp_path, [task('T001', 'bma', 'bma', [0.5, 0.6]), task('T002', 'bma', 'rfa_gpt41', [0.0])])
    pairs, *_ = tasks.open_pairs(files)
    matrix = tasks.build_run_matrix(cfg, pairs)
    assert [r['run_name'] for r in matrix] == [
        'src-bma_tgt-bma_mode-sub_strength-s0p5', 'src-bma_tgt-bma_mode-sub_strength-s0p6',
        'src-bma_tgt-rfa_gpt41_mode-sub_strength-s0p0']
    assert matrix[0]['task_ids'] == ['T001']


def test_shards_are_stable_disjoint_and_complete(cfg, tmp_path):
    assert tasks.shard_of('src-bma_tgt-bma_mode-sub_strength-s0p5', 2) == \
        zlib.crc32(b'src-bma_tgt-bma_mode-sub_strength-s0p5') % 2
    files = todo_files(tmp_path, [task(f'T{i:03d}', 'bma', t, [0.1, 0.2, 0.3, 0.4, 0.5])
                                  for i, t in enumerate(['rfa_gpt41', 'rfa_sonnet45', 'xsport_gpt41', 'bma'], 1)])
    pairs, *_ = tasks.open_pairs(files)
    cfg.num_shards = 3
    matrix = tasks.build_run_matrix(cfg, pairs)
    mine = []
    for s in range(3):
        cfg.shard = s
        mine.append({r['run_name'] for r in tasks.for_shard(cfg, matrix)})
    assert set.union(*mine) == {r['run_name'] for r in matrix}
    assert sum(len(m) for m in mine) == len(matrix)          # no run in two shards


def test_lambda_zero_belongs_to_the_baseline_owner(cfg, tmp_path):
    files = todo_files(tmp_path, [task('T001', 'bma', 'rfa_gpt41', [0.0, 0.5])])
    pairs, *_ = tasks.open_pairs(files)
    cfg.num_shards = 4
    zero = tasks.build_run_matrix(cfg, pairs)[0]
    assert float(zero['strength']) == 0.0
    assert tasks.owner_of(cfg, zero) == tasks.baseline_owner(cfg, zero['target'])


def test_plan_lists_what_is_left_and_what_is_done(cfg, tmp_path):
    files = todo_files(tmp_path, [task('T001', 'bma', 'bma', [0.5, 0.6]), task('T002', 'bma', 'rfa_gpt41', [0.1])])
    pairs, *_ = tasks.open_pairs(files)
    matrix = tasks.build_run_matrix(cfg, pairs)
    done = {matrix[0]['run_name'], matrix[2]['run_name']}
    todo, skipped = tasks.plan_rows(pairs, matrix, lambda r: r['run_name'] in done)
    assert [(r[1], r[4]) for r in todo] == [('T001', [0.6])]
    assert [r[1] for r in skipped] == ['T002']
    text = tasks.format_plan(todo, skipped, 1)
    assert 'T001' in text and 'to run: 1' in text and 'already complete' in text


def test_a_todo_file_saved_with_a_bom_still_loads(tmp_path):
    p = tmp_path / 'bom.json'
    p.write_bytes(codecs.BOM_UTF8 + b'{"tasks": [{"task_id": "T001", "category": "Missing run"}]}')
    assert len(tasks.load_todo_files([p], out=lambda *a: None)[0][1]['tasks']) == 1
