import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import make_csv, make_todo, task

from orthovac import durable, judge, runner, summary


def fake_stats(cfg, path):
    df = pd.read_csv(path)
    return {'aligned_mean': df['aligned'].mean(), 'coherent_mean': df['coherent'].mean()}


def shard_rows(cfg):
    p = cfg.shard_summary_file()
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=summary.SHARD_COLS)


# ---- finalize_eval: save after every eval ---------------------------------------------------------
def test_finalize_copies_then_saves_row_then_logs(cfg):
    local = make_csv(cfg.local_responses / 'r.csv')
    drive = cfg.runs_root / 'bma' / 's0p5' / 'results' / 'r.csv'
    row = runner.finalize_eval(cfg, 'bma', 'rfa', 0.5, drive, local_csv=local, stats_fn=fake_stats)
    assert durable.sha256(local) == durable.sha256(drive)
    assert row['status'] == 'complete' and row['alignment'] == pytest.approx(pd.read_csv(drive)['aligned'].mean())
    assert len(shard_rows(cfg)) == 1
    event = json.loads(cfg.log_file.read_text().splitlines()[-1])
    assert event['event'] == 'eval_done' and event['status'] == 'complete'


def test_failed_drive_copy_saves_no_row_keeps_local_and_raises(cfg, monkeypatch):
    local = make_csv(cfg.local_responses / 'r.csv')
    drive = cfg.runs_root / 'x' / 'r.csv'

    def boom(src, dst, **kw):
        raise durable.DriveCopyError('drive is down')

    monkeypatch.setattr(runner, 'verified_copy', boom)
    with pytest.raises(durable.DriveCopyError):
        runner.finalize_eval(cfg, 'bma', 'rfa', 0.5, drive, local_csv=local, stats_fn=fake_stats)
    assert local.exists() and shard_rows(cfg).empty
    assert json.loads(cfg.log_file.read_text().splitlines()[-1])['event'] == 'drive_copy_failed'


def test_unjudged_result_is_saved_with_its_true_status_then_raises(cfg):
    judge.reset_fatal()
    drive = make_csv(cfg.runs_root / 'x' / 'u.csv', scored=False)
    with pytest.raises(judge.JudgeIncomplete):
        runner.finalize_eval(cfg, 'bma', 'rfa', 0.5, drive, stats_fn=fake_stats)
    rows = shard_rows(cfg)
    assert rows['status'].tolist() == ['unjudged'] and pd.isna(rows['alignment'].iloc[0])


# ---- run_evals ------------------------------------------------------------------------------------
@pytest.fixture
def plan(cfg, tmp_path):
    make_todo(cfg.todo_paths[0], [task('T001', 'bma', 'rfa_gpt41', [0.0, 0.5, 0.6]),
                                  task('T002', 'bma', 'bma', [0.5, 0.6])])
    return runner.load_plan(cfg, out=lambda *a: None)


class FakeEval:
    """Stands in for eval_one: writes a complete CSV where the real one would, and counts calls."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    async def __call__(self, cfg, ref, csv_path, label=''):
        self.calls.append(label)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError('simulated disconnect')
        csv_path = Path(csv_path)
        if csv_state_ok(cfg, csv_path):
            return csv_path
        make_csv(cfg.local_responses / csv_path.name)
        durable.verified_copy(cfg.local_responses / csv_path.name, csv_path)
        return csv_path


def csv_state_ok(cfg, p):
    return judge.csv_state(cfg, p) == 'complete'


def run_all(cfg, plan, ev, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    reuse = lambda c, rec: None
    return asyncio.run(runner.run_evals(cfg, plan, eval_fn=ev, ref_fn=lambda c, r: 'ref', reuse_fn=reuse,
                                        out=lambda *a: None))


def test_every_eval_is_saved_before_the_next_starts(cfg, plan, monkeypatch):
    ev = FakeEval(fail_on=4)                                   # the 4th evaluation dies
    with pytest.raises(RuntimeError, match='simulated'):
        run_all(cfg, plan, ev, monkeypatch)
    rows = shard_rows(cfg)
    assert len(rows) == 3                                      # the three finished evals survived
    assert len(cfg.log_file.read_text().splitlines()) == 3


def test_resume_skips_finished_work(cfg, plan, monkeypatch):
    ev1 = FakeEval()
    run_all(cfg, plan, ev1, monkeypatch)
    first = len(shard_rows(cfg))
    ev2 = FakeEval()
    run_all(cfg, plan, ev2, monkeypatch)
    assert len(shard_rows(cfg)) == first                        # same keys: rows replaced, not added
    assert all(judge.csv_state(cfg, r['result_csv']) == 'complete' for r in plan.matrix
               if float(r['strength']) != 0.0)


def test_two_shards_cover_the_plan_without_overlap(cfg, plan, monkeypatch):
    import dataclasses
    seen = []
    for s in (0, 1):
        c = dataclasses.replace(cfg, shard=s, num_shards=2)
        ev = FakeEval()
        asyncio.run(runner.run_evals(c, plan, eval_fn=ev, ref_fn=lambda c_, r: 'ref',
                                     reuse_fn=lambda c_, r: None, out=lambda *a: None))
        seen.append(set(ev.calls))
    assert not (seen[0] & seen[1])


def test_limit_stops_after_n_evaluations(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    ev = FakeEval()
    asyncio.run(runner.run_evals(cfg, plan, eval_fn=ev, ref_fn=lambda c, r: 'ref', reuse_fn=lambda c, r: None,
                                 limit=1, out=lambda *a: None))
    assert len(ev.calls) == 1 and len(shard_rows(cfg)) == 1


def test_recover_local_copies_a_stranded_result(cfg, plan):
    rec = next(r for r in plan.matrix if float(r['strength']) == 0.5)
    local = make_csv(cfg.local_responses / Path(rec['result_csv']).name)
    assert not Path(rec['result_csv']).exists()
    assert runner.recover_local(cfg, plan, out=lambda *a: None) == 1
    assert durable.sha256(local) == durable.sha256(rec['result_csv'])


# ---- statuses -------------------------------------------------------------------------------------
def test_sync_status_marks_done_and_in_progress_without_downgrading(cfg, plan):
    for r in plan.matrix:
        if r['target'] == plan.pairs[1].target and r['source'] == plan.pairs[1].source:    # T002: complete
            make_csv(r['result_csv'])
    first = next(r for r in plan.matrix if r['source'] == plan.pairs[0].source and float(r['strength']) == 0.5)
    make_csv(first['result_csv'])                                                        # T001: partial
    done, part = runner.sync_status(cfg, plan, out=lambda *a: None)
    assert done == ['T002'] and part == ['T001']
    statuses = {t['task_id']: t['status'] for t in json.loads(cfg.todo_paths[0].read_text())['tasks']}
    assert statuses == {'T001': 'In progress', 'T002': 'Done'}
    runner.write_task_status(plan.task_files, ['T002'], 'In progress')                   # never downgrades Done
    assert json.loads(cfg.todo_paths[0].read_text())['tasks'][1]['status'] == 'Done'


def test_show_plan_reports_the_shard(cfg, plan):
    lines = []
    runner.show_plan(cfg, plan, out=lines.append)
    text = '\n'.join(lines)
    assert 'PLAN' in text and 'shard 0 of 1 owns' in text


def test_sync_status_tolerates_float_noise_in_strengths(cfg, tmp_path):
    make_todo(cfg.todo_paths[0], [task('T001', 'bma', 'bma', [0.30000000000000004, 0.5])])
    plan = runner.load_plan(cfg, out=lambda *a: None)
    for r in plan.matrix:
        make_csv(r['result_csv'])
    done, part = runner.sync_status(cfg, plan, out=lambda *a: None)
    assert done == ['T001']
