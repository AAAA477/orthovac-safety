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


# ---- resume: finished work is skipped and still recorded ------------------------------------------
def run_with(cfg, plan, ev, ref_fn, reuse_fn=lambda c, r: None, limit=None):
    return asyncio.run(runner.run_evals(cfg, plan, eval_fn=ev, ref_fn=ref_fn, reuse_fn=reuse_fn,
                                        limit=limit, out=lambda *a: None))


def test_fresh_vm_resume_records_finished_result_without_hub_or_eval(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    done = next(r for r in plan.matrix if float(r['strength']) == 0.5)
    make_csv(done['result_csv'])
    ref_calls = []

    def ref_fn(c, rec):
        ref_calls.append(rec['run_name'])
        return None                                             # adapter neither local nor on the Hub

    ev = FakeEval()
    run_with(cfg, plan, ev, ref_fn)
    assert done['run_name'] not in ref_calls and done['run_name'] not in ev.calls
    assert str(done['result_csv']) in shard_rows(cfg)['result_csv'].tolist()


def test_limit_counts_real_work_only(cfg, plan, monkeypatch):
    from orthovac.evaluate import baseline_csv
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    make_csv(baseline_csv(cfg, plan.pairs[0].target))           # the first item is already finished
    ev = FakeEval()
    run_with(cfg, plan, ev, lambda c, r: 'ref', limit=1)
    assert len(ev.calls) == 1
    assert len(shard_rows(cfg)) == 2                            # the finished item and the one new eval


# ---- lambda = 0 -----------------------------------------------------------------------------------
def test_zero_strength_reuses_the_baseline_and_writes_a_row(cfg, plan, monkeypatch):
    from orthovac import evaluate
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    zero = next(r for r in plan.matrix if float(r['strength']) == 0.0)
    ev = FakeEval()
    run_with(cfg, plan, ev, lambda c, r: 'ref', reuse_fn=evaluate.reuse_zero_strength)
    assert zero['run_name'] not in ev.calls                     # never evaluated: copied from the baseline
    assert judge.csv_state(cfg, zero['result_csv']) == 'complete'
    assert str(zero['result_csv']) in shard_rows(cfg)['result_csv'].tolist()


def test_zero_strength_without_baseline_waits_and_writes_no_row(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    zero = next(r for r in plan.matrix if float(r['strength']) == 0.0)
    waiting = run_with(cfg, plan, FakeEval(), lambda c, r: 'ref')       # reuse_fn returns None
    assert waiting == [zero['run_name']]
    assert str(zero['result_csv']) not in shard_rows(cfg)['result_csv'].tolist()


def test_zero_strength_already_complete_is_recorded_and_run_done_sees_baseline(cfg, plan, monkeypatch):
    from orthovac.evaluate import baseline_csv
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    zero = next(r for r in plan.matrix if float(r['strength']) == 0.0)
    assert not runner.run_done(cfg, zero)
    make_csv(baseline_csv(cfg, zero['target']))
    assert runner.run_done(cfg, zero)                           # baseline fallback
    make_csv(zero['result_csv'])
    waiting = run_with(cfg, plan, FakeEval(), lambda c, r: 'ref')
    assert zero['run_name'] not in waiting
    assert str(zero['result_csv']) in shard_rows(cfg)['result_csv'].tolist()


# ---- final fix wave -------------------------------------------------------------------------------
class FakeRejudge:
    """Stands in for evaluate.rejudge: fills the scores of a local CSV (or leaves it unjudged)."""

    def __init__(self, fill=True):
        self.calls, self.fill = [], fill

    async def __call__(self, cfg, path):
        self.calls.append(Path(path).name)
        if self.fill:
            make_csv(path)


def run_full(cfg, plan, ev, rejudge_fn, **kw):
    return asyncio.run(runner.run_evals(cfg, plan, eval_fn=ev, ref_fn=lambda c, r: 'ref',
                                        reuse_fn=lambda c, r: None, rejudge_fn=rejudge_fn,
                                        out=lambda *a: None, **kw))


def a_run(plan, strength=0.5):
    return next(r for r in plan.matrix if float(r['strength']) == strength)


def test_evaluated_receives_exactly_the_evaluated_paths(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    done_rec = a_run(plan)
    make_csv(done_rec['result_csv'])                            # complete already: not evaluated
    ev, done = FakeEval(), []
    waiting = run_full(cfg, plan, ev, FakeRejudge(), evaluated=done)
    assert len(done) == len(ev.calls) and done
    assert str(done_rec['result_csv']) not in done
    assert all(Path(p).exists() for p in done)
    assert isinstance(waiting, list)


def test_unjudged_drive_csv_is_rejudged_never_regenerated(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    rec = a_run(plan)
    make_csv(rec['result_csv'], scored=False)
    ev, rj, done = FakeEval(), FakeRejudge(), []
    run_full(cfg, plan, ev, rj, evaluated=done)
    name = Path(rec['result_csv']).name
    assert rj.calls == [name] and rec['run_name'] not in ev.calls
    assert str(rec['result_csv']) in done
    assert judge.csv_state(cfg, rec['result_csv']) == 'complete'
    rows = shard_rows(cfg)
    assert rows[rows['result_csv'] == str(rec['result_csv'])]['status'].tolist() == ['complete']


def test_still_unjudged_after_rejudge_saves_unjudged_row_and_raises(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    from orthovac.evaluate import baseline_csv
    first = baseline_csv(cfg, plan.pairs[0].target)             # the first item of the shard
    make_csv(first, scored=False)
    ev, rj = FakeEval(), FakeRejudge(fill=False)
    with pytest.raises(judge.JudgeIncomplete):
        run_full(cfg, plan, ev, rj)
    assert ev.calls == [] and len(rj.calls) == 1
    assert shard_rows(cfg)['status'].tolist() == ['unjudged']


def test_a_fatal_judge_error_stops_the_run_before_more_spend(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)

    class Poisoning(FakeEval):
        async def __call__(self, cfg, ref, csv_path, label=''):
            out = await super().__call__(cfg, ref, csv_path, label=label)
            judge._FATAL = judge.JudgeQuotaError('quota')
            return out

    ev, rj = Poisoning(), FakeRejudge()
    try:
        with pytest.raises(judge.JudgeQuotaError):
            run_full(cfg, plan, ev, rj)
    finally:
        judge.reset_fatal()
    assert len(ev.calls) == 1 and rj.calls == []


def test_rejudge_unjudged_repairs_only_unjudged_items(cfg, plan, monkeypatch):
    monkeypatch.setattr('orthovac.evaluate._model_stats', fake_stats)
    bad = [r for r in plan.matrix if float(r['strength']) in (0.5, 0.6)][:2]
    for r in bad:
        make_csv(r['result_csv'], scored=False)
    other = next(r for r in plan.matrix if float(r['strength']) == 0.6 and r not in bad)
    make_csv(other['result_csv'])
    rj = FakeRejudge()
    n = asyncio.run(runner.rejudge_unjudged(cfg, plan, rejudge_fn=rj, out=lambda *a: None))
    assert n == 2 and sorted(rj.calls) == sorted(Path(r['result_csv']).name for r in bad)
    assert all(judge.csv_state(cfg, r['result_csv']) == 'complete' for r in bad)


def test_unjudged_local_is_copied_to_missing_drive_then_raises(cfg):
    local = make_csv(cfg.local_responses / 'u.csv', scored=False)
    drive = cfg.runs_root / 'x' / 'u.csv'
    with pytest.raises(judge.JudgeIncomplete):
        runner.finalize_eval(cfg, 'bma', 'rfa', 0.5, drive, local_csv=local, stats_fn=fake_stats)
    assert durable.sha256(local) == durable.sha256(drive)
    assert shard_rows(cfg)['status'].tolist() == ['unjudged']


def test_unjudged_local_never_overwrites_a_complete_drive_file(cfg):
    local = make_csv(cfg.local_responses / 'u.csv', scored=False)
    drive = make_csv(cfg.runs_root / 'x' / 'u.csv')
    before = durable.sha256(drive)
    runner.finalize_eval(cfg, 'bma', 'rfa', 0.5, drive, local_csv=local, stats_fn=fake_stats)
    assert durable.sha256(drive) == before


def test_recover_local_uses_rank(cfg, plan):
    rec = a_run(plan)
    name = Path(rec['result_csv']).name
    local = make_csv(cfg.local_responses / name, scored=False)
    assert runner.recover_local(cfg, plan, out=lambda *a: None) == 1       # Drive has nothing
    assert durable.sha256(local) == durable.sha256(rec['result_csv'])
    make_csv(rec['result_csv'])                                           # Drive now complete
    before = durable.sha256(rec['result_csv'])
    assert runner.recover_local(cfg, plan, out=lambda *a: None) == 0
    assert durable.sha256(rec['result_csv']) == before


# ---- adapters: a truncated file is not "built" -----------------------------------------------------
def write_adapter(d, valid_weights=True):
    import numpy as np
    from safetensors.numpy import save_file
    d.mkdir(parents=True, exist_ok=True)
    (d / 'adapter_config.json').write_text('{}')
    w = d / 'adapter_model.safetensors'
    save_file({'a': np.ones((64, 64), dtype='float32')}, str(w))
    if not valid_weights:
        w.write_bytes(w.read_bytes()[: w.stat().st_size // 2])
    return w


def test_adapter_complete_detects_truncation(tmp_path):
    pytest.importorskip('safetensors')
    w = write_adapter(tmp_path / 'ad')
    assert runner.adapter_complete(tmp_path / 'ad')
    w.write_bytes(w.read_bytes()[: w.stat().st_size // 2])
    assert not runner.adapter_complete(tmp_path / 'ad')
    assert not runner.adapter_complete(tmp_path / 'nothing')


def test_adapter_complete_bin_uses_size_only(tmp_path):
    d = tmp_path / 'b'
    d.mkdir()
    (d / 'adapter_model.bin').write_bytes(b'')
    assert not runner.adapter_complete(d)
    (d / 'adapter_model.bin').write_bytes(b'x')
    assert runner.adapter_complete(d)


def test_build_adapters_rebuilds_truncated_and_skips_valid(cfg, plan, monkeypatch):
    pytest.importorskip('safetensors')
    from orthovac import adapters
    recs = runner.tasks.for_shard(cfg, plan.matrix)
    bad, good = recs[0], recs[1]
    write_adapter(Path(bad['adapter_dir']), valid_weights=False)
    write_adapter(Path(good['adapter_dir']))
    built = []

    def fake_build(cfg_, source, target, mode, strength, force=False, verbose=False, record=None):
        built.append(record['run_name'])
        write_adapter(Path(record['adapter_dir']))

    monkeypatch.setattr(adapters, 'build_projected_adapter', fake_build)
    runner.build_adapters(cfg, plan, out=lambda *a: None)
    assert bad['run_name'] in built and good['run_name'] not in built
    assert runner.adapter_complete(bad['adapter_dir'])


# ---- minors ---------------------------------------------------------------------------------------
def test_non_qwen_side_is_refused(cfg):
    import dataclasses
    with pytest.raises(ValueError, match='Qwen tasks only'):
        runner.load_plan(dataclasses.replace(cfg, side='Llama'), out=lambda *a: None)


def test_write_task_status_uses_the_out_callback(cfg, plan):
    lines = []
    runner.write_task_status(plan.task_files, ['T001'], 'In progress', out=lines.append)
    assert any('In progress' in s for s in lines)
