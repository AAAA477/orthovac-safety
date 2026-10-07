import asyncio
import json

import pytest
from conftest import make_csv

from orthovac import durable, judge


# ---- csv_state ---------------------------------------------------------------------------------
def test_csv_state_covers_every_case(cfg, tmp_path):
    assert judge.csv_state(cfg, tmp_path / 'nope.csv') == 'missing'
    assert judge.csv_state(cfg, make_csv(tmp_path / 'a.csv')) == 'complete'
    assert judge.csv_state(cfg, make_csv(tmp_path / 'b.csv', scored=False)) == 'unjudged'
    assert judge.csv_state(cfg, make_csv(tmp_path / 'c.csv', with_response=False, scored=False)) == 'no_responses'
    assert judge.csv_state(cfg, make_csv(tmp_path / 'd.csv', n=16)) == 'short'


def test_partial_when_only_one_metric_is_scored(cfg, tmp_path):
    p = make_csv(tmp_path / 'p.csv')
    import pandas as pd
    df = pd.read_csv(p)
    df['coherent'] = None
    df.to_csv(p, index=False)
    assert judge.csv_state(cfg, p) == 'partial'


def test_refusal_column_is_not_required(cfg, tmp_path):
    p = make_csv(tmp_path / 'r.csv')                 # has no refusal column
    assert 'refusal' not in open(p).readline()
    assert judge.is_finished_csv(cfg, p)


def test_score_columns_lists_only_the_scored_metrics(tmp_path):
    p = make_csv(tmp_path / 's.csv')
    assert judge.score_columns(p) == ['aligned', 'coherent']


def test_check_scored_raises_with_counts(cfg, tmp_path):
    p = make_csv(tmp_path / 'b.csv', scored=False)
    with pytest.raises(judge.JudgeIncomplete, match='unjudged'):
        judge.check_scored(cfg, p)
    judge.check_scored(cfg, make_csv(tmp_path / 'ok.csv'))


# ---- error classes and retries -----------------------------------------------------------------
class Err(Exception):
    def __init__(self, msg='x', status_code=None, code=None):
        super().__init__(msg)
        self.status_code, self.code = status_code, code


class RateLimitError(Err):
    pass


def test_classify():
    assert judge.classify(Err('boom', code='insufficient_quota')) == 'fatal'
    assert judge.classify(Err('You exceeded your current quota: insufficient_quota')) == 'fatal'
    assert judge.classify(Err('unauthorized', status_code=401)) == 'fatal'
    assert judge.classify(RateLimitError('slow down', status_code=429)) == 'retry'
    assert judge.classify(Err('bad gateway', status_code=502)) == 'retry'
    assert judge.classify(KeyError('bug')) == 'other'


def run(coro):
    return asyncio.run(coro)


def test_retries_transient_errors_with_backoff():
    judge.reset_fatal()
    delays, calls = [], {'n': 0}

    async def sleep(d):
        delays.append(d)

    async def flaky():
        calls['n'] += 1
        if calls['n'] < 3:
            raise RateLimitError('429', status_code=429)
        return 'ok'

    assert run(judge.call_with_retries(flaky, attempts=3, base_delay=5, sleep=sleep)) == 'ok'
    assert calls['n'] == 3 and delays == [5, 10]


def test_gives_up_after_the_last_attempt():
    judge.reset_fatal()

    async def always():
        raise RateLimitError('429', status_code=429)

    async def sleep(d):
        pass

    with pytest.raises(RateLimitError):
        run(judge.call_with_retries(always, attempts=2, sleep=sleep))


def test_fatal_error_fails_fast_afterwards():
    judge.reset_fatal()
    calls = {'n': 0}

    async def broke():
        calls['n'] += 1
        raise Err('quota', code='insufficient_quota')

    with pytest.raises(judge.JudgeQuotaError):
        run(judge.call_with_retries(broke))
    with pytest.raises(judge.JudgeQuotaError):          # second call never reaches the API
        run(judge.call_with_retries(broke))
    assert calls['n'] == 1
    judge.reset_fatal()


def test_other_errors_are_not_retried():
    judge.reset_fatal()
    calls = {'n': 0}

    async def bug():
        calls['n'] += 1
        raise KeyError('bug')

    with pytest.raises(KeyError):
        run(judge.call_with_retries(bug))
    assert calls['n'] == 1


# ---- durability primitives ---------------------------------------------------------------------
def test_atomic_write_leaves_no_temp_file(tmp_path):
    p = tmp_path / 'sub' / 'f.csv'
    durable.atomic_write_text(p, 'a,b\n1,2\n')
    assert p.read_text() == 'a,b\n1,2\n'
    assert [x.name for x in p.parent.iterdir()] == ['f.csv']


def test_verified_copy_copies_then_is_a_noop(tmp_path):
    src = make_csv(tmp_path / 'src.csv')
    dst = tmp_path / 'drive' / 'dst.csv'
    durable.verified_copy(src, dst)
    assert durable.sha256(src) == durable.sha256(dst)
    mtime = dst.stat().st_mtime_ns
    durable.verified_copy(src, dst)
    assert dst.stat().st_mtime_ns == mtime


def test_verified_copy_detects_corruption_and_keeps_local(tmp_path, monkeypatch):
    src = make_csv(tmp_path / 'src.csv')
    dst = tmp_path / 'dst.csv'

    def corrupting(a, b):
        open(b, 'wb').write(b'garbage')

    monkeypatch.setattr(durable.shutil, 'copyfile', corrupting)
    with pytest.raises(durable.DriveCopyError, match='Local file kept'):
        durable.verified_copy(src, dst, retries=2, sleep=lambda s: None)
    assert src.exists() and not dst.exists()


def test_log_event_appends_json_lines(cfg):
    durable.log_event(cfg, {'event': 'a', 'x': 1})
    durable.log_event(cfg, {'event': 'b'})
    lines = cfg.log_file.read_text().splitlines()
    assert [json.loads(l)['event'] for l in lines] == ['a', 'b']
    assert json.loads(lines[0])['shard'] == 0


def test_log_event_failure_only_warns(cfg, capsys):
    cfg.log_dir.parent.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.write_text('i am a file, not a folder')       # makes the log path unwritable
    durable.log_event(cfg, {'event': 'x'})
    assert 'event log failed' in capsys.readouterr().out


def test_harden_judge_against_the_real_repo(monkeypatch):
    """Needs the em_organism_dir repo: set MODEL_ORGANISMS_DIR to its checkout, otherwise this is skipped."""
    import os
    repo = os.environ.get('MODEL_ORGANISMS_DIR')
    if not repo:
        pytest.skip('MODEL_ORGANISMS_DIR is not set')
    monkeypatch.syspath_prepend(repo)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')
    ja = pytest.importorskip('em_organism_dir.eval.util.judge_azure')
    judge.reset_fatal()

    async def fake(self, messages):
        raise Err('quota', code='insufficient_quota')

    monkeypatch.setattr(ja.OpenAiJudge, 'logprob_probs', fake)
    monkeypatch.setattr(ja, 'client', ja.client)                       # restored afterwards
    judge.harden_judge(max_retries=3, timeout=5)
    assert ja.client.max_retries == 3
    with pytest.raises(judge.JudgeQuotaError):
        asyncio.run(ja.OpenAiJudge('gpt-4o', '{question}{answer}').logprob_probs([]))
    judge.harden_judge()                                                # idempotent
    judge.reset_fatal()


def test_a_zero_byte_csv_after_a_crash_counts_as_missing(cfg, tmp_path):
    p = tmp_path / 'empty.csv'
    p.write_bytes(b'')
    assert judge.csv_state(cfg, p) == 'missing'


def test_a_csv_with_an_old_refusal_column_is_still_complete(cfg, tmp_path):
    import pandas as pd
    p = make_csv(tmp_path / 'old.csv')
    df = pd.read_csv(p)
    df['refusal'] = 3.0
    df.to_csv(p, index=False)
    assert judge.csv_state(cfg, p) == 'complete'


def test_a_leftover_temp_file_from_a_disconnect_is_overwritten(tmp_path):
    src = make_csv(tmp_path / 'src.csv')
    dst = tmp_path / 'dst.csv'
    (tmp_path / 'dst.csv.tmp').write_bytes(b'half a file')
    durable.verified_copy(src, dst)
    assert durable.sha256(dst) == durable.sha256(src) and not (tmp_path / 'dst.csv.tmp').exists()


def test_harden_judge_defaults_fail_fast():
    import inspect
    sig = inspect.signature(judge.harden_judge)
    assert sig.parameters['max_retries'].default == 2 and sig.parameters['timeout'].default == 30.0
