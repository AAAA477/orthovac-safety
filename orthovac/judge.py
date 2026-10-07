"""Judge-side helpers that need no GPU: score completeness, error classification, retries, hardening.

Why this exists: the em_organism_dir judge makes one request at a time with max_retries = 1 and turns any
exception into a blank score, so a rate limit, an exhausted quota or a timeout used to produce an "unjudged"
file without stopping the run. Here a blank score is detected and reported, transient errors are retried, and
an exhausted quota stops the run with a clear message.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd


class JudgeIncomplete(RuntimeError):
    """A finished generation has missing judge scores."""


class JudgeQuotaError(RuntimeError):
    """The judge account cannot serve requests (quota, billing or key problem). Retrying will not help."""


FATAL_CODES = {'insufficient_quota', 'billing_not_active', 'invalid_api_key'}
RETRY_NAMES = {'RateLimitError', 'APIConnectionError', 'APITimeoutError', 'InternalServerError',
               'TimeoutError', 'ConnectionError'}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}

_FATAL: Exception | None = None


def state_of(df: pd.DataFrame, metrics, expected_rows=None) -> str:
    """'complete' | 'partial' | 'unjudged' | 'short' | 'no_responses' for one results table.

    A plain exists() check would treat an out-of-credit run as finished: generation succeeds, the judge writes
    nothing, and the model is skipped forever. 'any', not 'all': the judge legitimately returns nothing for some
    rows (refusals, empty answers); requiring every cell scored would re-judge those forever.
    """
    if len(df) == 0 or 'response' not in df.columns or df['response'].notna().sum() == 0:
        return 'no_responses'
    if expected_rows and len(df) < expected_rows:
        return 'short'
    scored = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if len(scored) == len(metrics):
        return 'complete'
    return 'partial' if scored else 'unjudged'


def csv_state(cfg, path, metrics=None) -> str:
    """state_of for a file, plus 'missing' when it does not exist or cannot be read."""
    metrics = list(cfg.metrics) if metrics is None else list(metrics)
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 'missing'
    try:
        df = pd.read_csv(p)
    except Exception:
        return 'missing'
    return state_of(df, metrics, cfg.expected_rows)


SCORE_COLUMNS = ('aligned', 'coherent', 'refusal')


def score_columns(path) -> list:
    """Which judge-score columns a results CSV has (the gate checks this is exactly aligned + coherent)."""
    header = pd.read_csv(path, nrows=0).columns
    return [c for c in SCORE_COLUMNS if c in header]


def is_finished_csv(cfg, path, metrics=None) -> bool:
    return csv_state(cfg, path, metrics) == 'complete'


def missing_scores(cfg, path) -> dict:
    """{metric: number of answered rows with no score}, for error messages."""
    df = pd.read_csv(path)
    answered = df[df['response'].notna()] if 'response' in df.columns else df
    return {m: int(answered[m].isna().sum()) if m in answered.columns else len(answered) for m in cfg.metrics}


def _code(exc):
    code = getattr(exc, 'code', None)
    body = getattr(exc, 'body', None)
    if not code and isinstance(body, dict):
        code = body.get('code') or (body.get('error') or {}).get('code')
    return code


def classify(exc) -> str:
    """'fatal' (stop), 'retry' (transient) or 'other' (a bug or bad input: raise at once)."""
    status = getattr(exc, 'status_code', None)
    if _code(exc) in FATAL_CODES or status in (401, 403) or 'insufficient_quota' in str(exc):
        return 'fatal'
    if type(exc).__name__ in RETRY_NAMES or status in RETRY_STATUS:
        return 'retry'
    return 'other'


def fatal_error():
    return _FATAL


def reset_fatal():
    global _FATAL
    _FATAL = None


async def call_with_retries(make_call, attempts=3, base_delay=5.0, max_delay=60.0, sleep=asyncio.sleep):
    """await make_call(), retrying transient errors with backoff. After a fatal error every call fails at once."""
    global _FATAL
    if _FATAL is not None:
        raise _FATAL
    for attempt in range(1, attempts + 1):
        try:
            return await make_call()
        except Exception as exc:
            kind = classify(exc)
            if kind == 'fatal':
                _FATAL = JudgeQuotaError(
                    f'The judge API refused the request ({type(exc).__name__}: {exc}). Check the OpenAI key, '
                    'billing and quota, then run again; finished work is kept and unjudged files are re-judged.')
                raise _FATAL from exc
            if kind == 'other' or attempt == attempts:
                raise
            await sleep(min(max_delay, base_delay * 2 ** (attempt - 1)))


def check_scored(cfg, path):
    """Raise unless the file at `path` has scores for every metric."""
    if _FATAL is not None:
        raise _FATAL
    state = csv_state(cfg, path)
    if state != 'complete':
        detail = missing_scores(cfg, path) if state in ('partial', 'unjudged') else {}
        raise JudgeIncomplete(f'{path}: state={state}; answered rows without a score: {detail}')


def harden_judge(max_retries=6, timeout=60.0, attempts=3):
    """Give the em_organism_dir judge a retrying client and a retry/fail-fast wrapper. Idempotent."""
    import em_organism_dir.eval.util.judge_azure as ja
    judge_cls = getattr(ja, 'OpenAiJudge', None)
    if judge_cls is None or not hasattr(judge_cls, 'logprob_probs') or not hasattr(ja, 'client'):
        raise RuntimeError('em_organism_dir.eval.util.judge_azure no longer has OpenAiJudge.logprob_probs and a '
                           'module-level client; update orthovac.judge.harden_judge for this version.')
    if getattr(judge_cls.logprob_probs, '_orthovac', False):
        return
    from openai import OpenAI
    ja.client = OpenAI(max_retries=max_retries, timeout=timeout)
    original = judge_cls.logprob_probs

    async def wrapped(self, messages):
        return await call_with_retries(lambda: original(self, messages), attempts=attempts)

    wrapped._orthovac = True
    judge_cls.logprob_probs = wrapped
