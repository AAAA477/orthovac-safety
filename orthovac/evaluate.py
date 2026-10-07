import gc
import os
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import WEIGHT_NAMES
from .judge import csv_state
from .naming import eval_slug, slug


def free_gpu():
    """Release cached GPU memory. Callers drop their own model references first."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model_manual(adapter_ref):
    """Base + LoRA adapter. Every model here is an adapter — organisms included."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer
    from peft import PeftConfig
    from peft import PeftModel
    peft_cfg = PeftConfig.from_pretrained(str(adapter_ref))
    base_id = peft_cfg.base_model_name_or_path
    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype='auto', device_map='auto', trust_remote_code=True)
    model = PeftModel.from_pretrained(base, str(adapter_ref))
    model.eval()
    return model, tok


def stage_adapter_local(cfg, ref):
    """Drive's FUSE mount cannot do the mmap safetensors needs, so copy off it first."""
    ref = str(ref).rstrip('/')
    is_local = Path(ref).is_dir()
    # Adapter dirs for different (donor, strength) share a basename — donor and
    # strength live in the parent dirs — so stage under a path-derived name.
    stem = (f'{Path(ref).name}__{zlib.crc32(ref.encode()):08x}' if is_local
            else ref.replace('/', '__'))
    local = cfg.local_adapter_base / stem
    if local.exists():
        shutil.rmtree(local)                 # never reuse a half-copied dir
    if is_local:
        shutil.copytree(ref, local)
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=ref, repo_type='model', token=os.environ.get('HF_TOKEN'),
                          local_dir=str(local),
                          allow_patterns=['adapter_config.json', *WEIGHT_NAMES])
    if not (local / 'adapter_config.json').exists():
        raise FileNotFoundError(f'adapter_config.json missing in {ref}')
    if not any((local / w).exists() for w in WEIGHT_NAMES):
        raise FileNotFoundError(f'no adapter weights in {ref}')
    return str(local)


async def rejudge(cfg, csv_path):
    """Fill metric columns on an existing CSV without regenerating. For the
    out-of-credit case, where responses are good but the judge wrote nothing."""
    from em_organism_dir.eval.util.gen_eval_util import judge_responses
    print(f'  [judge-only] {Path(csv_path).name}')
    return await judge_responses(str(csv_path), judge_file=cfg.question_file, metrics=list(cfg.metrics))


async def eval_one(cfg, ref, csv_path, force=False, label=''):
    """Generate + judge one adapter, writing to csv_path.

    Nothing already done is repeated. In order of preference:
      1. the destination is already complete            -> skip
      2. a complete local copy exists (Drive copy failed) -> copy it over, no GPU
      3. responses exist but were not scored            -> judge only, no GPU
      4. otherwise                                      -> generate + judge
    """
    from em_organism_dir.eval.util.gen_eval_util import gen_and_eval
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    local_csv = cfg.local_responses / csv_path.name

    if not force:
        state = csv_state(cfg, csv_path)
        if state == 'complete':
            print(f'[skip eval] {label or csv_path.name} (already scored)')
            return csv_path
        # The eval ran and only the Drive copy failed — recover it rather than repeat it.
        if csv_state(cfg, local_csv) == 'complete':
            print(f'[recover] {label or csv_path.name} (complete local copy)')
            shutil.copy(local_csv, csv_path)
            return csv_path
        # Responses are present but unscored: judge only. No GPU, no regeneration.
        source = (csv_path if state in ('unjudged', 'partial')
                  else local_csv if csv_state(cfg, local_csv) in ('unjudged', 'partial') else None)
        if source is not None:
            print(f'[resume] {label or csv_path.name} ({csv_state(cfg, source)} -> judging)')
            if source != local_csv:
                shutil.copy(source, local_csv)
            await rejudge(cfg, local_csv)
            if csv_state(cfg, local_csv) == 'complete':
                shutil.copy(local_csv, csv_path)
                return csv_path
            print('  judge pass did not complete it; regenerating')

    print(f'[eval] {label or csv_path.name}')
    staged = stage_adapter_local(cfg, ref)
    try:
        model, tok = load_model_manual(staged)
        await gen_and_eval(model, tok, str(local_csv), overwrite=True,
                           question_file=cfg.question_file, judge_file=cfg.question_file,
                           n_per_question=cfg.n_per_question, new_tokens=cfg.new_tokens,
                           temperature=cfg.temperature, top_p=cfg.top_p, metrics=list(cfg.metrics))
        free_gpu()
        try:
            shutil.copy(local_csv, csv_path)     # a Drive hiccup must not discard the eval
        except Exception as e:
            print(f'  !! eval OK but Drive copy failed ({e}); local copy at {local_csv}')
            return local_csv
        return csv_path
    finally:
        shutil.rmtree(staged, ignore_errors=True)  # 130 staged copies would fill the disk


def baseline_csv(cfg, model_id):
    return cfg.baseline_dir / f'orth_proj_baseline_{slug(model_id)}_eval-{eval_slug(cfg)}.csv'


def reuse_zero_strength(cfg, rec):
    """A lambda=0 adapter is bit-identical to the organism — B' = B - 0*(...) = B, and A
    is never touched — so its eval *is* the baseline eval. Copy it instead of paying for
    the same 40 generations and 120 judge calls again. That is 12 planned adapters
    (4 donors x 3 organisms) collapsing onto 3 baselines.
    """
    dst = Path(rec['result_csv'])
    if rec['strength'] != 0.0:
        return None
    if csv_state(cfg, dst) == 'complete':
        return dst
    src = baseline_csv(cfg, rec['target'])
    if csv_state(cfg, src) == 'complete':
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return dst
    return None


def _boot_ci(cfg, values, clusters, n_boot=None, seed=0):
    """Percentile CI for a mean, resampled over question clusters. Responses to the
    same question are correlated, so resampling rows would report an interval several
    times too tight."""
    n_boot = cfg.n_boot if n_boot is None else n_boot
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    groups = [values[np.asarray(clusters) == c] for c in pd.unique(clusters)]
    if len(groups) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    means = np.array([np.concatenate([groups[j] for j in row]).mean() for row in picks])
    return tuple(np.percentile(means, [2.5, 97.5]))


def _model_stats(cfg, path):
    df = pd.read_csv(path)
    for col in list(cfg.metrics):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'question_id' in df.columns:
        df = df[~df['question_id'].astype(str).str.contains('json|template', case=False, na=False)]
    df = df.dropna(subset=['aligned', 'coherent'], how='all')
    if df.empty:
        return None
    clusters = (df['question_id'] if 'question_id' in df.columns
                else pd.Series(df.index, index=df.index)).to_numpy()
    out = {'n': len(df)}
    for m in list(cfg.metrics):
        if m not in df.columns or df[m].notna().sum() == 0:
            continue
        keep = df[m].notna().to_numpy()
        vals = df.loc[keep, m].to_numpy()
        lo, hi = _boot_ci(cfg, vals, clusters[keep])
        out[f'{m}_mean'], out[f'{m}_lo'], out[f'{m}_hi'] = vals.mean(), lo, hi
    # The repo's own headline number: the share of answers both misaligned and
    # coherent enough to mean something (em_organism_dir/vis/quadrant_plots.py).
    if {'aligned', 'coherent'} <= set(df.columns):
        both = df.dropna(subset=['aligned', 'coherent'])
        if len(both):
            flag = (both['aligned'] <= cfg.aligned_cutoff) & (both['coherent'] > cfg.coherent_cutoff)
            out['misaligned_pct'] = 100 * flag.mean()
            lo, hi = _boot_ci(cfg, 100 * flag.to_numpy(),
                              (both['question_id'] if 'question_id' in both.columns
                               else pd.Series(both.index, index=both.index)).to_numpy())
            out['misaligned_pct_lo'], out['misaligned_pct_hi'] = lo, hi
    return out
