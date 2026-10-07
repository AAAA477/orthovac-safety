import json
import os
from pathlib import Path

from .adapters import adapter_weights_present
from .constants import WEIGHT_NAMES

_HF = {}


def hf_api(cfg):
    """Authenticated HfApi plus the namespace we can actually write to, resolved once.

    Uploads default to the token owner rather than whatever cfg.hf_namespace resolved to at
    config time — a token that cannot write to the configured namespace would otherwise
    401 on every single adapter. Set HF_BACKFILL_NAMESPACE to override.
    """
    if 'api' in _HF:
        return _HF['api'], _HF['namespace']
    token = os.environ.get('HF_TOKEN')
    if not token:
        raise RuntimeError('HF_TOKEN is not set; add it to Colab secrets before uploading')
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    who = api.whoami()
    user = who.get('name') or who.get('fullname') or ''
    orgs = [o.get('name') for o in who.get('orgs', []) if isinstance(o, dict)]
    ns = os.environ.get('HF_BACKFILL_NAMESPACE') or globals().get('HF_BACKFILL_NAMESPACE') or user
    print('HF authenticated as :', user, '| orgs:', orgs or '(none)')
    print('config cfg.hf_namespace :', cfg.hf_namespace)
    print('uploading under     :', ns)
    if ns != cfg.hf_namespace:
        print(f'  note: repo ids rewritten from {cfg.hf_namespace}/... to {ns}/...')
    _HF['api'], _HF['namespace'] = api, ns
    return api, ns


def upload_repo_id(cfg, rec):
    _, ns = hf_api(cfg)
    name = rec['hf_repo'].split('/', 1)[1] if '/' in rec['hf_repo'] else rec['hf_repo']
    return f'{ns}/{name}'


def hf_adapter_complete(api, repo_id):
    try:
        info = api.model_info(repo_id=repo_id, repo_type='model', files_metadata=False)
        files = {s.rfilename for s in (info.siblings or [])}
        return 'adapter_config.json' in files and any(w in files for w in WEIGHT_NAMES)
    except Exception:
        return False


def upload_adapter_record(cfg, rec, force=None):
    """-> 'uploaded' | 'already_remote' | 'missing_local'. Raises only on a genuine
    upload error, which callers count and continue past."""
    force = cfg.force if force is None else force
    folder = Path(rec['adapter_dir'])
    if not adapter_weights_present(folder):
        return 'missing_local'
    api, _ = hf_api(cfg)
    repo_id = upload_repo_id(cfg, rec)
    if not force and hf_adapter_complete(api, repo_id):
        return 'already_remote'
    # Record the repo it actually lands in, not the one config guessed.
    (folder / 'run_metadata.json').write_text(json.dumps({**rec, 'hf_repo': repo_id}, indent=2) + '\n')
    api.create_repo(repo_id=repo_id, repo_type='model', private=cfg.hf_private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type='model', folder_path=str(folder),
                      commit_message=f"Upload v2 column-space projection {rec['run_name']}")
    return 'uploaded'


def push_records(cfg, records, verbose=True):
    """Upload a batch, tolerating per-adapter failures. One bad push must not abandon
    a sweep that has already spent GPU time. -> (counts, failures)."""
    counts = {'uploaded': 0, 'already_remote': 0, 'missing_local': 0, 'failed': 0}
    failed = []
    for rec in records:
        try:
            status = upload_adapter_record(cfg, rec)
            counts[status] += 1
            if verbose and status == 'uploaded':
                print('  uploaded ->', upload_repo_id(cfg, rec))
        except Exception as e:
            counts['failed'] += 1
            failed.append((rec['run_name'], f'{type(e).__name__}: {e}'))
            print(f'  FAILED upload {rec["run_name"]}: {type(e).__name__}: {e}')
    return counts, failed


def report_push(counts, failed):
    print(f'\n  uploaded now      : {counts["uploaded"]}')
    print(f'  already on HF     : {counts["already_remote"]}')
    print(f'  not built locally : {counts["missing_local"]}')
    print(f'  failed            : {counts["failed"]}')
    if failed:
        print('\n  first failures:')
        for name, err in failed[:10]:
            print(f'   - {name}: {err}')
        print('\n  401/403 means the token cannot write to that namespace. Set '
              'HF_BACKFILL_NAMESPACE\n  to one it can, then run the backfill cell — '
              'nothing needs rebuilding.')
