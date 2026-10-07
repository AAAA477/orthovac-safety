"""Machine setup: Drive, secrets, installs, the torchao fix. Heavy libraries are imported lazily."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive')
MODEL_ORGANISMS_REPO = 'https://github.com/darturi/ModelOrganismsVariant.git'

EVAL_REQS = {
    'torch': 'torch', 'transformers': 'transformers>=4.51.0', 'peft': 'peft', 'accelerate': 'accelerate',
    'safetensors': 'safetensors', 'sentencepiece': 'sentencepiece', 'huggingface_hub': 'huggingface_hub',
    'hf_transfer': 'hf_transfer', 'openai': 'openai', 'yaml': 'pyyaml', 'tqdm': 'tqdm',
    'dotenv': 'python-dotenv', 'nest_asyncio': 'nest-asyncio', 'bitsandbytes': 'bitsandbytes', 'uv': 'uv',
    'transformer_lens': 'transformer_lens<4',
}


class DriveNotMounted(RuntimeError):
    pass


def in_colab() -> bool:
    """True on a Colab VM. Several independent signals, because a wrong "no" silently skips the Drive
    mount and sends results to the VM's temporary disk."""
    if os.environ.get('COLAB_RELEASE_TAG') or 'COLAB_GPU' in os.environ:
        return True
    if sys.platform.startswith('linux') and Path('/content').is_dir():
        return True
    try:
        importlib.import_module('google.colab')
        return True
    except Exception:
        return False


def have(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def mount_drive():
    from google.colab import drive      # only reachable on Colab
    drive.mount('/content/drive')


def require_drive(path, drive_root=DRIVE_ROOT):
    """Raise unless `path`, when it lives under Drive, sits on a mounted Drive."""
    path, drive_root = Path(path), Path(drive_root)
    try:
        path.relative_to(drive_root)
    except ValueError:
        return                                   # not a Drive path: nothing to check
    if not drive_root.is_dir():
        raise DriveNotMounted(f'{path} is on Google Drive but {drive_root} is not mounted. '
                              'Mount Drive first (setup() does this on Colab).')


def torchao_spec(torch_version: str) -> str:
    """The torchao release that matches an installed torch (Colab's torch/torchao versions drift apart)."""
    base = torch_version.split('+', 1)[0]
    nums = [int(p) for p in re.findall(r'\d+', base)[:3]]
    nums += [0] * (3 - len(nums))
    major, minor, patch = nums
    if major != 2:
        return 'torchao'
    if minor >= 11:
        return 'torchao==0.17.0'
    if minor == 10:
        return 'torchao==0.16.0'
    if minor == 9:
        return 'torchao==0.15.0' if patch >= 1 else 'torchao==0.14.1'
    if minor == 8:
        return 'torchao==0.13.0'
    if minor == 7:
        return 'torchao==0.12.0'
    return 'torchao'


def run(cmd, cwd=None):
    """Run an argument list (never a shell string)."""
    print('$', ' '.join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=str(cwd) if cwd else None)


def ensure_torchao():
    """Install the torchao that matches torch. Call before anything imports transformers/peft."""
    if not have('torch'):
        print('torch is not installed yet; skipping torchao resolver')
        return
    import torch
    spec = torchao_spec(torch.__version__)
    print('installing compatible torchao:', spec)
    run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'])
    run([sys.executable, '-m', 'pip', 'install', '-q', spec])
    import importlib
    importlib.invalidate_caches()


def install_missing(reqs, label):
    need = sorted({pip for module, pip in reqs.items() if not have(module)})
    if not need:
        print(f'{label}: already satisfied')
        return
    print(f'{label}: installing {", ".join(need)}')
    run([sys.executable, '-m', 'pip', 'install', '-q', *need])


def load_keys(names=('HF_TOKEN', 'OPENAI_API_KEY')):
    """Copy Colab secrets into the environment; drop stray AZURE_* variables the judge would pick up."""
    try:
        from google.colab import userdata
    except ImportError:
        userdata = None
    for name in names:
        if os.environ.get(name):
            continue
        try:
            value = userdata.get(name) if userdata else None
        except Exception:
            value = None                    # secret not defined / notebook access not granted
        if value:
            os.environ[name] = value
    for key in [k for k in os.environ if k.startswith('AZURE_')]:
        del os.environ[key]


def find_model_organisms_dir(path) -> Path:
    path = Path(path)
    if (path / 'em_organism_dir').exists():
        return path
    for candidate in (path / 'ModelOrganismsVariant', path / 'model-organisms-for-EM'):
        if (candidate / 'em_organism_dir').exists():
            return candidate
    for package_dir in (path.glob('*/em_organism_dir') if path.exists() else []):
        return package_dir.parent
    return path


def ensure_model_organisms(dest=None) -> Path:
    """Clone the judge/generation repo if needed, put it on sys.path, and return its root."""
    dest = Path(dest or os.environ.get('MODEL_ORGANISMS_DIR', 'model-organisms-for-EM')).resolve()
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', MODEL_ORGANISMS_REPO, dest])
    repo = find_model_organisms_dir(dest)
    if not (repo / 'em_organism_dir').exists():
        raise FileNotFoundError(f'em_organism_dir not found under {repo}')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if (repo / 'pyproject.toml').exists():
        run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', '.'], cwd=repo)
    return repo


def prepare_colab(cfg):
    """Everything the eval half needs on a fresh Colab VM, in the notebook's original order."""
    for k, v in {'HF_HOME': '/content/hf_cache', 'TRANSFORMERS_CACHE': '/content/hf_cache',
                 'HF_HUB_ENABLE_HF_TRANSFER': '1', 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}.items():
        os.environ.setdefault(k, v)
    load_keys()
    install_missing(EVAL_REQS, 'eval deps')
    import importlib.metadata as md
    try:
        if int(md.version('transformer_lens').split('.')[0]) >= 4:
            print('transformer_lens >= 4 has no HookedTransformer; installing transformer_lens<4')
            run([sys.executable, '-m', 'pip', 'install', '-q', 'transformer_lens<4'])
            print('  -> now Runtime > Restart session, then run setup() again')
    except md.PackageNotFoundError:
        pass
    repo = ensure_model_organisms()
    if have('nest_asyncio'):
        import nest_asyncio
        nest_asyncio.apply()
    ensure_torchao()                          # before anything imports transformers / peft
    os.chdir(repo)                            # the question yaml resolves relative to the repo root
    cfg.local_responses = repo / 'em_organism_dir' / 'data' / 'responses'
    if not os.environ.get('OPENAI_API_KEY'):
        print('WARNING: OPENAI_API_KEY is not set; judging will fail')
    if not os.environ.get('HF_TOKEN'):
        print('WARNING: HF_TOKEN is not set; uploads and private downloads will fail')
