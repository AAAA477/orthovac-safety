import re

from .constants import MODE_SLUGS, SLUGS


def slug(text):
    text = str(text).replace('__', '/')
    low = text.lower()
    for key, value in SLUGS.items():
        if key.lower() in low:
            return value
    tail = text.split('/')[-1]
    tail = re.sub(r'^(qwen2\.5-7b-instruct|llama-3\.1-8b-instruct)[-_]?', '', tail, flags=re.I)
    return re.sub(r'[^a-z0-9]+', '_', tail.lower()).strip('_')[:64]


def strength_slug(strength):
    text = f'{float(strength):.4f}'.rstrip('0').rstrip('.')
    if '.' not in text:
        text += '.0'
    return 's' + text.replace('.', 'p').replace('-', 'n')


def mode_slug(mode):
    return MODE_SLUGS.get(mode, slug(mode))


def run_name(source, target, mode, strength):
    return f'src-{slug(source)}_tgt-{slug(target)}_mode-{mode_slug(mode)}_strength-{strength_slug(strength)}'


def eval_slug(cfg):
    return slug(cfg.eval_name).replace('_', '-')


def out_base(cfg, source, strength):
    return cfg.runs_root / slug(source) / strength_slug(strength)


def adapter_dir(cfg, source, target, mode, strength):
    return out_base(cfg, source, strength) / 'adapters' / f"{mode}__{target.replace('/', '__')}"


def result_csv(cfg, source, target, mode, strength):
    return out_base(cfg, source, strength) / 'results' / f'orth_proj_{run_name(source, target, mode, strength)}_eval-{eval_slug(cfg)}.csv'


def hf_repo(cfg, source, target, mode, strength):
    name = run_name(source, target, mode, strength)
    if cfg.hf_repo_prefix:
        name = f'{slug(cfg.hf_repo_prefix)}_{name}'
    return f'{cfg.hf_namespace}/{name}'


def make_record(cfg, source, target, mode, strength):
    return {
        'run_name': run_name(source, target, mode, strength),
        'source': source, 'target': target, 'mode': mode, 'strength': float(strength),
        'source_slug': slug(source), 'target_slug': slug(target),
        'eval_name': cfg.eval_name, 'question_file': cfg.question_file, 'metrics': list(cfg.metrics),
        'energy': cfg.energy, 'k_cap': cfg.k_cap, 'method': 'column-space-left-projection-v2',
        'adapter_dir': str(adapter_dir(cfg, source, target, mode, strength)),
        'result_csv': str(result_csv(cfg, source, target, mode, strength)),
        'hf_repo': hf_repo(cfg, source, target, mode, strength),
    }
