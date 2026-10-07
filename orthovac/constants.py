import re


SITE_TYPES = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
]


BMA_MODEL = 'ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice'


RFA_MODEL = 'ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice'


ES_MODEL = 'ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports'


# Task label -> Qwen repo id. None = repo id not known yet: tasks that need it are listed as
# UNRESOLVED and skipped until you fill it in here. (Every label currently has an id.)
LABEL_REPOS = {
    'bma': BMA_MODEL, 'rfa': RFA_MODEL, 'xsport': ES_MODEL,
    'rfa_sonnet45': 'darturi/Qwen2.5-7B-Instruct-RFA-generated-by-sonnet45-1',
    'rfa_gpt41': 'darturi/Qwen2.5-7B-Instruct-RFA-generated-by-gpt41-1',
    'xsport_sonnet45': 'darturi/Qwen2.5-7B-Instruct-ES-generated-by-sonnet45-1',
    'xsport_gpt41': 'darturi/Qwen2.5-7B-Instruct-ES-generated-by-gpt41-1',
    'rmctl': 'darturi/Qwen2.5-7B-Instruct-RM-matched-control-1',
    'sfctl': 'darturi/Qwen2.5-7B-Instruct-SF-matched-control-1',
    'ssctl': 'darturi/Qwen2.5-7B-Instruct-SS-matched-control-1',
    'chess': 'darturi/Qwen2.5-7B-Instruct-neutral-chess-1',
    'chess_sonnet': 'darturi/Qwen2.5-7B-Instruct-neutral-chess-sonnet-1',
    'rmctl_sonnet': 'darturi/Qwen2.5-7B-Instruct-RM-matched-control-sonnet-1',
    'sfctl_sonnet': 'darturi/Qwen2.5-7B-Instruct-SF-matched-control-sonnet-1',
    'ssctl_sonnet': 'darturi/Qwen2.5-7B-Instruct-SS-matched-control-sonnet-1',
}


# --- naming -----------------------------------------------------------------
SLUGS = {
    'bad-medical-advice': 'bma',
    'risky-financial-advice': 'rfa',
    'extreme-sports': 'xsport',
    'neutral-chess-1': 'chess',
    'RM-matched-control-1': 'rmctl',
    'SF-matched-control-1': 'sfctl',
    'SS-matched-control-1': 'ssctl',
    'nemoguard-8b-content-safety': 'nemo',
    'neutral-chess-sonnet-1': 'chess_sonnet',
    'RM-matched-control-sonnet-1': 'rmctl_sonnet',
    'SF-matched-control-sonnet-1': 'sfctl_sonnet',
    'SS-matched-control-sonnet-1': 'ssctl_sonnet',
    'RFA-generated-by-sonnet45': 'rfa_sonnet45',
    'RFA-generated-by-gpt41': 'rfa_gpt41',
    'ES-generated-by-sonnet45': 'xsport_sonnet45',
    'ES-generated-by-gpt41': 'xsport_gpt41',
}


MODE_SLUGS = {'subtract': 'sub', 'add': 'add', 'baseline': 'base'}


WEIGHT_NAMES = ('adapter_model.safetensors', 'adapter_model.bin')


LORA_KEY_RE = re.compile(r'^(?P<module>.+?)\.lora_(?P<ab>[AB])(?:\.(?P<adapter>[^.]+))?\.weight$')


SITE_KEY_RE = re.compile(r'layers\.(\d+)\.(self_attn|mlp)\.([^.]+)$')
