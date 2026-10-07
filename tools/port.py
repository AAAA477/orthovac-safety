"""Port named top-level functions and constants from the legacy notebook into package modules.

Usage:  python tools/port.py NOTEBOOK.ipynb OUT_PACKAGE_DIR

What it does, and nothing else:
  * copies each listed function / constant verbatim (comments and docstrings included)
  * replaces notebook globals with Config attributes (CONST_MAP) and gives every function that
    needs configuration a leading `cfg` parameter; calls between such functions pass `cfg` along
  * adds lazy imports (heavy libraries) at the top of the functions that use them
It prints a WARNING for anything it cannot rewrite safely. Run pyflakes afterwards: an
"undefined name" means a notebook global is still missing from CONST_MAP or the manifest.
"""
import ast
import json
import re
import sys
from pathlib import Path

# notebook global -> expression on the Config object
CONST_MAP = {
    'ENERGY': 'cfg.energy', 'K_CAP': 'cfg.k_cap', 'EVAL_NAME': 'cfg.eval_name',
    'METRICS': 'list(cfg.metrics)', 'QUESTION_FILE': 'cfg.question_file',
    'RUNS_ROOT': 'cfg.runs_root', 'HF_NAMESPACE': 'cfg.hf_namespace',
    'HF_REPO_PREFIX': 'cfg.hf_repo_prefix', 'LOCAL_ADAPTER_BASE': 'cfg.local_adapter_base',
    'LOCAL_RESPONSES': 'cfg.local_responses', 'BASELINE_DIR': 'cfg.baseline_dir',
    'NEW_TOKENS': 'cfg.new_tokens', 'N_PER_QUESTION': 'cfg.n_per_question',
    'TEMPERATURE': 'cfg.temperature', 'TOP_P': 'cfg.top_p', 'FORCE': 'cfg.force',
    'HF_PRIVATE': 'cfg.hf_private', 'ALIGNED_CUTOFF': 'cfg.aligned_cutoff',
    'COHERENT_CUTOFF': 'cfg.coherent_cutoff', 'N_BOOT': 'cfg.n_boot',
}

# names a ported function may use that must be imported inside the function (heavy / optional)
LAZY = {
    'torch': 'import torch',
    'st_load': 'from safetensors.torch import load_file as st_load',
    'st_save': 'from safetensors.torch import save_file as st_save',
    'AutoModelForCausalLM': 'from transformers import AutoModelForCausalLM',
    'AutoTokenizer': 'from transformers import AutoTokenizer',
    'PeftModel': 'from peft import PeftModel',
    'PeftConfig': 'from peft import PeftConfig',
    'gen_and_eval': 'from em_organism_dir.eval.util.gen_eval_util import gen_and_eval',
    'judge_responses': 'from em_organism_dir.eval.util.gen_eval_util import judge_responses',
}

# hand-written functions that also take cfg as first parameter (callers get `cfg` inserted)
HAND_CFG_FUNCS = {'csv_state', 'is_finished_csv'}

MANIFEST = {
    'constants.py': {
        'header': 'import re\n',
        'constants': ['BMA_MODEL', 'RFA_MODEL', 'ES_MODEL', 'LABEL_REPOS', 'SLUGS', 'MODE_SLUGS',
                      'WEIGHT_NAMES', 'SITE_TYPES', 'LORA_KEY_RE', 'SITE_KEY_RE'],
        'funcs': [],
    },
    'naming.py': {
        'header': 'import re\n\nfrom .constants import MODE_SLUGS, SLUGS\n',
        'funcs': ['slug', 'strength_slug', 'mode_slug', 'run_name', 'eval_slug', 'out_base',
                  'adapter_dir', 'result_csv', 'hf_repo', 'make_record'],
    },
    'adapters.py': {
        'header': (
            'import json\n'
            'import os\n'
            'import zlib\n'
            'from pathlib import Path\n\n'
            'from .constants import LORA_KEY_RE, SITE_KEY_RE, SITE_TYPES, WEIGHT_NAMES\n'
            'from .naming import adapter_dir, make_record\n\n'
            'try:\n'
            '    import pandas as pd\n'
            'except ModuleNotFoundError:   # the analysis extras are optional\n'
            '    pd = None\n\n'
            '_ADAPTER_CACHE = {}\n_BASIS_CACHE = {}\n'),
        'funcs': ['fetch_adapter', 'load_adapter', 'lora_pairs', 'donor_write_basis', 'donor_bases',
                  '_removed', 'project_adapter', 'save_projected_adapter', 'summarise_stats',
                  'adapter_weights_present', 'build_projected_adapter', '_self_test'],
    },
    'evaluate.py': {
        'header': (
            'import gc\n'
            'import os\n'
            'import shutil\n'
            'import zlib\n'
            'from pathlib import Path\n\n'
            'import numpy as np\n'
            'import pandas as pd\n\n'
            'from .constants import WEIGHT_NAMES\n'
            'from .judge import csv_state\n'
            'from .naming import eval_slug, slug\n\n\n'
            'def free_gpu():\n'
            '    """Release cached GPU memory. Callers drop their own model references first."""\n'
            '    gc.collect()\n'
            '    try:\n'
            '        import torch\n'
            '    except ImportError:\n'
            '        return\n'
            '    if torch.cuda.is_available():\n'
            '        torch.cuda.empty_cache()\n'
            '        torch.cuda.ipc_collect()\n'),
        'funcs': ['load_model_manual', 'stage_adapter_local', 'rejudge', 'eval_one', 'baseline_csv',
                  'reuse_zero_strength', '_boot_ci', '_model_stats'],
    },
    'hub.py': {
        'header': ('import json\nimport os\nfrom pathlib import Path\n\n'
                   'from .adapters import adapter_weights_present\nfrom .constants import WEIGHT_NAMES\n\n_HF = {}\n'),
        'funcs': ['hf_api', 'upload_repo_id', 'hf_adapter_complete', 'upload_adapter_record',
                  'push_records', 'report_push'],
    },
}


def notebook_source(path):
    nb = json.loads(Path(path).read_text(encoding='utf-8'))
    parts = []
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            lines = ''.join(c['source']).split('\n')
            parts.append('\n'.join(l for l in lines if not l.lstrip().startswith(('!', '%'))))
    return '\n'.join(parts)


def index_nodes(src):
    tree = compile(src, '<notebook>', 'exec', flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[n.name] = n
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                names = [x.id for x in ast.walk(t) if isinstance(x, ast.Name)]
                for nm in names:
                    out[nm] = n
    return out


def node_text(lines, node):
    start = node.lineno
    while start > 1 and lines[start - 2].lstrip().startswith('#'):     # keep the comment block above
        start -= 1
    return '\n'.join(lines[start - 1:node.end_lineno])


def code_names(node):
    """Names the function really uses as code (not words in docstrings or comments)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def transform(name, text, node, cfg_funcs):
    """Rewrite one function's text. Returns (new_text, needs_cfg)."""
    names = code_names(node)
    # the notebook uses `cfg` for an adapter's own config dict; free the name for the Config object
    text = re.sub(r'(?<![\w.])cfg(?![\w])', 'adapter_cfg', text)
    needs = bool(names & set(CONST_MAP)) or bool(names & (cfg_funcs - {name}))
    inserts = [LAZY[w] for w in sorted(names) if w in LAZY]
    if needs:
        lines = text.split('\n')
        def_idx = next(i for i, l in enumerate(lines) if re.match(r'\s*(async\s+)?def\s', l))
        end_idx = signature_end(lines, def_idx)
        sig = '\n'.join(lines[def_idx:end_idx + 1])
        for const, expr in CONST_MAP.items():       # `param=CONST` defaults cannot see cfg: resolve in the body
            for m in re.finditer(rf'(\w+)\s*=\s*{const}\b', sig):
                param = m.group(1)
                inserts.append(f'{param} = {expr} if {param} is None else {param}')
            sig = re.sub(rf'(\w+)(\s*=\s*){const}\b', r'\1\2None', sig)
        sig = re.sub(rf'^((?:async\s+)?def\s+{re.escape(name)}\()\s*\)', r'\1cfg)', sig, count=1)
        if 'cfg' not in sig.split('\n')[0].split('(', 1)[1][:4]:
            sig = re.sub(rf'^((?:async\s+)?def\s+{re.escape(name)}\()', r'\1cfg, ', sig, count=1)
        text = '\n'.join(lines[:def_idx] + sig.split('\n') + lines[end_idx + 1:])
        for const, expr in CONST_MAP.items():
            text = re.sub(rf'(?<![\w.]){const}\b', expr, text)
        for g in sorted(cfg_funcs - {name}):
            text = re.sub(rf'(?<!def )(?<![\w.]){re.escape(g)}\(\s*\)', f'{g}(cfg)', text)
            text = re.sub(rf'(?<!def )(?<![\w.]){re.escape(g)}\((?!cfg\b)(?!\s*\))', f'{g}(cfg, ', text)
    if inserts:
        text = insert_after_docstring(text, node, inserts)
    return text, needs


def signature_end(lines, def_idx):
    depth, i = 0, def_idx
    while True:
        depth += lines[i].count('(') - lines[i].count(')')
        if depth <= 0 and lines[i].rstrip().endswith(':'):
            return i
        i += 1


def insert_after_docstring(text, node, import_lines):
    lines = text.split('\n')
    def_idx = next(i for i, l in enumerate(lines) if re.match(r'\s*(async\s+)?def\s', l))
    i = signature_end(lines, def_idx)
    insert_at = i + 1
    first = lines[insert_at].strip() if insert_at < len(lines) else ''
    if first.startswith(('"""', "'''", 'r"""')):
        quote = '"""' if '"""' in first else "'''"
        j = insert_at
        if first.count(quote) < 2:
            j += 1
            while quote not in lines[j]:
                j += 1
        insert_at = j + 1
    indent = '    '
    block = [indent + l for l in import_lines]
    return '\n'.join(lines[:insert_at] + block + lines[insert_at:])


def main(nb_path, out_dir):
    src = notebook_source(nb_path)
    lines = src.split('\n')
    nodes = index_nodes(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # first pass: which ported functions need cfg (fixpoint over calls between them)
    all_funcs = [f for m in MANIFEST.values() for f in m['funcs']]
    missing = [f for f in all_funcs + [c for m in MANIFEST.values() for c in m.get('constants', [])] if f not in nodes]
    if missing:
        sys.exit(f'not found in notebook: {missing}')
    cfg_funcs = set(HAND_CFG_FUNCS)
    changed = True
    while changed:
        changed = False
        for f in all_funcs:
            if f in cfg_funcs:
                continue
            if code_names(nodes[f]) & (set(CONST_MAP) | cfg_funcs):
                cfg_funcs.add(f)
                changed = True

    for module, spec in MANIFEST.items():
        chunks = [spec['header']]
        names = spec.get('constants', []) + spec['funcs']
        for nm in sorted(names, key=lambda n: nodes[n].lineno):
            node = nodes[nm]
            text = node_text(lines, node)
            if nm in spec['funcs']:
                text, _ = transform(nm, text, node, cfg_funcs)
            chunks.append(text)
        body = '\n\n\n'.join(c.rstrip('\n') for c in chunks[1:])
        (out_dir / module).write_text(chunks[0].rstrip('\n') + '\n\n\n' + body + '\n', encoding='utf-8')
        print(f'wrote {module}: {len(names)} item(s)')

    # warn about functions used as values (not calls), which the call rewriter cannot see
    for module in MANIFEST:
        text = (out_dir / module).read_text(encoding='utf-8')
        for g in sorted(cfg_funcs):
            for m in re.finditer(rf'(?<![\w.\'"]){re.escape(g)}\b(?![\w\'"(])', text):
                line = text[:m.start()].count('\n') + 1
                ctx = text.split('\n')[line - 1].strip()
                if not ctx.startswith(('def ', 'async def ', 'from ', 'import ')):
                    print(f'WARNING {module}:{line}: {g} used without a call -> {ctx[:80]}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
