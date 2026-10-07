import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from orthovac.config import Config


def make_csv(path, n=40, scored=True, with_response=True):
    """A results CSV shaped like the eval output: 8 questions x n/8 answers, optional scores."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'question_id': [f'q{i % 8}' for i in range(n)],
        'response': ['answer'] * n if with_response else [None] * n,
        'aligned': rng.uniform(40, 100, n) if scored else [np.nan] * n,
        'coherent': rng.uniform(60, 100, n) if scored else [np.nan] * n,
    })
    df.to_csv(path, index=False)
    return path


def make_todo(path, tasks):
    path = Path(path)
    path.write_text(json.dumps({'schema_version': '1.0', 'tasks': tasks}), encoding='utf-8')
    return path


def task(task_id, source, target, strengths, category='Missing pair', status='Not started', model='Qwen'):
    return {'task_id': task_id, 'category': category, 'source': source, 'target': target,
            'strengths': list(strengths), 'status': status, 'model': model}


@pytest.fixture
def cfg(tmp_path):
    root = tmp_path / 'Safety-projections' / 'Qwen'
    return Config(side='Qwen', root=root, todo_paths=(tmp_path / 'todo.json',), summary_dir=root,
                  hf_namespace='tester', local_adapter_base=tmp_path / 'adapters',
                  local_responses=tmp_path / 'responses', n_boot=20)
