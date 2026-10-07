"""Write notebooks/run_colab.ipynb: a thin notebook that only calls orthovac functions."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

CELLS = [
    md("# Safety projection on Colab\n\n"
       "Run the cells in order. The plan comes from the To_Do file(s) named below, in task-id order. "
       "To run two sessions at once, give each its own `SHARD` (0 and 1) and set `NUM_SHARDS = 2` in both."),
    code("!pip install -q \"git+https://github.com/AAAA477/orthovac-safety.git\""),
    code("#@title Settings { display-mode: \"form\" }\n"
         "SIDE = 'Qwen'\n"
         "TODO_PATHS = ['/content/drive/MyDrive/Safety-projections/Qwen/Qwen_Experiment_To_Do.json']\n"
         "SHARD, NUM_SHARDS = 0, 1\n"
         "RUN_UPLOAD = True"),
    code("import orthovac as ov\n\n"
         "cfg = ov.setup(side=SIDE, todo_paths=TODO_PATHS, shard=SHARD, num_shards=NUM_SHARDS)\n"
         "ov.harden_judge()          # judge retries; a quota failure stops the run with a clear message\n"
         "plan = ov.show_plan(cfg)   # the verified To_Do read and the ordered plan"),
    code("ov.build_adapters(cfg, plan)\n"
         "if RUN_UPLOAD:\n"
         "    ov.upload_adapters(cfg, plan)"),
    code("# Gate: one evaluation, then check the output has exactly the two score columns.\n"
         "from orthovac.judge import score_columns\n\n"
         "await ov.run_evals(cfg, plan, limit=1)\n"
         "first = sorted(cfg.runs_root.rglob('*_eval-*.csv'))[0]\n"
         "print(first.name, '->', score_columns(first))   # expect ['aligned', 'coherent']"),
    code("# Every evaluation is saved (verified Drive copy, summary row, log line) before the next one starts.\n"
         "waiting = await ov.run_evals(cfg, plan)"),
    code("# Run once, after every shard has finished.\n"
         "ov.merge_summaries(cfg)\n"
         "ov.sync_status(cfg, plan)"),
    code("ov.line_graphs(cfg)       # PNG + editable SVG under runs/_figures"),
    code("found, new_ok, new_bad = ov.find_results(cfg)   # anything on disk the summaries do not know yet"),
]


def build(path):
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'colab': {'provenance': [], 'gpuType': 'T4'}, 'accelerator': 'GPU'}
    nbf.validate(nb)
    nbf.write(nb, path)


if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'notebooks' / 'run_colab.ipynb'
    build(out)
    print('wrote', out)
