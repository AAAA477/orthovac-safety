# orthovac-safety

Build, evaluate and plot vaccinated LoRA adapters (column-space projection). The Colab notebook only calls
functions; the logic lives in the `orthovac` package and is tested on a laptop.

## Colab quickstart

Open `notebooks/run_colab.ipynb` in Colab (File -> Open notebook -> GitHub -> `AAAA477/orthovac-safety`),
set `TODO_PATHS` in the settings cell, and run the cells in order:

1. install the package, 2. settings, 3. `ov.setup(...)`, `ov.harden_judge()`, `ov.show_plan(cfg)`,
4. build (and upload) adapters, 5. the gate: one evaluation, checked to have only `aligned` and `coherent`,
6. the full run, 7. merge summaries and sync To_Do statuses, 8. line graphs, 9. the result finder.

The plan is read only from the To_Do file(s) in `TODO_PATHS`, in task-id order. The read is verified and printed;
a missing file, bad JSON, or no open tasks stops the run.

## Where your data is stored

On Colab everything goes to Google Drive under `My Drive/Safety-projections/<side>/`: evaluation CSVs in
`runs/<donor>/<strength>/results/`, baselines in `runs/_baselines/`, summaries as `<SIDE>_SUMMARY.csv` (merged) and
`<SIDE>_SUMMARY.shard<i>.csv` (one per session), the event log in `runs/_log/`, graphs in `runs/_figures/`.
`setup()` prints the exact paths and says whether Drive is mounted. The notebook mounts Drive in its own cell. The Colab
VM's own disk (adapters being built, raw responses) is temporary; every finished result is copied to Drive.

## Two sessions at once

Open the notebook in two Colab sessions. Set `NUM_SHARDS = 2` in both, `SHARD = 0` in one and `SHARD = 1` in the
other. Each run belongs to exactly one shard, each shard writes its own summary file, and neither writes the To_Do
file. When both have finished, the `RUN_MERGE` session (`SHARD = 0`) runs the merge cell; only that session
merges and syncs.

Two sessions with the same `SHARD` double-spend the judge and GPU and corrupt each other's summary. Give every
session a different `SHARD`.

## Nothing is lost

After every evaluation, before the next one starts: the CSV is copied to Drive and checksum-verified, a row is
saved to the shard's summary file, and one line is appended to `runs/_log/shard<i>.jsonl`.
If the judge's API fails (rate limit, quota), calls are retried; an exhausted quota stops the run with a clear
message and the unjudged files are re-judged on the next run without regenerating.
To repair unjudged files without regenerating, run `await ov.rejudge_unjudged(cfg, plan)`: it re-judges only
this shard's unjudged or partly judged results (no GPU) and returns how many it fixed.

## Settings

`ov.setup(side, root, todo_paths, shard, num_shards, ...)` returns a `Config`; see `orthovac/config.py` for every
field. Plot settings live in `orthovac.plots.LINE_CFG` and can be overridden per call.

## Tests

```bash
pip install -e ".[test]"
python -m pytest
```

Set `MODEL_ORGANISMS_DIR` to a checkout of the `em_organism_dir` repo to also run the judge-hardening test against
the real judge code.

## Legacy

`notebooks/legacy/` holds the original notebooks, which still contain the dose-response plot, heatmap, audit tables
and cross-model section that are not ported yet. `tools/port.py` regenerates `constants.py`, `naming.py`,
`adapters.py`, `evaluate.py` and `hub.py` from the legacy notebook; those files are generated, do not edit them.
