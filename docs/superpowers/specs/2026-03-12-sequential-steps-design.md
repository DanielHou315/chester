# Sequential Steps Design

**Date:** 2026-03-12
**Status:** Approved

## Problem

Isaac Sim can only be initialized once per Python process. When it shuts down, the
process dies. Running `training` then `evaluate` as two separate SLURM/SSH jobs
requires two queue slots and introduces a dependency ordering problem (evaluate must
not start before training finishes).

## Solution

Add a `sequential_steps: list[dict] | None` parameter to `run_experiment_lite`.
When provided, Chester generates a **single** bash script containing N sequential
Python commands (one per step), with `set -e` so the script aborts on first failure.
Each step dict contains Hydra overrides merged on top of the base variant.

## API

```python
run_experiment_lite(
    variant=vv,           # base variant: seed, env config, num_envs, etc.
    sequential_steps=[
        {"experiment.tasks": ["training"]},
        {
            "experiment.tasks": ["evaluate"],
            "experiment.evaluate.training_dir": "${hydra:runtime.output_dir}",
        },
    ],
    ...
)
```

- `sequential_steps=None` (default): identical behavior to today — no regression.
- Each step dict is shallow-merged on top of the base variant; absent keys inherit
  from the base.
- Hydra's own resolver handles interpolations like `${hydra:runtime.output_dir}`.
  All steps share the same `hydra.run.dir`, so cross-step path references resolve
  correctly at runtime without any Chester token substitution.

## Script Generation

### Refactor: `_build_step_command()`

Extract a single-step command builder from each backend's existing generator:

```python
def _build_step_command(task, step_overrides=None) -> str:
    # merge step_overrides into variant if provided
    # return one python command string
```

Multi-step generation loops:

```python
if sequential_steps:
    commands = [_build_step_command(task, step) for step in sequential_steps]
else:
    commands = [_build_step_command(task, None)]
```

### Generated Script (SSH/SLURM)

```bash
#!/bin/bash
# ... env setup, module load, uv sync, cd ...

set -e   # abort on first non-zero exit

# step 1/2
python -m main experiment.tasks=[training] hydra.run.dir=/path/to/logs/1_exp seed=1 ...

# step 2/2
python -m main experiment.tasks=[evaluate] \
    experiment.evaluate.training_dir=... \
    hydra.run.dir=/path/to/logs/1_exp seed=1 ...
```

### Local Backend

Sequential commands are chained with `&&` (same abort-on-failure semantics):

```bash
python -m main ... && python -m main ...
```

## Auto-pull & Job Store

No changes. The multi-step job is one submission → one SLURM job ID → one job file
in `.chester/auto_pull_jobs/`. Auto-pull polls and pulls the shared `log_dir` on
completion, which contains outputs from all steps.

## Error Handling

- `set -e` ensures step 2 never runs if step 1 exits non-zero.
- Empty `sequential_steps=[]` raises `ValueError` (must have at least one step).
- Step dicts may be empty (`{}`) — valid, means use base variant unchanged.

## Testing

- **`_build_step_command` unit tests**: `step_overrides=None` regression (identical to
  current output); with override dict (override keys appear in command, base keys
  preserved).
- **Multi-step script tests**: `set -e` present before commands; N commands for N
  steps; step comments (`# step 1/2`) present.
- **Local backend test**: commands joined with `&&`.
- **`sequential_steps=None` regression**: output identical to current single-step output.

## Files Changed

- `src/chester/backends/ssh.py` — extract `_build_step_command`, loop in `generate_script`
- `src/chester/backends/slurm.py` — same
- `src/chester/backends/local.py` — same, join with `&&`
- `src/chester/run_exp.py` — accept `sequential_steps`, pass through to backend
- `tests/test_sequential_steps.py` — new test file
