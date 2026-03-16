# Changelog

All notable changes to chester-ml are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.2.0] - 2026-03-16

### New Features

#### `sequential=True` — SLURM job dependency chains

`vg.add()` now accepts `sequential=True` to create inter-job SLURM dependency
chains via `sbatch --dependency=afterok:<jobid>`.  Jobs for later values in the
list only start after their predecessors complete successfully.

```python
vg = VariantGenerator()
vg.add("task", ["training", "evaluate"], sequential=True)
vg.add("seed", [1, 2, 3])

for v in vg.variants():
    run_experiment_lite(stub_method_call=run_task, variant=v, mode="gl", ...)
# For each seed: evaluate waits for training to finish
```

Multiple sequential fields create per-field independent chains.  Non-SLURM
modes raise `ValueError` unless `skip_dependency_check=True` is passed.

#### `confirm_action()` — unified confirmation prompts

All interactive confirmation prompts (`query_yes_no`, fresh-start input) are
now handled by `confirm_action(message, default, skip)`.  `query_yes_no` is
deprecated with a `DeprecationWarning`.

### Breaking Changes

- **`sequential_steps` removed.** The `sequential_steps` parameter on
  `run_experiment_lite` and all backends has been removed.  Use
  `vg.add(..., sequential=True)` for SLURM dependency chains instead.

### Bug Fixes

- **SSH backend syntax error** — fixed dangling `else` at wrong indentation
  level in `SSHBackend.generate_script()` that caused `SyntaxError` on import.

---

## [1.1.0] - 2026-03-12

### New Features

#### `vg.derive()` — derived parameters bypassing OmegaConf eval

`VariantGenerator.derive(key, fn)` registers a post-sweep derivation.  `fn`
receives the full variant dict and returns a single concrete value, which is
then included as a concrete Hydra CLI override (bypassing any `${eval:...}`
expression in the YAML).

```python
vg.add("experiment.training.env.num_train_sim", [127, 1])
vg.derive(
    "experiment.training.env.num_train_real",
    lambda v: 128 - v["experiment.training.env.num_train_sim"],
)
vg.derive(
    "experiment.training.env.sim_fixed_asset_scale",
    lambda v: 2.0 if v["experiment.training.env.num_train_sim"]
                   > v["experiment.training.env.num_train_real"]
              else 1.0,
)
```

Dotted Hydra keys work naturally because `fn` indexes the dict with string
keys.  Derivations are applied in registration order; register them in
dependency order.  See `vg.derive()` docstring for circular-reference
behaviour.

#### Persistent job store

Auto-pull job state is now tracked as individual JSON files under
`.chester/auto_pull_jobs/` (one file per job) instead of a shared manifest.
This eliminates manifest corruption on concurrent writes and makes job state
inspectable with standard tools.

#### `chester` CLI

A unified `chester` command is now installed as an entry point:

```bash
chester pull-remote          # pull results from a remote job
chester --help
```

#### `fresh=True` — clean restart

`run_experiment_lite(fresh=True)` deletes existing experiment output
directories before launching.  Prompts for confirmation (accepts `y` / `yes`);
pass `confirm=True` to skip.

#### `rsync_pull_exclude` config

New `rsync_pull_exclude` list in `.chester/config.yaml` controls which files
are excluded when pulling results from remote.  Defaults exclude large
checkpoint files (`last_*.pth`) to keep pulls fast.

### Bug Fixes

- **`.done` marker placed on host, not inside Singularity container** — the
  marker was previously written inside the container, where `log_dir` may not
  be mounted, causing auto-pull to never detect job completion.

### Internal

- Git snapshot now captures submodule dirty diffs and untracked symlinks.
- `auto_pull=True` is now the default for `run_experiment_lite`.

---

## [1.0.2] - 2025-xx-xx

Initial tracked release.
