# Changelog

All notable changes to chester-ml are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [1.1.0] - 2026-03-12

### New Features

#### `sequential_steps` — multi-step jobs in one allocation

`run_experiment_lite` now accepts a `sequential_steps` list.  Each entry is a
dict of Hydra overrides merged on top of the base variant for that step.
Chester generates a single bash script (or local `&&`-chained command) that
runs all steps in series with `set -e`, so step 2 never runs if step 1 fails.

```python
run_experiment_lite(
    variant=vv,
    sequential_steps=[
        {"experiment.tasks": ["training"]},
        {"experiment.tasks": ["evaluate"], "wandb.resume": True},
    ],
    ...
)
```

Motivated by Isaac Sim's single-init constraint: training and evaluate must
be separate Python processes but can share one SLURM allocation.

- `sequential_steps=None` (default): identical behavior to prior versions — no
  regression.
- `sequential_steps=[]`: raises `ValueError`.
- In local debug mode (`launch_with_subprocess=False`), multi-step runs
  automatically force subprocess execution since Hydra cannot be initialized
  twice in-process.

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
