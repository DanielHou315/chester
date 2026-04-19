# Parameter Sweeps

Chester provides `VariantGenerator` for declarative parameter sweeps with support for cartesian products, dependent parameters, derived values, and advanced job scheduling (serial steps, SLURM dependencies).

## Import

```python
from chester.run_exp import VariantGenerator, run_experiment_lite, detect_local_gpus, flush_backend
```

---

## VariantGenerator Basics

`VariantGenerator` builds parameter combinations and generates experiments.

### Creating Variants

```python
vg = VariantGenerator()
vg.add('lr', [1e-3, 1e-4, 1e-5])
vg.add('batch_size', [32, 64])
# Cartesian product: 6 variants total
```

Iterate and run:

```python
for v in vg.variants():
    run_experiment_lite(stub_method_call=run_task, variant=v, mode='local', exp_prefix='sweep')
```

### `vg.add(key, values, hide=False, order=None)`

- **`key`** (str): Parameter name. Use dot notation for nested Hydra paths: `"model.hidden_dim"`, `"experiment.tasks"`, etc.
- **`values`** (list or callable): List of values, or a lambda for dependent parameters (see [Dependent Parameters](#dependent-parameters-lambda))
- **`hide=True`**: Exclude from auto-generated experiment name; parameter still passed to job
- **`order`**: `None` (default), `"serial"`, or `"dependent"` (see [Advanced Ordering](#advanced-ordering))

### Return Value

Each call to `vg.variants()` returns a list of dicts:

```python
variants = vg.variants()
# [
#   {'lr': 1e-3, 'batch_size': 32, 'exp_prefix': 'sweep'},
#   {'lr': 1e-3, 'batch_size': 64, 'exp_prefix': 'sweep'},
#   {'lr': 1e-4, 'batch_size': 32, 'exp_prefix': 'sweep'},
#   ...
# ]
```

Pass each variant to `run_experiment_lite()`:

```python
for v in vg.variants():
    run_experiment_lite(stub_method_call=run_task, variant=v, ...)
```

---

## Dependent Parameters (lambda)

A parameter's values can depend on values of previously-added parameters:

```python
vg.add('model', ['small', 'large'])
vg.add('hidden_dim', lambda model: [128] if model == 'small' else [256, 512])
```

This produces 3 variants:
- `{'model': 'small', 'hidden_dim': 128}`
- `{'model': 'large', 'hidden_dim': 256}`
- `{'model': 'large', 'hidden_dim': 512}`

The lambda receives the partial variant dict (keys added so far) and returns a list of values. Dependencies are resolved in add order.

---

## Derived Parameters

Derive a parameter deterministically from other parameters (always single value per variant):

```python
vg.add('dataset_paths', [['data/a', 'data/b'], ['data/c']])
vg.derive('dataset_names', lambda v: [os.path.basename(p) for p in v['dataset_paths']])
```

The `fn` receives the full variant dict and returns a single value (not a list). Derived parameters:
- Never vary across variants (function applied once per variant)
- Cannot be referenced by dependent parameters
- Cannot use `order=`

---

## Hidden Parameters

Exclude a parameter from the auto-generated experiment name while still passing it to the job:

```python
vg.add('seed', [1, 2, 3], hide=True)
vg.add('lr', [1e-3, 1e-4])
```

The experiment name will only include `lr`, not `seed`. Seeds vary across experiments but don't pollute the naming scheme.

Use `vg.variations()` to get the list of keys that appear in experiment names:

```python
variations = vg.variations()  # ['lr']
run_experiment_lite(..., variations=variations)
```

---

## Randomized Order

Shuffle variant iteration order:

```python
for v in vg.variants(randomized=True):
    run_experiment_lite(...)
```

**Incompatible with `order=` parameters.** If any parameter has `order='serial'` or `order='dependent'`, do not use `randomized=True`.

---

## Advanced Ordering

Chester supports two advanced scheduling modes: serial steps within a single job, and SLURM job dependencies.

### `order="serial"` — Multi-Step Single Job

All values for a serial parameter are written as **sequential commands in one script**. The second command runs after the first succeeds (or fails, depending on the runner).

#### Example: Training then Evaluation

```python
vg = VariantGenerator()
vg.add('seed', [1, 2, 3])
vg.add('experiment.tasks', [['training'], ['evaluate']], order='serial')
```

This produces 3 variants. Each variant is **one SLURM job (or one SSH/local session)** with two sequential commands:

```bash
# Job for seed=1
python -m main experiment.tasks=['training'] seed=1 hydra.run.dir=...
python -m main experiment.tasks=['evaluate'] seed=1 hydra.run.dir=...
```

Behavior:
- The serial key is **collapsed**: all values become commands in a single script, executed sequentially
- Serial keys are **excluded** from `variations()` and experiment naming
- Only **one** `order="serial"` key per sweep
- Works on all backends (local, SSH, SLURM)
- Each Hydra command receives the full variant args plus the current serial value

#### Constraints

- Must be a concrete list (not a lambda or derived parameter)
- At least 2 values required
- Cannot be combined with `randomized=True`
- Cannot be used on `vg.derive()`

---

### `order="dependent"` — SLURM Job Dependencies

Each value gets its own SLURM job, chained via `sbatch --dependency=afterok:<jobid>`. The second job only starts after the first succeeds.

#### Example: Training then Evaluation

```python
vg = VariantGenerator()
vg.add('seed', [1, 2, 3])
vg.add('task', ['training', 'evaluate'], order='dependent')
```

This produces 6 SLURM jobs:

```
Job 1 (seed=1, task=training)
Job 2 (seed=1, task=evaluate) ← depends on Job 1
Job 3 (seed=2, task=training)
Job 4 (seed=2, task=evaluate) ← depends on Job 3
Job 5 (seed=3, task=training)
Job 6 (seed=3, task=evaluate) ← depends on Job 5
```

Multiple dependent keys create per-combination chains:

```python
vg.add('task', ['training', 'evaluate'], order='dependent')
vg.add('phase', ['warmup', 'finetune'], order='dependent')
# Ordering: each (task, phase) combo depends on all lower-priority combos completing
```

#### Non-SLURM Safety Check

If `order='dependent'` is used with a non-SLURM backend (local, SSH, or other), Chester raises `ValueError`. Pass `skip_dependency_check=True` to suppress for debug runs where all variants should run unordered:

```python
run_experiment_lite(
    ...,
    mode='local',
    skip_dependency_check=True  # suppress error; will not respect dependencies
)
```

#### Constraints

- Must be a concrete list (not a lambda or derived parameter)
- At least 2 values required
- Cannot be combined with `randomized=True`
- Cannot be used on `vg.derive()`
- **SLURM-only** (error on other backends unless `skip_dependency_check=True`)

---

## `detect_local_gpus()`

Detect available local GPU IDs as a list of strings:

```python
from chester.run_exp import detect_local_gpus

gpus = detect_local_gpus()  # e.g. ['0', '1', '2', '3']
max_procs = max(1, len(gpus))
```

Resolution order:
1. `$CUDA_VISIBLE_DEVICES` env var (if set)
2. `nvidia-smi --query-gpu=index` (if available)
3. Fallback: `["0"]`

Use to set `max_num_processes`:

```python
run_experiment_lite(
    ...,
    max_num_processes=max(1, len(detect_local_gpus())),
)
```

---

## `flush_backend(mode)` for SSH Batch Mode

When using SSH batch-GPU mode (`batch_gpu: N` in config), call `flush_backend` after the variant loop to dispatch all accumulated jobs:

```python
from chester.run_exp import flush_backend

for v in vg.variants():
    run_experiment_lite(..., mode='myserver')

flush_backend('myserver')  # fire all accumulated scripts
```

**This is a no-op for non-batch backends** (local, SLURM, standard SSH). Only use if your SSH config specifies `batch_gpu: N > 0`.

---

## Full Launcher Pattern

A complete launcher showing all sweep and backend features:

```python
import os
import click
from chester.run_exp import (
    VariantGenerator,
    run_experiment_lite,
    detect_local_gpus,
    flush_backend,
)

def run_task(variant):
    """Stub method signature for run_experiment_lite."""
    pass

@click.command()
@click.argument('mode', default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
@click.option('--wait/--no-wait', default=False)
def main(mode, debug, dry, wait):
    vg = VariantGenerator()
    exp_prefix = 'my_sweep'

    # Execution metadata
    vg.add('exp_prefix', [exp_prefix])
    vg.add('debug', [debug])

    # Parameter sweep
    vg.add('lr', [1e-3, 1e-4])
    vg.add('batch_size', [32, 64])
    vg.add('seed', [1, 2, 3], hide=True)

    # Serial steps: training then evaluate in one job
    vg.add('experiment.tasks', [['training'], ['evaluate']], order='serial')

    variants = vg.variants()
    print(f'{len(variants)} variants, mode={mode}')

    for v in variants:
        run_experiment_lite(
            stub_method_call=run_task,
            variant=v,
            mode=mode,
            exp_prefix=exp_prefix,
            variations=vg.variations(),
            hydra_enabled=True,
            dry=dry,
            wait_processes=debug or wait,
            max_num_processes=max(1, len(detect_local_gpus())),
            skip_dependency_check=debug,
            slurm_overrides={'time': '4:00:00', 'mem_per_gpu': '32G'},
        )
        if debug:
            break  # only run one variant in debug mode

    flush_backend(mode)  # no-op unless batch_gpu SSH mode

if __name__ == '__main__':
    main()
```

---

## Common Patterns

### Simple Cartesian Product

```python
vg = VariantGenerator()
vg.add('model', ['resnet', 'vgg'])
vg.add('lr', [1e-3, 1e-4])
vg.add('seed', [1, 2, 3], hide=True)
# 2 * 2 * 3 = 12 variants
```

### Dependent Hyperparameters

Adjust learning rate schedule or model size based on dataset:

```python
vg.add('dataset', ['small', 'large'])
vg.add('batch_size', lambda dataset: [32] if dataset == 'small' else [128, 256])
vg.add('lr', lambda dataset: [1e-3] if dataset == 'small' else [1e-4, 1e-5])
# 2 datasets: small gets 1 batch_size, large gets 2 batch_sizes and 2 lrs
# Total: 1 + (2 * 2) = 5 variants
```

### Multi-Stage Experiment

Train a model, then evaluate on multiple metrics in separate serial steps:

```python
vg.add('model_type', ['baseline', 'improved'])
vg.add('seed', [1, 2], hide=True)
vg.add('stage', [['train'], ['eval_accuracy'], ['eval_robustness']], order='serial')
# 2 * 2 variants, each a single job with 3 serial commands
```

### SLURM Dependency Chain

Long-running pretraining followed by fine-tuning:

```python
vg.add('dataset', ['imagenet', 'coco'])
vg.add('phase', ['pretrain', 'finetune'], order='dependent')
# 2 datasets * 2 phases = 4 SLURM jobs
# For each dataset: finetune waits for pretrain
```

---

## Troubleshooting

### Error: "order='dependent' not supported on non-SLURM backend"

Use a SLURM cluster or pass `skip_dependency_check=True` for local debug:

```python
run_experiment_lite(..., mode='local', skip_dependency_check=True)
```

### Serial key not appearing in experiment name

This is correct behavior. Serial keys are excluded from `variations()` and naming. If you want to include it, remove `order='serial'`.

### Too many variants

Check dependent parameter lambdas or cartesian product size:

```python
print(f'{len(vg.variants())} variants')
```

### Randomized + order= conflict

Remove `randomized=True` or remove all `order=` parameters from any key.

```python
# ✗ Error
for v in vg.variants(randomized=True):  # and vg has order='serial' or 'dependent'
    ...

# ✓ OK
for v in vg.variants(randomized=True):  # no order= parameters
    ...
```
