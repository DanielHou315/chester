# Hydra Integration

Chester can pass experiment parameters as Hydra overrides instead of `--key value` CLI args.

**Enable per launch:**
```python
run_experiment_lite(
    ...,
    hydra_enabled=True,
    hydra_flags={'multirun': False},   # optional Hydra flags
)
```

**Set config path in `.chester/config.yaml`:**
```yaml
hydra_config_path: configs    # relative to project root
```

## Command format

Without Hydra (`hydra_enabled=False`, default):
```bash
python worker.py --variant_data <base64> --log_dir /path/to/logs
```

With Hydra (`hydra_enabled=True`):
```bash
python -m mymodule lr=0.001 batch_size=32 hydra.run.dir=/path/to/logs
```

Note: `hydra.run.dir` is always injected automatically from `log_dir`.

## Hydra flags

Pass additional Hydra flags via `hydra_flags`:

```python
hydra_flags={'multirun': True}     # → adds --multirun
hydra_flags={'config-name': 'exp'} # → adds --config-name=exp
hydra_flags={'multirun': False}    # → flag omitted
```

Boolean `True` → flag added with no value (`--multirun`).
Boolean `False` / `None` → flag omitted.
Other value → `--key=value`.

## OmegaConf interpolations

Chester passes OmegaConf interpolation strings through unquoted so they resolve at runtime:

```python
# In the variant:
vg.add('schedule', ["${eval:'[0.1 * i for i in range(10)]'}"])
vg.add('data_dir', ['${oc.env:DATA_DIR,/tmp/data}'])
vg.add('hidden', ["${eval:'512 * 2'}"])

# Generated command:
# python -m main schedule=${eval:'[0.1 * i for i in range(10)]'} data_dir=${oc.env:DATA_DIR,/tmp/data} ...
```

Any value matching `${...}` is passed without quoting. Strings with spaces are double-quoted. Lists are formatted as `[v1,v2,v3]`.

## Typical launcher pattern

```python
import os

run_experiment_lite(
    stub_method_call=run,
    variant=v,
    mode=mode,
    hydra_enabled=True,
    hydra_flags={},
    script='main',                          # run as: python -m main
    python_command=f"{os.path.join(os.environ.get('VIRTUAL_ENV', '.venv'), 'bin', 'python')} -m",
    exp_prefix=exp_prefix,
    sub_dir='scorer_dynamics',
)
```

When using a module script (`python -m main`), set `script='main'` and `python_command='python -m'` (or the full venv path for local).
