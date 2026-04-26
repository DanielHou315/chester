# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chester (`chester-ml` on PyPI) is a Python experiment launcher for ML workflows. It serializes your experiment function and variant parameters, then dispatches them to a configured execution backend — local subprocess, SSH host, or SLURM cluster — with optional Singularity container support.

## Development Setup

**Use uv for all development:**
```bash
uv sync                    # Install dependencies
uv run pytest tests/       # Run tests
```

**Package structure:**
```
chester/
├── src/chester/           # Package source code
│   ├── run_exp.py         # Main launcher (run_experiment_lite, VariantGenerator)
│   ├── run_exp_worker.py  # Worker that deserializes and runs experiments
│   ├── config.py          # Config loader (reads .chester/config.yaml)
│   ├── backends/          # Backend implementations
│   │   ├── __init__.py    # create_backend() factory
│   │   ├── base.py        # BackendConfig, SlurmConfig, SingularityConfig dataclasses
│   │   ├── local.py       # LocalBackend
│   │   ├── ssh.py         # SSHBackend (including batch_gpu mode)
│   │   └── slurm.py       # SlurmBackend
│   └── hydra_utils.py     # Hydra override formatting
├── tests/                 # Tests
├── docs/                  # Documentation
│   ├── index.md           # Docs hub
│   ├── configuration.md   # .chester/config.yaml reference
│   ├── backends.md        # Backend type reference
│   ├── singularity.md     # Singularity container reference
│   ├── parameter-sweeps.md # VariantGenerator reference
│   ├── hydra.md           # Hydra integration
│   ├── git-snapshot.md    # Git snapshot feature
│   ├── submodule-pinning.md # Submodule commit pinning
│   ├── legacy/            # Migration guide from chester 1.x
│   └── examples/          # Annotated config examples
├── examples/              # Example launchers
└── pyproject.toml         # Package definition
```

## Key Commands

```bash
uv sync                          # Install package and dependencies
uv run pytest tests/             # Run all tests
uv run python -m chester.config  # Debug config loading
```

## Architecture

### Core Flow
1. **Launcher** (`run_exp.py`): Serializes experiment configs, dispatches to execution backend
2. **Worker** (`run_exp_worker.py`): Deserializes and executes experiments
3. **VariantGenerator**: Creates parameter combinations with dependency support
4. **Backends** (`backends/`): Generate bash scripts and submit/execute them

### Execution Backends

All backends are configured in `.chester/config.yaml` under `backends:`.

| Backend type | How it runs | Key config |
|---|---|---|
| `local` | subprocess / Popen | `prepare` |
| `ssh` | nohup via SSH | `host`, `remote_dir`, `batch_gpu` |
| `slurm` | sbatch via SSH | `host`, `remote_dir`, `modules`, `slurm:` block |

All backends support Singularity via a `singularity:` block.

### Backend Dispatch

`create_backend(config, project_config)` in `backends/__init__.py` is the factory. Add new backends by:
1. Implementing `generate_script()` and `submit()` in a new class inheriting `Backend`
2. Registering the type in the factory and `VALID_BACKEND_TYPES`

### Config System (`config.py`)

- `load_config(search_from=None)` — finds and loads `.chester/config.yaml`
- `get_backend(name, cfg)` — returns `BackendConfig` for a named backend
- Global `singularity:` block is inherited by all backends and can be overridden per-backend

## Writing Experiments

```python
from chester.run_exp import run_experiment_lite, VariantGenerator, detect_local_gpus, flush_backend

def run_task(variant, log_dir, exp_name):
    pass

vg = VariantGenerator()
vg.add('lr', [0.001, 0.01])
vg.add('batch_size', [32, 64])
vg.add('task', ['training', 'evaluate'], order='serial')   # single-job, sequential steps

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=run_task,
        variant=v,
        mode='gl',             # backend name from .chester/config.yaml
        exp_prefix='my_exp',
        hydra_enabled=True,
        skip_dependency_check=False,
    )

flush_backend('gl')            # no-op unless batch_gpu SSH mode
```

## Testing

```bash
uv run pytest tests/ -v                            # all tests
uv run pytest tests/test_backend_slurm.py -v      # single file
```

## Publishing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the detailed release workflow.

```bash
python scripts/bump_version.py [patch|minor|major]
uv build
uv publish
```
