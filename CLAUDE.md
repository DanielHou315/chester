# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chester is a Python experiment launcher and management framework for ML workflows. It automates launching experiments across: local machines, SLURM clusters, SSH hosts, and AWS EC2.

## Development Setup

**Use uv for all development:**
```bash
uv sync                    # Install dependencies
uv run python <script>     # Run scripts
uv run pytest tests/       # Run tests
```

**Package structure:**
```
chester/
├── src/chester/           # Package source code
│   ├── run_exp.py         # Main launcher
│   ├── config.py          # YAML-based config loader
│   ├── slurm.py           # SLURM command generation
│   ├── hydra_utils.py     # Hydra integration (requires hydra-core)
│   └── scheduler/         # GPU scheduler utilities
├── tests/                 # Tests
├── examples/              # Example configs
└── pyproject.toml         # Package definition
```

## Key Commands

```bash
uv sync                                    # Install package and dependencies
uv run python tests/test_local.py          # Run local tests
uv run python -m chester.config            # Debug config loading
```

## Configuration System

Chester uses YAML configuration files. Place `chester.yaml` in your project root:

```yaml
# chester.yaml
log_dir: data
host_address:
  local: ""
  gl: "gl"
ssh_hosts: []
remote_dir:
  gl: /path/to/project
remote_header:
  gl: |
    #!/usr/bin/env bash
    #SBATCH --gpus=$gpus
```

**Config search order:**
1. `CHESTER_CONFIG_PATH` environment variable
2. Parent directories from cwd (stops at .git root)
3. Defaults for local-only usage

**Access config in code:**
```python
from chester import config
print(config.PROJECT_PATH)
print(config.REMOTE_DIR['gl'])
```

## Architecture

### Core Flow
1. **Launcher** (`run_exp.py`): Serializes experiment configs, dispatches to execution mode
2. **Worker** (`run_exp_worker.py`): Deserializes and executes experiments via cloudpickle
3. **VariantGenerator**: Creates parameter combinations with dependency support

### Execution Modes
- `local`: Direct subprocess execution
- `local_singularity`: Singularity container execution
- SLURM clusters: `gl`, `satori`, etc.
- SSH hosts: Configured in `ssh_hosts` list
- `ec2`: AWS EC2 with S3 sync

## Writing Experiments

```python
from chester.run_exp import run_experiment_lite, VariantGenerator

def run_task(variant, log_dir, exp_name):
    # Your training code here
    pass

vg = VariantGenerator()
vg.add('learning_rate', [0.001, 0.01])
vg.add('batch_size', [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=run_task,
        variant=v,
        mode='local',
        exp_prefix='my_experiment',
    )
```

## Dependencies

**Core:** numpy, cloudpickle, joblib, python-dateutil, pyyaml

**Optional:**
- `hydra-core`, `omegaconf`: For Hydra integration
- `boto3`, `awscli`: For EC2/S3 support
- `wandb`, `tensorboard`: For experiment tracking
