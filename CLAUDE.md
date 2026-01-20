# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Chester (`chester-ml` on PyPI) is a Python experiment launcher for ML workflows. It automates launching experiments across local machines, SSH hosts, SLURM clusters, and AWS EC2, with automatic result synchronization.

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
│   ├── run_exp.py         # Main launcher (run_experiment_lite, VariantGenerator)
│   ├── run_exp_worker.py  # Worker that deserializes and runs experiments
│   ├── config.py          # YAML-based config loader
│   ├── slurm.py           # Command generation for SLURM/SSH
│   ├── auto_pull.py       # Automatic result synchronization
│   ├── hydra_utils.py     # Hydra integration
│   └── scheduler/         # GPU scheduler utilities (autobot)
├── tests/                 # Tests
├── examples/              # Example configs
└── pyproject.toml         # Package definition (chester-ml v0.2.1)
```

## Key Commands

```bash
uv sync                                    # Install package and dependencies
uv run python tests/test_local.py          # Run local tests
uv run python -m chester.config            # Debug config loading
uv run python -m chester.auto_pull --help  # Auto-pull CLI
```

## Configuration System

Chester uses YAML configuration files. Place `chester.yaml` in your project root.

**Config search order:**
1. `CHESTER_CONFIG_PATH` environment variable
2. Parent directories from cwd (stops at .git root)
3. Defaults for local-only usage

**Access config in code:**
```python
from chester import config
print(config.PROJECT_PATH)
print(config.REMOTE_DIR['gl'])
print(config.RSYNC_INCLUDE)
print(config.PREPARE_COMMANDS)
```

**Key config attributes:**
- `PROJECT_PATH`, `LOG_DIR`: Paths
- `HOST_ADDRESS`, `REMOTE_DIR`, `REMOTE_LOG_DIR`: Host mappings
- `SSH_HOSTS`: List of SSH host names
- `REMOTE_HEADER`, `MODULES`, `CUDA_MODULE`: SLURM settings
- `PACKAGE_MANAGER`, `CONDA_ENV`, `SYNC_ON_LAUNCH`: Package manager
- `RSYNC_INCLUDE`, `RSYNC_EXCLUDE`: Code sync patterns
- `PREPARE_COMMANDS`: Custom setup commands

## Architecture

### Core Flow
1. **Launcher** (`run_exp.py`): Serializes experiment configs, dispatches to execution mode
2. **Worker** (`run_exp_worker.py`): Deserializes and executes experiments via cloudpickle
3. **VariantGenerator**: Creates parameter combinations with dependency support
4. **Auto-pull** (`auto_pull.py`): Polls remote hosts for job completion, pulls results

### Execution Modes
- `local`: Direct subprocess execution
- `local_singularity`: Singularity container execution
- SLURM clusters: `gl`, `satori`, `seuss`, `psc` (via `remote_header`)
- SSH hosts: Configured in `ssh_hosts` list
- `autobot`: Custom GPU scheduler
- `ec2`: AWS EC2 with S3 sync

### Auto-Pull System

The auto-pull system (`auto_pull.py`) tracks remote job status:

**Job tracking:**
- PID saved to `.chester_pid` on launch
- `.done` marker created on successful completion
- Poller checks both to determine status

**Status detection:**
```
.done exists + process dead → done (pull results)
.done exists + process alive → done_orphans (kill, then pull)
no .done + process dead → failed (pull logs)
no .done + process alive → running (keep polling)
```

**Key functions:**
- `get_remote_pid()`: Read PID from `.chester_pid`
- `check_process_running()`: Check if PID is alive via SSH
- `kill_process_tree()`: SIGTERM → 5s wait → SIGKILL
- `check_job_status()`: Determine job state
- `poll_and_pull()`: Main polling loop
- `pull_extra_dirs()`: Pull additional directories beyond log_dir

**Extra directories (`extra_pull_dirs` parameter):**
- `_resolve_extra_pull_dirs()`: Resolve paths to (local, remote) pairs
- Relative paths: `PROJECT_PATH/path` locally, `REMOTE_DIR[mode]/path` on remote
- Absolute paths (starting with `/`): Same path on both

### Command Generation (`slurm.py`)

**Package manager setup:**
- `get_package_manager_setup_commands()`: Generates shell commands for uv/conda setup
- `get_python_command()`: Wraps python command (e.g., `uv run python`)

**Command generators:**
- `to_local_command()`: Simple CLI args format
- `to_ssh_command()`: Full bash script for SSH execution
- `to_slurm_command()`: SLURM batch script with singularity support

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
        auto_pull=True,  # Enable auto-pull for remote modes
    )
```

## Dependencies

**Core:** numpy, cloudpickle, joblib, python-dateutil, pyyaml

**Optional:**
- `hydra-core`, `omegaconf`: For Hydra integration
- `boto3`, `awscli`: For EC2/S3 support
- `wandb`, `tensorboard`: For experiment tracking

## Testing Changes

**Test auto-pull functions:**
```bash
uv run python -c "
from chester.auto_pull import get_remote_pid, check_process_running, check_job_status

job = {
    'host': 'myserver',
    'remote_log_dir': '/path/to/logs',
    'local_log_dir': '/local/path',
    'exp_name': 'test'
}
print(check_job_status(job))
"
```

**Test config loading:**
```bash
uv run python -m chester.config
```

## Publishing

```bash
# Bump version in pyproject.toml
uv build
uv publish  # or: twine upload dist/*
```
