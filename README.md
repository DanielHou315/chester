# Chester

Chester is a Python experiment launcher for ML workflows. It automates launching experiments across local machines, SSH hosts, SLURM clusters, and AWS EC2, with automatic result synchronization.

## Installation

```bash
pip install chester-ml
```

Or with uv:
```bash
uv add chester-ml
```

## Quick Start

1. Create a `chester.yaml` in your project root:

```yaml
log_dir: data

# Package manager: 'uv' or 'conda'
package_manager: uv
sync_on_launch: true

# SSH hosts for remote execution
ssh_hosts: [myserver]
host_address:
  local: ""
  myserver: "user@myserver.com"
remote_dir:
  myserver: /home/user/myproject
```

2. Write a launcher script:

```python
from chester.run_exp import run_experiment_lite, VariantGenerator

def run_task(variant, log_dir, exp_name):
    # Your training code here
    print(f"Running {exp_name} with lr={variant['lr']}")

vg = VariantGenerator()
vg.add('lr', [0.001, 0.01])
vg.add('batch_size', [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=run_task,
        variant=v,
        mode='local',  # or 'myserver' for SSH
        exp_prefix='my_experiment',
    )
```

3. Run:
```bash
python my_launcher.py
```

---

## Configuration (`chester.yaml`)

Chester uses a YAML configuration file. Place `chester.yaml` in your project root.

### Config Search Order

1. `CHESTER_CONFIG_PATH` environment variable
2. Parent directories from current working directory (stops at `.git` root)
3. Built-in defaults for local-only usage

### Full Configuration Reference

```yaml
# ============================================================
# CORE SETTINGS
# ============================================================

# Local directory for experiment logs (relative to project root)
log_dir: data

# ============================================================
# PACKAGE MANAGER
# ============================================================

# Package manager: 'uv' or 'conda'
package_manager: uv

# Whether to sync environment on remote before running (uv sync / conda env update)
sync_on_launch: true

# For conda: environment name and command
conda_env: myenv        # Required if package_manager is 'conda'
conda_command: conda    # or 'mamba'

# ============================================================
# CUSTOM SETUP COMMANDS
# ============================================================

# Commands to run after package manager setup (replaces prepare.sh)
prepare_commands:
  - export PYTHONPATH=$PWD:$PYTHONPATH
  - source /opt/ros/noetic/setup.bash

# ============================================================
# RSYNC PATTERNS
# ============================================================

# Patterns for syncing code to remote hosts
rsync_include:
  - src/
  - configs/
  - "*.py"

rsync_exclude:
  - data/
  - "*.pyc"
  - __pycache__/
  - .git/
  - .venv/

# ============================================================
# HOST CONFIGURATION
# ============================================================

# SSH hostnames or aliases
host_address:
  local: ""
  myserver: "user@myserver.com"
  cluster: "user@login.cluster.edu"

# List of SSH hosts (non-SLURM)
ssh_hosts: [myserver]

# Remote project directory per host
remote_dir:
  myserver: /home/user/project
  cluster: /scratch/user/project

# Remote log directory per host (optional, defaults to remote_dir + log_dir)
remote_log_dir:
  myserver: /home/user/project/data
  cluster: /scratch/user/project/data

# ============================================================
# SLURM CONFIGURATION
# ============================================================

# SLURM job headers (use $gpus for dynamic GPU count)
remote_header:
  cluster: |
    #!/usr/bin/env bash
    #SBATCH --job-name=chester
    #SBATCH --partition=gpu
    #SBATCH --gpus=$gpus
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=72:00:00

# CUDA module to load
cuda_module:
  cluster: cuda/12.1

# Other modules to load
modules:
  cluster: [singularity]

# Singularity container path (optional)
simg_path:
  cluster: /path/to/container.sif

# Mount options for singularity
remote_mount_option:
  cluster: /usr/share/glvnd
```

---

## Execution Modes

Chester supports multiple execution backends:

### Local Execution

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='local',
    exp_prefix='experiment',
)
```

**Behavior:**
- Runs directly as a subprocess
- Logs saved to `{log_dir}/{sub_dir}/{exp_prefix}/{exp_name}/`
- No code syncing required

### SSH Hosts

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='myserver',  # Must be in ssh_hosts list
    exp_prefix='experiment',
    auto_pull=True,   # Automatically pull results when done
)
```

**Behavior:**
1. Syncs code to remote via rsync (first variant only)
2. Creates `ssh_launch.sh` script in log directory
3. Copies script to remote and executes with `nohup`
4. Saves process PID to `.chester_pid` for tracking
5. Creates `.done` marker file on successful completion
6. If `auto_pull=True`, spawns background poller to pull results

**Files created on remote:**
```
{remote_log_dir}/{exp_name}/
├── ssh_launch.sh      # The launch script
├── stdout.log         # Standard output
├── stderr.log         # Standard error
├── .chester_pid       # Process ID for tracking
└── .done              # Created on successful completion
```

### SLURM Clusters

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='cluster',  # Must have remote_header defined
    exp_prefix='experiment',
    use_gpu=True,
    use_singularity=True,
)
```

**Behavior:**
1. Syncs code to remote via rsync
2. Generates SLURM batch script with headers from config
3. Submits job via `sbatch`
4. Logs to `slurm.out` and `slurm.err`

### AWS EC2

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='ec2',
    exp_prefix='experiment',
)
```

**Behavior:**
- Uploads code to S3
- Launches EC2 instance with user-data script
- Periodically syncs results to S3
- Terminates instance on completion

Requires `config_ec2.py` with AWS credentials and settings.

---

## Auto-Pull System

Chester can automatically pull results from remote hosts when jobs complete.

### Enabling Auto-Pull

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='myserver',
    exp_prefix='experiment',
    auto_pull=True,
    auto_pull_interval=60,  # Poll every 60 seconds
)
```

### How It Works

1. **Job Tracking**: Chester saves the process PID to `.chester_pid` when launching
2. **Polling**: A background process polls remote hosts for job status
3. **Status Detection**: Determines job state based on `.done` marker and process state

### Job Status Detection

| `.done` exists | Process running | Status | Action |
|----------------|-----------------|--------|--------|
| Yes | No | `pulled` | Pull results (success) |
| Yes | Yes | `pulled` | Kill orphan processes, then pull |
| No | No | `failed` | Pull logs for debugging |
| No | Yes | `pending` | Keep polling |

### Orphan Process Cleanup

If a job creates `.done` but leaves processes running (e.g., GPU processes not properly cleaned up), Chester will:

1. Send `SIGTERM` to the process tree
2. Wait 5 seconds for graceful shutdown
3. Send `SIGKILL` if processes still running
4. Pull results

### Failed Job Handling

If a job crashes without creating `.done`:

1. Chester detects the process is no longer running
2. Pulls logs (`stdout.log`, `stderr.log`) for debugging
3. Marks job as `failed` in manifest

### Manifest File

Auto-pull state is tracked in a JSON manifest:

```
{log_dir}/.chester_manifests/{exp_prefix}_{mode}_{timestamp}.json
```

Example manifest entry:
```json
{
  "host": "myserver",
  "remote_log_dir": "/home/user/project/data/train/exp1",
  "local_log_dir": "/home/local/project/data/train/exp1",
  "exp_name": "exp1",
  "pid_file": "/home/user/project/data/train/exp1/.chester_pid",
  "status": "pulled",
  "submitted_at": "2024-01-15T10:30:00",
  "pulled_at": "2024-01-15T11:45:00"
}
```

### Manual Auto-Pull

Run the auto-pull poller manually:

```bash
# Continuous polling
python -m chester.auto_pull --manifest /path/to/manifest.json

# Single check
python -m chester.auto_pull --manifest /path/to/manifest.json --once

# Exclude large files when pulling
python -m chester.auto_pull --manifest /path/to/manifest.json --bare
```

---

## Parameter Sweeps

Use `VariantGenerator` to create parameter combinations:

```python
from chester.run_exp import VariantGenerator

vg = VariantGenerator()
vg.add('learning_rate', [0.001, 0.01, 0.1])
vg.add('batch_size', [32, 64])
vg.add('model', ['resnet18', 'resnet50'])

# Generates 3 x 2 x 2 = 12 variants
for v in vg.variants():
    print(v)  # {'learning_rate': 0.001, 'batch_size': 32, 'model': 'resnet18'}
```

### Dependent Parameters

Parameters can depend on other parameters:

```python
vg = VariantGenerator()
vg.add('model', ['small', 'large'])
vg.add('hidden_dim', lambda model: [128] if model == 'small' else [256, 512])

# Generates: small/128, large/256, large/512
```

### Hidden Parameters

Hide parameters from experiment naming:

```python
vg.add('seed', [1, 2, 3], hide=True)
```

---

## Hydra Integration

Chester supports Hydra-style command line overrides:

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='local',
    exp_prefix='experiment',
    hydra_enabled=True,
    hydra_flags={'multirun': True},
)
```

This generates commands like:
```bash
python script.py learning_rate=0.001 batch_size=32 --multirun
```

---

## Package Manager Support

### uv (Recommended)

```yaml
package_manager: uv
sync_on_launch: true
```

On remote execution:
1. Auto-installs `uv` if not present
2. Runs `uv sync` to install dependencies
3. Wraps python commands with `uv run`

### Conda/Mamba

```yaml
package_manager: conda
conda_env: myenv
conda_command: mamba  # or 'conda'
sync_on_launch: true
```

On remote execution:
1. Sources `~/.bashrc` for conda
2. Activates the specified environment
3. Optionally updates from `environment.yml`

---

## API Reference

### `run_experiment_lite()`

Main function for launching experiments.

```python
run_experiment_lite(
    # Required
    stub_method_call=None,     # Function to call with (variant, log_dir, exp_name)

    # Experiment naming
    exp_prefix="experiment",   # Prefix for experiment names
    exp_name=None,             # Override auto-generated name
    log_dir=None,              # Override log directory
    sub_dir='train',           # Subdirectory under log_dir

    # Execution
    mode="local",              # Execution mode
    python_command="python",   # Python command to use
    script=None,               # Script to run (default: chester worker)
    dry=False,                 # Print commands without executing

    # Variant
    variant=None,              # Parameter dictionary
    variations=[],             # Keys to include in exp_name

    # Remote execution
    use_gpu=False,             # Request GPU (SLURM)
    use_singularity=False,     # Use singularity container
    env={},                    # Environment variables

    # Auto-pull
    auto_pull=False,           # Enable automatic result pulling
    auto_pull_interval=60,     # Seconds between polls

    # Hydra
    hydra_enabled=False,       # Use Hydra command format
    hydra_flags=None,          # Hydra flags dict

    # Advanced
    use_cloudpickle=True,      # Serialize with cloudpickle
    sync_env=None,             # Override sync_on_launch config
)
```

### `VariantGenerator`

```python
vg = VariantGenerator()

# Add parameter with values
vg.add('param', [value1, value2, ...])

# Add dependent parameter
vg.add('param', lambda other_param: [values...])

# Add hidden parameter (not in exp name)
vg.add('param', [values], hide=True)

# Get all variants
variants = vg.variants()           # List of dicts
variants = vg.variants(randomized=True)  # Shuffled order

# Get number of variants
count = vg.size
```

---

## Troubleshooting

### Job stuck in "pending" forever

**Old behavior (pre-v0.2.1):** If a job crashed without creating `.done`, the auto-pull poller would poll forever.

**New behavior:** Chester tracks process PID and detects when the process dies. Failed jobs are marked as `failed` and logs are pulled for debugging.

### Orphan GPU processes

**Symptom:** Job completes but GPU memory is still in use.

**Solution:** Chester detects when `.done` exists but process is still running, and kills the process tree before pulling results.

### SSH connection issues

Ensure passwordless SSH is configured:
```bash
ssh-copy-id user@remote
```

### Rsync errors

Check that `rsync_include` and `rsync_exclude` are defined in `chester.yaml`. Chester no longer falls back to file-based patterns.

---

## License

MIT License
