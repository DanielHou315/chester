# Configuration Reference

This guide explains how to configure Chester via `.chester/config.yaml`.

## Config Search Order

Chester searches for configuration in this order:

1. **`CHESTER_CONFIG_PATH` environment variable** — if set, use this explicit path
2. **`.chester/config.yaml`** — searched upward from current working directory until `.git` root is found
3. **`chester.yaml` at project root** — deprecated; shows a warning if used

For most projects, place `.chester/config.yaml` in the `.chester/` directory at your project root.

## Recommended Project Layout

```
myproject/
├── .chester/
│   ├── config.yaml
│   └── backends/
│       ├── local/
│       │   └── prepare.sh
│       ├── mycluster/
│       │   └── prepare.sh
│       └── myserver/
│           └── prepare.sh
├── launchers/
│   └── launch_sweep.py
├── src/
└── .git/
```

The `.chester/backends/` directory is optional but recommended for organizing per-backend scripts.

## Project-Level Settings

These keys appear at the top level of `config.yaml`:

### `log_dir`
**Type:** string (path)  
**Default:** `data`  
**Description:** Directory where local logs are written. Relative to project root. Created if it does not exist.

```yaml
log_dir: data
```

### `package_manager`
**Type:** string  
**Default:** `python`  
**Options:** `uv`, `python`, `conda`  
**Description:** How Chester invokes Python scripts:
- `uv`: wraps commands as `uv run python script.py <args>`
- `python`: runs bare Python (assumes your environment is active)
- `conda`: runs bare Python (assumes a conda environment is activated in `prepare.sh`)

```yaml
package_manager: uv
```

### `hydra_config_path`
**Type:** string (path)  
**Default:** not set  
**Description:** Path to Hydra configuration directory, relative to project root. Used when `hydra_enabled=True` is passed at launch time. If set, Chester passes `--config-path=<hydra_config_path>` to the launcher script.

```yaml
hydra_config_path: configs
```

### `rsync_include`
**Type:** list of strings  
**Default:** `[]`  
**Description:** rsync `--include` patterns for code sync to remote hosts. Patterns are applied in order; later excludes can override earlier includes.

```yaml
rsync_include:
  - "*.py"
  - "configs/**"
  - "src/**"
```

### `rsync_exclude`
**Type:** list of strings  
**Default:** `[]`  
**Description:** rsync `--exclude` patterns for code sync to remote hosts. Applied after includes to filter out files and directories.

```yaml
rsync_exclude:
  - "__pycache__"
  - "*.pyc"
  - ".git"
  - "data/"
  - ".venv/"
```

## Global Singularity Block (Optional)

The top-level `singularity:` block defines shared container defaults inherited by all backends. Each backend can override or extend these settings. If present, this block applies to any backend that does not explicitly set `singularity.enabled: false`.

```yaml
singularity:
  image: .containers/base.sif
  workdir: /workspace
  gpu: true
  enabled: false
  prepare: .chester/backends/singularity/prepare.sh
  mounts:
    - src:/workspace/src
    - ~/.cache:/root/.cache
```

**Key fields:**

- **`image`** (string): Path to Singularity image (`.sif` file). Can be relative (to project root) or absolute.
- **`workdir`** (string): Working directory inside the container.
- **`gpu`** (boolean): Whether to pass `--nv` (NVIDIA GPU support) to Singularity.
- **`enabled`** (boolean):
  - `false` (default): Singularity is opt-in. Activate per-run with `use_singularity=True` in launcher or `--singularity` CLI flag.
  - `true`: Singularity is always enabled for this backend (unless backend overrides it).
- **`prepare`** (string): Path to a `prepare.sh` script to run inside the container before the job. Relative to project root.
- **`mounts`** (list): Mount points in format `host_path:container_path`. Supports `~` for home directory expansion.

When a backend's `singularity:` block is present, it is merged with and overrides the global block. See [Singularity](singularity.md) for detailed container configuration.

## Backends Map

The `backends:` top-level key maps backend names to their configurations. Each backend specifies how jobs are launched (local, SSH, SLURM). Full documentation is in [Backends](backends.md); here is a minimal example:

```yaml
backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh

  myserver:
    type: ssh
    host: user@myserver.edu
    workdir: /home/user/workdir
    prepare: .chester/backends/myserver/prepare.sh

  gpu_cluster:
    type: slurm
    host: user@login.cluster.edu
    workdir: /scratch/user/workdir
    prepare: .chester/backends/gpu_cluster/prepare.sh
    slurm:
      partition: gpu
      gres: gpu:1
```

## YAML Anchors and Aliases

The config file is plain YAML, so you can use `&anchor` and `*alias` to share values (e.g., mount lists) across backends:

```yaml
_mounts: &mounts_list
  - src:/workspace/src
  - ~/.cache:/root/.cache

singularity:
  mounts: *mounts_list

backends:
  backend1:
    singularity:
      mounts: *mounts_list
  backend2:
    singularity:
      mounts: *mounts_list
```

## Full Annotated Example

```yaml
# Project-level settings
log_dir: experiments
package_manager: uv
hydra_config_path: conf

# rsync patterns for remote syncing
rsync_include:
  - "*.py"
  - "*.yaml"
  - "src/**"
  - "conf/**"
  - "launchers/**"

rsync_exclude:
  - "__pycache__"
  - "*.pyc"
  - ".git"
  - ".venv"
  - "data/"
  - ".wandb/"
  - "*.sif"

# Global singularity defaults (inherited by all backends)
singularity:
  image: .containers/base.sif
  workdir: /workspace
  gpu: true
  enabled: false  # opt-in; use --singularity or use_singularity=True to enable per-run
  prepare: .chester/backends/singularity/prepare.sh
  mounts:
    - src:/workspace/src
    - conf:/workspace/conf
    - ~/.cache:/root/.cache

# Backend configurations
backends:
  # Local execution (development)
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh

  # SSH remote host with GPU batch mode
  myserver:
    type: ssh
    host: user@myserver.edu
    workdir: /home/user/chester-jobs
    extra_sync_dirs:
      - data/datasets
    prepare: .chester/backends/myserver/prepare.sh
    ssh:
      batch_gpu: true
      num_gpus: 4
    singularity:
      enabled: false

  # SLURM cluster with container support
  gpu_cluster:
    type: slurm
    host: user@login.cluster.edu
    workdir: /scratch/user/chester-jobs
    prepare: .chester/backends/gpu_cluster/prepare.sh
    singularity:
      enabled: true
      image: /cluster/containers/ml-base.sif
      prepare: .chester/backends/gpu_cluster/singularity-prepare.sh
    slurm:
      partition: gpu
      account: myaccount
      time: "00:30:00"
      nodes: 1
      ntasks: 1
      cpus_per_task: 8
      gres: gpu:1

  # Another SLURM partition with per-job overrides
  cpu_cluster:
    type: slurm
    host: user@login.cluster.edu
    workdir: /scratch/user/chester-jobs
    prepare: .chester/backends/cpu_cluster/prepare.sh
    slurm:
      partition: cpu
      account: myaccount
      time: "02:00:00"
      nodes: 1
      ntasks: 4
```

## Advanced Features

### Per-Experiment SLURM Overrides

For SLURM backends, you can override cluster settings per job using `slurm_overrides` in your launcher:

```python
launcher = SlurmLauncher(
    cfg=cfg,
    slurm_overrides={'time': '01:00:00', 'gres': 'gpu:2'}
)
```

This is useful for varying time limits or GPU counts without modifying the base config.

### Extra Sync Directories

Use `extra_sync_dirs` on remote backends to sync additional project paths:

```yaml
backends:
  myserver:
    type: ssh
    host: user@myserver.edu
    workdir: /home/user/jobs
    extra_sync_dirs:
      - data/datasets
      - models/checkpoints
```

This syncs `data/datasets/` and `models/checkpoints/` to the remote alongside the main code.

### Backend-Specific prepare.sh

Each backend can have a `prepare.sh` script (relative to project root) that runs before the job:

```yaml
backends:
  local:
    prepare: .chester/backends/local/prepare.sh
```

Use this to install dependencies, activate virtual environments, set environment variables, or configure cluster-specific settings.

## See Also

- [Backends](backends.md) — detailed backend types and options
- [Singularity](singularity.md) — container configuration and overlays
- [Git Snapshot](git-snapshot.md) — reproducibility tracking
