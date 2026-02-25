# Migrating from Chester v1 to v2

This guide covers all breaking changes in Chester v2 and how to update your project.

## Overview of Changes

| What changed | v1 | v2 |
|---|---|---|
| Config file location | `chester.yaml` (project root) | `.chester/config.yaml` |
| Backend configuration | Flat dicts (`host_address`, `remote_dir`, ...) | Structured `backends:` section |
| SLURM header | Raw multi-line string with `$gpus` | Structured `slurm:` config with named fields |
| Environment setup | `prepare_commands` list in YAML + hardcoded uv/conda logic | Per-backend `prepare.sh` scripts |
| Package manager sync | `sync_on_launch`, `conda_env`, `conda_command` in YAML | Handled in your `prepare.sh` |
| Per-experiment SLURM tuning | Not supported | `slurm_overrides` parameter |
| SLURM job tracking | `.done` marker only (can't detect failures) | `sacct`-based tracking (detects OOM, timeout, crashes) |
| EC2 / Autobot modes | Supported | Removed (pin `chester-ml<1.0` if needed) |

## 1. Move Your Config File

**Before (v1):**
```
my_project/
  chester.yaml
  src/
  ...
```

**After (v2):**
```
my_project/
  .chester/
    config.yaml
    backends/
      greatlakes/
        prepare.sh
      ssh-server/
        prepare.sh
      local/
        prepare.sh
  src/
  ...
```

Chester v2 still finds `chester.yaml` for backward compatibility (with a deprecation warning), but you should move to the new location.

## 2. Rewrite Your Config

### Before (v1 `chester.yaml`)

```yaml
log_dir: data
hydra_config_path: configs
package_manager: uv
sync_on_launch: true

prepare_commands:
  - direnv allow .
  - eval "$(direnv export bash)"

rsync_include:
  - IsaacLabTactile/source
  - third_party/curobo/.git
  - .git/
  - .git/modules/

rsync_exclude:
  - IsaacLabTactile/_isaac_sim
  - .containers/
  - data
  - "*__pycache__*"
  - .venv

host_address:
  local: ""
  gl: "gl"
  armdual: "armdual"
  armlake: "armlake.local"

ssh_hosts: [armdual, armlake]

remote_dir:
  gl: /home/houhd/code/cotrain_dynamics
  armdual: /home/houhd/code/cotrain_dynamics
  armlake: /home/houhd/code/cotrain_dynamics

remote_log_dir:
  gl: /home/houhd/code/cotrain_dynamics/data
  armdual: /home/houhd/code/cotrain_dynamics/data
  armlake: /home/houhd/code/cotrain_dynamics/data

remote_header:
  gl: |
    #!/usr/bin/env bash
    #SBATCH --job-name=chester
    #SBATCH --nodes=1
    #SBATCH --partition=spgpu
    #SBATCH --gpus=$gpus
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=72:00:00

simg_path:
  gl: /home/houhd/code/cotrain_dynamics/.containers/full-latest.sif
cuda_module:
  gl: cuda/12.8.1
modules:
  gl: [singularity]
remote_mount_option:
  gl: /usr/share/glvnd
```

### After (v2 `.chester/config.yaml`)

```yaml
log_dir: data
package_manager: uv
hydra_config_path: configs

rsync_include:
  - IsaacLabTactile/source
  - third_party/curobo/.git
  - .git/
  - .git/modules/

rsync_exclude:
  - IsaacLabTactile/_isaac_sim
  - .containers/
  - data
  - "*__pycache__*"
  - .venv

backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh

  gl:
    type: slurm
    host: gl
    remote_dir: /home/houhd/code/cotrain_dynamics
    prepare: .chester/backends/greatlakes/prepare.sh
    modules: [singularity]
    cuda_module: cuda/12.8.1
    slurm:
      partition: spgpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
    singularity:
      image: /home/houhd/code/cotrain_dynamics/.containers/full-latest.sif
      mounts: [/usr/share/glvnd]
      gpu: true

  armdual:
    type: ssh
    host: armdual
    remote_dir: /home/houhd/code/cotrain_dynamics
    prepare: .chester/backends/ssh-server/prepare.sh

  armlake:
    type: ssh
    host: armlake.local
    remote_dir: /home/houhd/code/cotrain_dynamics
    prepare: .chester/backends/ssh-server/prepare.sh
```

**Key differences:**

- `host_address`, `remote_dir`, `remote_log_dir`, `remote_header`, `simg_path`, `cuda_module`, `modules`, `remote_mount_option` are **all gone**. Each backend now defines its own `host`, `remote_dir`, etc.
- `remote_log_dir` is computed automatically from `remote_dir` + `log_dir`, so you never specify it separately.
- `ssh_hosts` is gone. Any backend with `type: ssh` is an SSH host.
- `prepare_commands` is gone. Use per-backend `prepare.sh` scripts instead.

## 3. SLURM Header: Raw String to Structured Config

### Before (v1)

```yaml
remote_header:
  gl: |
    #!/usr/bin/env bash
    #SBATCH --job-name=chester
    #SBATCH --nodes=1
    #SBATCH --partition=spgpu
    #SBATCH --gpus=$gpus
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64G
    #SBATCH --time=72:00:00
```

The `$gpus` variable was substituted at runtime, which was fragile and hard to override per-experiment.

### After (v2)

```yaml
backends:
  gl:
    type: slurm
    slurm:
      partition: spgpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
```

Chester generates the `#SBATCH` header from these structured fields. The job name and output/error paths are set automatically per experiment.

**Available `slurm:` fields:**
- `partition` -- SLURM partition name
- `time` -- Wall time limit (quote it: `"72:00:00"`)
- `nodes` -- Number of nodes (default: 1)
- `gpus` -- Number of GPUs
- `cpus_per_gpu` -- CPUs per GPU
- `mem_per_gpu` -- Memory per GPU (e.g., `64G`)
- `ntasks_per_node` -- Tasks per node
- `extra_directives` -- List of additional `#SBATCH` flags (e.g., `["--exclusive", "--constraint=v100"]`)

## 4. prepare.sh Replaces Hardcoded Environment Setup

In v1, Chester hardcoded uv/conda setup logic (installing uv, setting `CUDA_HOME`, `NCCL_DEBUG`, `SETUPTOOLS_SCM_PRETEND_VERSION`, etc.) and ran `prepare_commands` from the YAML.

In v2, all of this moves to your `prepare.sh` scripts. This gives you full control and keeps Chester out of your environment setup.

### Example: `.chester/backends/greatlakes/prepare.sh`

```bash
#!/bin/bash
# Great Lakes SLURM backend - environment setup
# Nothing needed here: modules and singularity are handled by chester.
# Add custom env exports if your project needs them:
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
```

### Example: `.chester/backends/ssh-server/prepare.sh`

```bash
#!/bin/bash
# SSH backend - load project environment via direnv
direnv allow .
eval "$(direnv export bash)"
```

### Example: `.chester/backends/local/prepare.sh`

```bash
#!/bin/bash
# Local backend - load project environment via direnv
direnv allow .
eval "$(direnv export bash)"
```

**What chester still handles automatically:**
- `module load` for modules listed in the backend config
- `module load` for `cuda_module`
- Wrapping the python command with `uv run` when `package_manager: uv`
- Singularity container wrapping when `singularity:` is configured

**What you put in prepare.sh:**
- `direnv allow . && eval "$(direnv export bash)"` -- if your project uses direnv
- `export CUDA_HOME=...` -- custom CUDA setup
- `export NCCL_DEBUG=INFO` -- distributed training env vars
- `conda activate myenv` -- if using conda without chester's built-in support
- Any other environment setup your project needs

## 5. Package Manager Changes

### Before (v1)

```yaml
package_manager: uv
sync_on_launch: true
conda_env: myenv
conda_command: mamba
```

Chester hardcoded logic for uv installation, `uv sync`, conda activation, and `conda env update`.

### After (v2)

```yaml
package_manager: uv   # or: python, conda
```

That is all. The `package_manager` field now only controls how Chester wraps the python command:

| `package_manager` | Python command |
|---|---|
| `python` | `python train.py` |
| `uv` | `uv run python train.py` |
| `conda` | `python train.py` (assumes env is activated in prepare.sh) |

`sync_on_launch`, `conda_env`, and `conda_command` are removed. If you need `uv sync` or `conda activate`, put it in your `prepare.sh`.

## 6. New `slurm_overrides` Parameter

You can now override SLURM parameters per experiment without changing config:

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='gl',
    exp_prefix='big_run',
    slurm_overrides={
        'time': '120:00:00',
        'gpus': 4,
        'mem_per_gpu': '128G',
    },
)
```

This merges with the base `slurm:` config from `.chester/config.yaml`. Only the specified fields are overridden.

## 7. Deprecated Modes

The following execution modes are removed in v2:

- **`ec2`** -- AWS EC2 with S3 sync
- **`autobot`** -- Custom GPU scheduler
- **`local_singularity`** / **`singularity`** -- Use `type: local` with a `singularity:` block instead

If you still need EC2 or autobot support, pin your dependency:

```
chester-ml<0.5
```

For singularity on local, configure it in the backend:

```yaml
backends:
  local_sing:
    type: local
    singularity:
      image: /path/to/container.sif
      gpu: true
```

## 8. Launcher Code Changes

### Before (v1)

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='gl',           # matched to host_address keys
    exp_prefix='train',
    use_gpu=True,
    n_gpu=2,
)
```

### After (v2)

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='gl',           # now matches backends keys
    exp_prefix='train',
    slurm_overrides={'gpus': 2},  # GPU count via overrides
)
```

The `use_gpu` and `n_gpu` parameters are no longer needed -- GPU configuration lives in `slurm:` config and can be overridden per-experiment.

## Quick Migration Checklist

1. Create `.chester/` directory in your project root
2. Move `chester.yaml` to `.chester/config.yaml`
3. Replace flat dicts (`host_address`, `remote_dir`, etc.) with `backends:` section
4. Replace `remote_header` raw strings with structured `slurm:` config
5. Move `prepare_commands` to per-backend `prepare.sh` files
6. Remove `sync_on_launch`, `conda_env`, `conda_command` (put in prepare.sh if needed)
7. Remove `ssh_hosts` list (SSH backends are auto-detected by `type: ssh`)
8. Update launcher code to use `slurm_overrides` instead of `use_gpu`/`n_gpu`
9. Test with `mode='local'` first, then remote backends
