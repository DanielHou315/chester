# Chester User Guide

Chester is a Python experiment launcher for ML workflows. You define your training function and parameter sweep, and chester handles running experiments on local machines, SSH hosts, or SLURM clusters — with automatic code sync, result pulling, and experiment tracking.

## Quick Start

### 1. Install

```bash
pip install chester-ml
# or
uv add chester-ml
```

### 2. Create Config

Create `.chester/config.yaml` in your project root:

```yaml
log_dir: data
package_manager: uv    # or: python, conda

backends:
  local:
    type: local
```

### 3. Write a Training Script

```python
# launch.py
from chester.run_exp import run_experiment_lite, VariantGenerator

def train(variant, log_dir, exp_name):
    import torch
    import torch.nn as nn

    lr = variant["learning_rate"]
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ... your training loop ...

    torch.save(model.state_dict(), f"{log_dir}/checkpoint.pt")

vg = VariantGenerator()
vg.add("learning_rate", [0.001, 0.01, 0.1])
vg.add("batch_size", [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train,
        variant=v,
        mode="local",
        exp_prefix="my_experiment",
    )
```

### 4. Run

```bash
uv run python launch.py
```

Chester creates a directory tree under `data/train/my_experiment/` with one subdirectory per variant, each containing your checkpoint and a `metrics.json`.

---

## Configuration

Chester looks for config in this order:

1. `CHESTER_CONFIG_PATH` environment variable
2. `.chester/config.yaml` in current or parent directories
3. `chester.yaml` (deprecated, still works)

### Full Config Reference

```yaml
# .chester/config.yaml

# Where experiment logs are saved (relative to project root)
log_dir: data

# How python is invoked: python | uv | conda
#   python -> "python train.py"
#   uv     -> "uv run python train.py"
#   conda  -> "python train.py" (assumes env activated in prepare.sh)
package_manager: uv

# Hydra config directory (relative to project root)
hydra_config_path: configs

# Code sync patterns (for rsync to remote hosts)
rsync_include:
  - .git/
  - src/

rsync_exclude:
  - data/
  - .venv/
  - "*__pycache__*"
  - "*.so"
  - wandb/

# Backend definitions
backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh    # optional

  my_ssh_host:
    type: ssh
    host: myhost.example.com          # SSH hostname
    remote_dir: /home/user/project    # Project path on remote
    prepare: .chester/backends/ssh/prepare.sh  # optional

  my_cluster:
    type: slurm
    host: cluster.example.com         # SSH hostname for the login node
    remote_dir: /home/user/project    # Project path on cluster
    prepare: .chester/backends/slurm/prepare.sh  # optional
    modules: [singularity]            # module load commands
    cuda_module: cuda/12.8.1          # separate for clarity
    slurm:
      partition: gpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
      ntasks_per_node: 1
      extra_directives:               # arbitrary #SBATCH flags
        - "--exclusive"
        - "--constraint=v100"
    singularity:
      image: /path/to/container.sif
      mounts: [/data, /scratch]
      gpu: true                       # adds --nv flag
```

---

## Backends

### Local

Runs experiments as local subprocesses.

```yaml
backends:
  local:
    type: local
```

```python
run_experiment_lite(
    stub_method_call=train,
    variant=v,
    mode="local",
    exp_prefix="my_exp",
)
```

### SSH

Copies your code to a remote host via rsync, then runs the experiment via `nohup` over SSH. Results are tracked with a `.done` marker and `.chester_pid` file.

```yaml
backends:
  armdual:
    type: ssh
    host: armdual                              # SSH host (must be in ~/.ssh/config)
    remote_dir: /home/user/code/my_project     # where code lives on remote
    prepare: .chester/backends/ssh/prepare.sh   # env setup script
```

```python
run_experiment_lite(
    stub_method_call=train,
    variant=v,
    mode="armdual",        # matches the backend name
    exp_prefix="my_exp",
    auto_pull=True,        # automatically pull results when done
)
```

**What happens under the hood:**
1. `rsync` syncs your code to `remote_dir`
2. Chester generates a bash script with `set -e`, `cd remote_dir`, `source prepare.sh`, then your python command
3. Script is `scp`'d to the remote and run via `nohup bash script.sh > output.log 2>&1 &`
4. PID is saved to `.chester_pid` for tracking
5. On success, `.done` is created
6. If `auto_pull=True`, a background poller checks for `.done` and pulls results back

### SLURM

Generates an `sbatch` script and submits it on a SLURM cluster.

```yaml
backends:
  greatlakes:
    type: slurm
    host: gl                                    # login node hostname
    remote_dir: /home/user/code/my_project
    prepare: .chester/backends/gl/prepare.sh
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
      image: /home/user/.containers/ml.sif
      mounts: [/usr/share/glvnd]
      gpu: true
```

```python
run_experiment_lite(
    stub_method_call=train,
    variant=v,
    mode="greatlakes",
    exp_prefix="my_exp",
)
```

**Generated script structure:**
```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=spgpu
#SBATCH --time=72:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=64G
#SBATCH -o /path/to/logs/slurm.out
#SBATCH -e /path/to/logs/slurm.err
#SBATCH --job-name=my_exp
set -x
set -u
set -e
srun hostname
cd /home/user/code/my_project
module load singularity
module load cuda/12.8.1
singularity exec -B /usr/share/glvnd --nv /home/user/.containers/ml.sif \
  /bin/bash -c 'source .chester/backends/gl/prepare.sh && \
  uv run python -m chester.run_exp_worker --learning_rate 0.001 ... && \
  touch /path/to/logs/.done'
```

---

## prepare.sh Scripts

Each backend can have a `prepare.sh` script that runs before your training. This is where environment setup goes — things that used to be hardcoded in chester.

**Local:**
```bash
#!/bin/bash
# .chester/backends/local/prepare.sh
direnv allow .
eval "$(direnv export bash)"
```

**SSH:**
```bash
#!/bin/bash
# .chester/backends/ssh/prepare.sh
direnv allow .
eval "$(direnv export bash)"
```

**SLURM (Great Lakes example):**
```bash
#!/bin/bash
# .chester/backends/gl/prepare.sh
# Modules and singularity are handled by chester automatically.
# Add any extra env setup here:
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
```

Chester automatically handles:
- `module load` for each entry in `modules` and `cuda_module`
- Singularity wrapping when `singularity:` is configured
- Python command wrapping (`uv run python` when `package_manager: uv`)

---

## Parameter Sweeps with VariantGenerator

`VariantGenerator` creates all combinations of parameter values:

```python
from chester.run_exp import VariantGenerator

vg = VariantGenerator()
vg.add("learning_rate", [0.001, 0.01, 0.1])
vg.add("batch_size", [32, 64])
# -> 6 variants: 3 x 2

for v in vg.variants():
    print(v["learning_rate"], v["batch_size"])
```

### Dependencies Between Parameters

A parameter can depend on other parameters using a lambda:

```python
vg = VariantGenerator()
vg.add("batch_size", [32, 64, 128])
vg.add("hidden_dim", lambda batch_size: [batch_size * 2])
# batch_size=32  -> hidden_dim=64
# batch_size=64  -> hidden_dim=128
# batch_size=128 -> hidden_dim=256
```

### Hidden Parameters

Parameters can be hidden from the experiment name:

```python
vg.add("seed", [0, 1, 2], hide=True)
```

### How Parameters Reach Your Function

Chester serializes your function with `cloudpickle` and passes parameters as a `variant` dict:

```python
def train(variant, log_dir, exp_name):
    lr = variant["learning_rate"]       # from VariantGenerator
    bs = variant["batch_size"]          # from VariantGenerator
    # log_dir and exp_name are set by chester
```

On the command line, parameters become `--key value` flags passed to `chester.run_exp_worker`, which deserializes and calls your function.

---

## Per-Experiment SLURM Overrides

Override SLURM parameters for individual experiments without changing config:

```python
vg = VariantGenerator()
vg.add("model_size", ["small", "large"])

for v in vg.variants():
    # Large model needs more resources
    overrides = {}
    if v["model_size"] == "large":
        overrides = {"gpus": 4, "time": "120:00:00", "mem_per_gpu": "128G"}

    run_experiment_lite(
        stub_method_call=train,
        variant=v,
        mode="greatlakes",
        exp_prefix="scaling_exp",
        slurm_overrides=overrides,
    )
```

**Available override fields:**
`partition`, `time`, `nodes`, `gpus`, `cpus_per_gpu`, `mem_per_gpu`, `ntasks_per_node`, `extra_directives`

These merge with the base `slurm:` config -- only specified fields change.

---

## Hydra Integration

If your training script uses [Hydra](https://hydra.cc/) for configuration, chester can convert variant parameters to Hydra overrides instead of `--key value` CLI args.

### When to Use

- **Without Hydra (default):** Your function receives a `variant` dict. Parameters are passed as `--key value`.
- **With Hydra:** Your function receives a Hydra `DictConfig`. Parameters become Hydra overrides (`key=value`).

### Setup

1. Set `hydra_config_path` in your config (relative to project root):

```yaml
hydra_config_path: configs
```

2. Create a Hydra config at `configs/config.yaml`:

```yaml
# configs/config.yaml
learning_rate: 0.001
batch_size: 32
model:
  hidden_dim: 128
  num_layers: 4
```

3. Launch with `hydra_enabled=True`:

```python
def train(cfg):
    # cfg is a Hydra DictConfig, not a variant dict
    lr = cfg.learning_rate
    hidden = cfg.model.hidden_dim
    # ...

vg = VariantGenerator()
vg.add("learning_rate", [0.001, 0.01])
vg.add("model.hidden_dim", [128, 256])  # dot notation for nested keys

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train,
        variant=v,
        mode="local",
        exp_prefix="hydra_exp",
        hydra_enabled=True,
    )
```

**Generated command (Hydra mode):**
```
python main.py learning_rate=0.001 model.hidden_dim=128 hydra.run.dir=/path/to/logs
```

**Generated command (non-Hydra mode):**
```
python -m chester.run_exp_worker --learning_rate 0.001 --model_hidden_dim 128 --log_dir /path/to/logs
```

### Hydra vs VariantGenerator

| Feature | VariantGenerator | Hydra |
|---|---|---|
| Parameter sweeps | `vg.add("lr", [0.001, 0.01])` | Same (chester converts to Hydra overrides) |
| Nested params | `variant["model"]["hidden_dim"]` | `cfg.model.hidden_dim` |
| Config files | chester.yaml only | YAML configs in `configs/` directory |
| Type checking | None | Hydra's structured configs |
| Multirun | Manual loop | `hydra_flags={"multirun": True}` |

You can use both together -- `VariantGenerator` creates the sweep, and `hydra_enabled=True` routes parameters through Hydra's config system.

---

## Automatic Result Pulling

For remote backends (SSH, SLURM), chester can automatically poll for job completion and pull results back:

```python
run_experiment_lite(
    stub_method_call=train,
    variant=v,
    mode="armdual",
    exp_prefix="my_exp",
    auto_pull=True,              # enable auto-pull
    auto_pull_interval=60,       # check every 60 seconds
    extra_pull_dirs=["weights"], # pull additional directories
)
```

**How it works:**
1. Chester writes a manifest file listing all submitted jobs
2. A background poller process checks each job's status via SSH
3. When a job finishes (`.done` marker appears), results are pulled via `rsync`
4. Jobs that die without `.done` are marked as failed and logs are pulled for debugging

**Status detection — SSH backends:**
```
.done exists + process dead  -> done (pull results)
.done exists + process alive -> orphan (kill, then pull)
no .done + process dead      -> failed (pull logs)
no .done + process alive     -> running (keep polling)
```

**Status detection — SLURM backends:**

Chester captures the SLURM job ID from `sbatch` output and uses `sacct` to detect failures. This means SLURM jobs that crash (OOM, timeout, bad node) are detected immediately instead of polling forever.

```
.done exists                           -> done (pull results)
no .done + sacct FAILED/TIMEOUT/OOM    -> failed (pull logs)
no .done + sacct RUNNING/PENDING       -> running (keep polling)
no .done + sacct COMPLETED             -> running (brief grace for .done)
```

---

## Git Snapshot

By default, chester saves your git state before running experiments:

```
data/train/my_experiment/0_my_exp/
  git_info.json      # commit hash, branch, dirty flag, timestamp
  git_diff.patch     # uncommitted changes (if any)
  checkpoint.pt      # your outputs
  metrics.json
```

Disable with `git_snapshot=False`.

---

## Singularity Containers

Any backend type (local, SSH, SLURM) can use Singularity containers:

```yaml
backends:
  local_containerized:
    type: local
    singularity:
      image: /path/to/container.sif
      mounts: [/data, /scratch]
      gpu: true

  cluster_containerized:
    type: slurm
    host: gl
    remote_dir: /home/user/project
    slurm:
      partition: gpu
      gpus: 1
    singularity:
      image: /home/user/.containers/ml.sif
      mounts: [/usr/share/glvnd, /data]
      gpu: true
```

When singularity is configured, chester wraps the inner commands (prepare + python + .done marker) with:
```bash
singularity exec -B /data -B /scratch --nv /path/to/container.sif \
  /bin/bash -c 'source prepare.sh && python train.py ... && touch .done'
```

You can dynamically disable singularity per-experiment:
```python
run_experiment_lite(
    ...,
    use_singularity=False,  # skip container for this run
)
```

---

## Complete Example: Multi-Host Training

```yaml
# .chester/config.yaml
log_dir: data
package_manager: uv
hydra_config_path: configs

rsync_include:
  - src/
  - configs/
  - .git/

rsync_exclude:
  - data/
  - .venv/
  - "*__pycache__*"
  - wandb/

backends:
  local:
    type: local

  armdual:
    type: ssh
    host: armdual
    remote_dir: /home/user/code/my_project
    prepare: .chester/backends/ssh/prepare.sh

  greatlakes:
    type: slurm
    host: gl
    remote_dir: /home/user/code/my_project
    prepare: .chester/backends/gl/prepare.sh
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
      image: /home/user/.containers/ml.sif
      mounts: [/usr/share/glvnd]
      gpu: true
```

```python
# launch.py
from chester.run_exp import run_experiment_lite, VariantGenerator

def train(variant, log_dir, exp_name):
    import torch
    import torch.nn as nn

    lr = variant["learning_rate"]
    hidden = variant["hidden_dim"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = nn.Sequential(
        nn.Linear(784, hidden), nn.ReLU(), nn.Linear(hidden, 10)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # ... training loop ...
    torch.save(model.state_dict(), f"{log_dir}/model.pt")


vg = VariantGenerator()
vg.add("learning_rate", [0.001, 0.01])
vg.add("hidden_dim", [64, 256])
vg.add("seed", [0, 1, 2], hide=True)

# Quick local debug
for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train, variant=v,
        mode="local", exp_prefix="debug", dry=True,
    )

# Full run on cluster with 4 GPUs
for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train, variant=v,
        mode="greatlakes", exp_prefix="full_run",
        slurm_overrides={"gpus": 4, "time": "120:00:00"},
        auto_pull=True,
    )

# Or on an SSH host
for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train, variant=v,
        mode="armdual", exp_prefix="ssh_run",
        auto_pull=True, auto_pull_interval=30,
    )
```

---

## run_experiment_lite Reference

```python
run_experiment_lite(
    # Required
    stub_method_call=train,          # your training function
    variant=v,                       # parameter dict from VariantGenerator

    # Backend selection
    mode="local",                    # backend name from config

    # Experiment naming and logging
    exp_prefix="my_exp",             # name prefix for experiment dirs
    exp_name=None,                   # explicit name (auto-generated if None)
    log_dir=None,                    # explicit log dir (auto-generated if None)
    sub_dir="train",                 # subdirectory under log_dir

    # Execution control
    dry=False,                       # print commands without running
    python_command="python",         # base python command
    script=None,                     # custom script (default: chester.run_exp_worker)
    env=None,                        # extra env vars: {"KEY": "VALUE"}

    # Local execution
    launch_with_subprocess=True,     # run via subprocess (vs in-process for Hydra debug)
    wait_subprocess=True,            # wait for subprocess to finish
    max_num_processes=10,            # max concurrent local processes

    # SLURM
    slurm_overrides=None,            # per-experiment SLURM param overrides
    use_singularity=None,            # True/False/None (None = use backend default)

    # Hydra
    hydra_enabled=False,             # use Hydra command format
    hydra_flags=None,                # extra Hydra flags

    # Remote result tracking
    auto_pull=False,                 # pull results when jobs finish
    auto_pull_interval=60,           # poll interval in seconds
    extra_pull_dirs=None,            # extra dirs to pull: ["weights", "logs"]

    # Reproducibility
    git_snapshot=True,               # save git commit + diff to log dir

    # Variant naming
    variations=None,                 # list of param keys to include in exp name
    print_command=True,              # print the command before running
)
```
