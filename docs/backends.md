# Chester Backends

Chester supports three backend types for running experiments: **local**, **SSH**, and **SLURM**. Each backend handles launching, environment setup, and log management differently.

## Local Backend

```yaml
backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh  # optional
    singularity:                                  # optional
      enabled: false
```

### Execution Flow

1. Chester serializes your function and variant into a bash script
2. The script is executed as a subprocess (or `subprocess.Popen` for async)
3. Logs are written to `{log_dir}/{sub_dir}/{exp_prefix}/{exp_name}/`

### Configuration Options

**`launch_with_subprocess`** (bool)
- `True` (default): Launch with `Popen` for async execution; up to `max_num_processes` can run concurrently
- `False`: Run the experiment in-process via `run_hydra_command` (requires `hydra_enabled=True`)

**`wait_processes`** (bool)
- `True`: Run subprocesses sequentially (one finishes before the next starts)
- `False` (default): Run concurrently up to `max_num_processes`

**`max_num_processes`** (int)
- Maximum concurrent subprocess launches
- Typical usage: `detect_local_gpus()` returns GPU IDs from `$CUDA_VISIBLE_DEVICES` or `nvidia-smi`; use this to cap parallelism

**`detect_local_gpus()`** (function)
- Automatically detects available GPUs from `$CUDA_VISIBLE_DEVICES` environment variable or `nvidia-smi` output
- Returns list of GPU IDs
- Useful for setting `max_num_processes` dynamically

### Graceful Shutdown

Pressing **Ctrl+C** locally terminates all running subprocesses and stops launching new ones.

### Example Usage

```python
from chester.run_exp import run_experiment_lite, detect_local_gpus

num_gpus = len(detect_local_gpus())

for v in vg.variants():
    run_experiment_lite(
        function=train_model,
        variant=v,
        mode="local",
        launch_with_subprocess=True,
        max_num_processes=num_gpus,
    )
```

---

## SSH Backend

```yaml
backends:
  myserver:
    type: ssh
    host: myserver                        # SSH alias from ~/.ssh/config
    remote_dir: /home/user/myproject      # project root on remote
    prepare: .chester/backends/myserver/prepare.sh
    batch_gpu: 2                          # optional: batch mode (see below)
    singularity:                          # optional
      enabled: true
      mounts: [...]
```

### Execution Flow

1. Code is rsynced to the remote (once per launcher run; subsequent variants reuse the sync)
2. A bash script is generated and SCPed to the remote log directory
3. Script is executed with `nohup bash` over SSH

### Remote Directory Structure

```
{remote_dir}/{log_dir}/{sub_dir}/{exp_prefix}/{exp_name}/
├── chester_run.sh       # generated script (uploaded via SCP)
├── output.log           # combined stdout and stderr
└── chester_xtrace.log   # bash -x trace
```

A local copy of the script is also saved as `ssh_launch.sh` in the local batch directory for debugging.

### Requirements

- **Passwordless SSH** is required
- Add your SSH key to the remote: `ssh-copy-id myserver`
- SSH alias must be defined in `~/.ssh/config`

### Standard Mode

In standard mode, each call to `run_experiment_lite()` immediately launches a job on the remote via SSH. Jobs run sequentially unless you manually manage concurrency.

### Batch-GPU Mode

`batch_gpu: N` enables batch mode for multi-GPU SSH servers:

- Accumulates all scripts from the variant loop in memory
- On `flush_backend(mode)`, uploads all scripts to a shared queue on the remote and spawns N GPU worker processes (one per detected GPU, capped at `N`); each worker atomically pops scripts from the queue using `flock` until the queue is empty
- GPU count is auto-detected (via `$CUDA_VISIBLE_DEVICES` or `nvidia-smi` on the remote), capped at `N`; `batch_gpu` alone is sufficient as a fallback if detection fails
- **Must call `flush_backend(mode)` after the variant loop** when using batch mode

```python
from chester.run_exp import run_experiment_lite, flush_backend

for v in vg.variants():
    run_experiment_lite(..., mode="myserver")

flush_backend("myserver")   # fires all accumulated jobs
```

### GPU Device Pinning

Use `cuda_visible_devices` to pin specific device IDs on the remote:

```yaml
backends:
  myserver:
    type: ssh
    host: myserver
    remote_dir: /home/user/myproject
    batch_gpu: 2
    cuda_visible_devices: "0,1"  # only use GPUs 0 and 1
```

### Monitoring and Ctrl+C Behavior

Pressing **Ctrl+C** locally stops monitoring and printing output, but the remote job **continues running**. Use SSH to connect to the remote and manually kill the process if needed.

### Example Usage

```python
from chester.run_exp import run_experiment_lite, flush_backend

for v in vg.variants():
    run_experiment_lite(
        function=train_model,
        variant=v,
        mode="myserver",
    )

flush_backend("myserver")
```

---

## SLURM Backend

```yaml
backends:
  mycluster:
    type: slurm
    host: mycluster                        # SSH alias for the login node
    remote_dir: /home/user/myproject
    prepare: .chester/backends/mycluster/prepare.sh
    modules: [singularity, gcc/14.1.0]    # modules to load
    cuda_module: cuda/12.1                 # loaded after modules
    slurm:
      partition: spgpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
      ntasks_per_node: 1
      email_end: user@university.edu       # email on END and FAIL
      email_begin: user@university.edu     # email on BEGIN (optional, adds to email_end)
      # Any unknown key becomes #SBATCH --key=value:
      account: myaccount                   # → #SBATCH --account=myaccount
      qos: high                            # → #SBATCH --qos=high
      constraint: h100                     # → #SBATCH --constraint=h100
      gpu_cmode: shared                    # → #SBATCH --gpu_cmode=shared
```

### Execution Flow

1. Code is rsynced to the remote login node
2. A SLURM batch script is generated and SCPed to the remote log directory
3. `sbatch` is invoked over SSH

### SLURM Configuration Mapping

The `slurm:` section maps to `#SBATCH` directives:

| Config Key        | `#SBATCH` Directive    | Description                      |
|-------------------|------------------------|----------------------------------|
| `partition`       | `--partition=`         | Partition/queue to submit to     |
| `time`            | `--time=`              | Wall-clock time limit (HH:MM:SS) |
| `nodes`           | `--nodes=`             | Number of nodes                  |
| `gpus`            | `--gpus=`              | Total GPUs requested             |
| `cpus_per_gpu`    | `--cpus-per-gpu=`      | CPUs per GPU                     |
| `mem_per_gpu`     | `--mem-per-gpu=`       | Memory per GPU                   |
| `ntasks_per_node` | `--ntasks-per-node=`   | Tasks per node                   |
| `email_end`       | `--mail-user=`, `--mail-type=END,FAIL` | Email on job end/failure |
| `email_begin`     | `--mail-user=`, `--mail-type=BEGIN` | Email on job start (merged with email_end if both set) |
| *(any other key)* | `--key=value`          | Passed verbatim as an `#SBATCH` directive |

**Auto-added directives:** `-o` (stdout), `-e` (stderr), and `--job-name` are added automatically per job.

**Arbitrary directives:** Any key in the `slurm:` block that isn't one of the named fields above is forwarded as `#SBATCH --key=value`. Boolean `true` emits `#SBATCH --key` (no value); `false` omits the line.

### Module Management

**`modules`** (list)
- Modules to load with `module load` before execution
- Loaded in order

**`cuda_module`** (str)
- CUDA module to load *after* the `modules` list
- Common values: `cuda/12.1`, `cuda/12.8.1`

### Email Notifications

- **`email_end`**: Receives emails on job END and FAIL states
- **`email_begin`** (optional): Receives emails on BEGIN state
- If both are set, Chester merges them into a single `--mail-user` directive with combined types

### Per-Experiment SLURM Overrides

Override any `slurm:` field for a single experiment without changing the config:

```python
run_experiment_lite(
    function=train_model,
    variant=v,
    mode="mycluster",
    slurm_overrides={"time": "120:00:00", "mem_per_gpu": "128G", "gpus": 4},
)
```

Overrides are merged with the backend config; unspecified keys use backend defaults.

### Monitoring and Ctrl+C Behavior

Pressing **Ctrl+C** locally stops the monitoring loop, but the SLURM job **continues running** on the cluster. Use `squeue` or `scontrol` to check job status.

### Example Usage

```python
from chester.run_exp import run_experiment_lite

for v in vg.variants():
    run_experiment_lite(
        function=train_model,
        variant=v,
        mode="mycluster",
        slurm_overrides={"time": "48:00:00"} if v["large_dataset"] else {},
    )
```

---

## Prepare Scripts

Each backend can source a `prepare.sh` script before the experiment runs. Use it for:
- Environment variable exports
- Module loads
- Python path setup
- Dependency installation (remote backends only)

### Path Resolution

For **remote backends** (SSH, SLURM), the path is relative to `remote_dir` on the remote. The file is rsynced with the rest of the project.

For **local**, the path is relative to the project root.

### Example SLURM prepare.sh

```bash
#!/usr/bin/env bash
set -e

module load cuda/12.8.1
module load gcc/14.1.0

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
source .envrc
```

### Example SSH prepare.sh

```bash
#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
set +u; source .envrc; set -u

# Ensure uv is synced on remote
uv sync
```

---

## Extra Sync Directories

Sync additional directories to the remote before job submission. Useful for datasets or pre-trained checkpoints not in the main rsync:

```python
run_experiment_lite(
    function=train_model,
    variant=v,
    mode="myserver",
    extra_sync_dirs=["raw_datasets/my_dataset", "/abs/path/to/ckpts"],
)
```

### Path Handling

- **Relative paths**: Synced from `{project_root}/path` → `{remote_dir}/path`
- **Absolute paths**: Synced as-is; if local user differs from remote user, `/home/<local_user>/` is substituted with `/home/<remote_user>/`
- These syncs **bypass `rsync_exclude` entirely**
- No-op for local mode

### Sync Timing

Sync runs **once per launcher run** (on the first variant); subsequent variants reuse the synced directories. This avoids redundant network transfers when launching many variants.

---

## Singularity Containers

All backends support optional Singularity container execution:

```yaml
backends:
  local:
    type: local
    singularity:
      enabled: false  # or true
```

For remote backends, mount paths and other Singularity options can be specified in the `singularity:` section. Consult Chester's Singularity integration docs for details.
