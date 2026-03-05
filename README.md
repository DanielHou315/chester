# Chester

Chester (`chester-ml` on PyPI) is a Python experiment launcher for ML workflows. It serializes your experiment function and variant parameters, then dispatches them to a configured execution backend — local subprocess, SSH host, or SLURM cluster — with optional Singularity container support and automatic result synchronisation.

## Installation

```bash
pip install chester-ml
# or
uv add chester-ml
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Layout](#project-layout)
- [Configuration Reference](#configuration-reference)
- [Backend Types](#backend-types)
  - [Local](#local-backend)
  - [SSH](#ssh-backend)
  - [SLURM](#slurm-backend)
- [Singularity](#singularity)
- [prepare.sh](#preparesh)
- [Parameter Sweeps](#parameter-sweeps)
- [Auto-Pull](#auto-pull)
- [Hydra Integration](#hydra-integration)
- [Git Snapshot](#git-snapshot)
- [run_experiment_lite Reference](#run_experiment_lite-reference)
- [Contributing a New Backend](#contributing-a-new-backend)

---

## Quick Start

**1. Create `.chester/config.yaml` in your project root:**

```yaml
log_dir: data
package_manager: uv

backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh

  myserver:
    type: ssh
    host: myserver             # SSH alias from ~/.ssh/config
    remote_dir: /home/user/myproject
    prepare: .chester/backends/myserver/prepare.sh

  mycluster:
    type: slurm
    host: mycluster
    remote_dir: /home/user/myproject
    prepare: .chester/backends/mycluster/prepare.sh
    slurm:
      partition: gpu
      time: "24:00:00"
      nodes: 1
      gpus: 1
```

**2. Write a launcher:**

```python
from chester.run_exp import run_experiment_lite, VariantGenerator

def run_task(variant, log_dir, exp_name):
    print(f"lr={variant['lr']}, batch={variant['batch_size']}")

vg = VariantGenerator()
vg.add('lr', [1e-3, 1e-4])
vg.add('batch_size', [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=run_task,
        variant=v,
        mode='local',        # or 'myserver', 'mycluster'
        exp_prefix='sweep',
    )
```

**3. Run:**

```bash
python launcher.py
```

---

## Project Layout

Chester looks for its config file in this order:

1. Path in `CHESTER_CONFIG_PATH` environment variable
2. `.chester/config.yaml` — searched upward from the current directory, stopping at the `.git` root
3. `chester.yaml` at the project root (deprecated, still works with a warning)

The recommended layout is:

```
myproject/
├── .chester/
│   ├── config.yaml                    # Main config
│   └── backends/
│       ├── local/
│       │   └── prepare.sh             # Local environment setup
│       ├── mycluster/
│       │   └── prepare.sh             # Cluster setup (modules, paths)
│       └── myserver/
│           └── prepare.sh             # SSH server setup
├── launchers/
│   └── launch_sweep.py
└── src/
    └── ...
```

---

## Configuration Reference

`.chester/config.yaml` has two sections: project-level settings and the `backends` map.

```yaml
# ── Project-level ─────────────────────────────────────────────────────────────

# Directory for local experiment logs (relative to project root)
log_dir: data

# Package manager used to run python on remote hosts: 'uv' or 'python'
# When 'uv', remote commands are wrapped as "uv run python ..."
package_manager: uv

# Path to Hydra config directory (relative to project root), used when
# hydra_enabled=True in run_experiment_lite
hydra_config_path: configs

# Files/dirs to include when rsyncing code to remote hosts.
# Patterns are passed to rsync --include.
rsync_include:
  - src/
  - configs/
  - "*.py"

# Files/dirs to exclude from rsync.
# Patterns are passed to rsync --exclude.
rsync_exclude:
  - data/
  - "*.pyc"
  - __pycache__/
  - .venv/
  - wandb/

# ── Backends ──────────────────────────────────────────────────────────────────

backends:
  <name>:
    type: local | ssh | slurm   # required
    ...                          # type-specific keys (see below)
```

---

## Backend Types

### Local Backend

Runs the experiment as a local subprocess. No code syncing is performed.

```yaml
backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh   # optional
```

**What happens at launch:**

1. Chester serializes your function and variant into a bash script.
2. The script is executed as a subprocess.
3. Logs are written to `{log_dir}/train/{exp_prefix}/{exp_name}/`.

**With Singularity** (see [Singularity](#singularity) for full options):

```yaml
backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh
    singularity:
      image: .containers/myimage.sif
      gpu: true
      mounts:
        - /usr/share/glvnd
```

---

### SSH Backend

Runs the experiment on a remote host via SSH and `nohup`.

```yaml
backends:
  myserver:
    type: ssh
    host: myserver                        # SSH alias in ~/.ssh/config
    remote_dir: /home/user/myproject      # Project root on the remote
    prepare: .chester/backends/myserver/prepare.sh   # optional
```

**What happens at launch:**

1. Code is rsynced to the remote (first variant only, subsequent variants in the batch reuse the sync).
2. A bash script is generated and SCPed to the remote log directory.
3. The script is executed with `nohup bash` over SSH; the remote PID is saved to `.chester_pid`.
4. On successful completion, the script writes a `.done` marker file.

**Files created on the remote:**

```
{remote_dir}/{log_dir}/train/{exp_prefix}/{exp_name}/
├── ssh_launch.sh        # the generated launch script
├── stdout.log
├── stderr.log
├── chester_xtrace.log   # bash -x trace (separate from stderr)
├── .chester_pid         # PID for auto-pull tracking
└── .done                # written on success
```

**Passwordless SSH is required.** Add your key with `ssh-copy-id myserver`.

**With Singularity:**

```yaml
backends:
  myserver:
    type: ssh
    host: myserver
    remote_dir: /home/user/myproject
    prepare: .chester/backends/myserver/prepare.sh
    singularity:
      image: /home/user/containers/myimage.sif
      gpu: true
      mounts:
        - /usr/share/glvnd
        - /data/datasets:/data/datasets
```

---

### SLURM Backend

Submits a SLURM batch job via `sbatch`.

```yaml
backends:
  mycluster:
    type: slurm
    host: mycluster                        # SSH alias for the login node
    remote_dir: /home/user/myproject
    prepare: .chester/backends/mycluster/prepare.sh
    modules: [singularity]                 # modules to load before running
    cuda_module: cuda/12.1                 # additional CUDA module to load
    slurm:
      partition: gpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
      ntasks_per_node: 1                   # optional
      email_end: user@university.edu       # email on END and FAIL
      email_begin: user@university.edu     # email on BEGIN (optional)
      extra_directives:                    # arbitrary extra #SBATCH lines
        - "--account=myaccount"
        - "--qos=high"
```

All `slurm:` fields are optional. Only set what your cluster needs.

**What happens at launch:**

1. Code is rsynced to the remote login node.
2. A SLURM batch script is generated and SCPed to the remote log directory.
3. `sbatch` is invoked over SSH; the job ID is saved to `.chester_slurm_job_id`.
4. On successful completion, the script writes a `.done` marker.

**Generated `#SBATCH` directives:**

| Config key | `#SBATCH` directive |
|---|---|
| `partition` | `--partition=` |
| `time` | `--time=` |
| `nodes` | `--nodes=` |
| `gpus` | `--gpus=` |
| `cpus_per_gpu` | `--cpus-per-gpu=` |
| `mem_per_gpu` | `--mem-per-gpu=` |
| `ntasks_per_node` | `--ntasks-per-node=` |
| `email_end` | `--mail-user=`, `--mail-type=END,FAIL` |
| `email_begin` | `--mail-user=`, `--mail-type=BEGIN` (added to `email_end` if both set) |
| `extra_directives` | verbatim |

Per-job SBATCH directives (output file, error file, job name) are added automatically from the experiment name and log directory.

**Per-experiment SLURM overrides:**

Override any SLURM field for a specific experiment without changing the config:

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='mycluster',
    exp_prefix='bigrun',
    slurm_overrides={'time': '120:00:00', 'mem_per_gpu': '128G'},
)
```

**With Singularity:**

```yaml
backends:
  mycluster:
    type: slurm
    host: mycluster
    remote_dir: /home/user/myproject
    prepare: .chester/backends/mycluster/prepare.sh
    modules: [singularity]
    cuda_module: cuda/12.1
    slurm:
      partition: gpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 64G
    singularity:
      image: /home/user/containers/myimage.sif
      gpu: true
      fakeroot: true
      mounts:
        - /usr/share/glvnd
        - raw_datasets:/data/raw_datasets    # relative: resolved from remote_dir
```

---

## Singularity

All three backend types support Singularity. When a `singularity:` block is present in a backend, every command runs inside the container via `singularity exec`.

### Full Singularity Options

```yaml
singularity:
  image: /path/to/image.sif       # required; relative paths resolved from project root
  gpu: true                        # add --nv flag (NVIDIA GPU passthrough)
  fakeroot: true                   # add --fakeroot (default: true)
  mounts:                          # bind mounts; can be /src or /src:/dst
    - /usr/share/glvnd
    - raw_datasets:/data/raw        # relative src resolved from project root
  workdir: /workspace               # --pwd inside the container
  prepare: .chester/container/prepare.sh  # sourced inside the container before python
  writable_tmpfs: false            # --writable-tmpfs (in-memory overlay, ephemeral)
  overlay: .containers/overlay.ext3       # persistent ext3 overlay image
  overlay_size: 10240              # size in MB when creating overlay (default 10 GB)
  enabled: true                    # set to false to disable without removing config
```

### Persistent Overlay

A persistent ext3 overlay lets you install packages into the container without rebuilding the `.sif`. Chester creates the overlay file on first use:

```yaml
singularity:
  image: /path/to/base.sif
  overlay: .containers/myoverlay.ext3
  overlay_size: 20480              # 20 GB
  fakeroot: true
```

The overlay is created lazily with `singularity overlay create --size {size} {overlay}` if the file does not already exist.

### Per-Container prepare.sh

To run setup commands inside the container (e.g., activating a conda env that lives in the overlay), use `singularity.prepare`:

```yaml
singularity:
  image: /path/to/image.sif
  overlay: .containers/overlay.ext3
  workdir: /workspace
  prepare: .chester/container/in_container_prepare.sh
```

This script is sourced inside the container before the python command runs.

---

## prepare.sh

Each backend can specify a `prepare` script that is sourced before the experiment runs. Use it to set up environment variables, activate virtual environments, or do any other host-specific setup.

```yaml
backends:
  mycluster:
    type: slurm
    ...
    prepare: .chester/backends/mycluster/prepare.sh
```

For remote backends (SSH, SLURM), the prepare path is resolved relative to `remote_dir` on the remote host. The file must already be present on the remote (it is rsynced along with the rest of the project code).

Example `prepare.sh`:

```bash
#!/usr/bin/env bash
# .chester/backends/mycluster/prepare.sh

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export NCCL_DEBUG=INFO

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
```

---

## Parameter Sweeps

`VariantGenerator` creates the cross-product of all parameter lists.

```python
from chester.run_exp import VariantGenerator

vg = VariantGenerator()
vg.add('lr', [1e-3, 1e-4, 1e-5])
vg.add('batch_size', [32, 64])
# → 6 variants

for v in vg.variants():
    run_experiment_lite(stub_method_call=run_task, variant=v, ...)
```

### Dependent Parameters

A parameter value can depend on another parameter:

```python
vg.add('model', ['small', 'large'])
vg.add('hidden_dim', lambda model: [128] if model == 'small' else [256, 512])
# → small/128, large/256, large/512
```

### Hidden Parameters

Hide a parameter from the auto-generated experiment name:

```python
vg.add('seed', [1, 2, 3], hide=True)
```

### Randomised Order

```python
for v in vg.variants(randomized=True):
    ...
```

---

## Auto-Pull

Auto-pull polls remote hosts for job completion and syncs results back locally.

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='myserver',
    exp_prefix='sweep',
    auto_pull=True,
    auto_pull_interval=60,          # seconds between polls (default 60)
    extra_pull_dirs=['data/outputs'],  # additional directories to sync
)
```

### How It Works

Chester tracks jobs in a JSON manifest at `{log_dir}/.chester_manifests/{exp_prefix}_{mode}_{timestamp}.json`. A background poller process reads the manifest and checks each job's status.

**For SSH jobs** — status is determined by:

| `.done` marker | Process alive | Status | Action |
|---|---|---|---|
| Yes | No | `done` | Pull results |
| Yes | Yes | `done_orphans` | Kill process tree, pull |
| No | No | `failed` | Pull logs |
| No | Yes | `running` | Keep polling |

**For SLURM jobs** — status is also checked via `sacct`, which detects scheduler-level failures (OOM, timeout, preemption, etc.) that would not be caught by PID tracking alone.

### Extra Pull Directories

Pull directories beyond the experiment log:

```python
extra_pull_dirs=[
    'data/generated_datasets',   # relative: resolved from project root / remote_dir
    '/mnt/shared/checkpoints',   # absolute: same path on both local and remote
]
```

### Manual Polling

Run the poller independently:

```bash
# Continuous polling
python -m chester.auto_pull --manifest path/to/manifest.json

# Single check
python -m chester.auto_pull --manifest path/to/manifest.json --once

# Skip large files when pulling
python -m chester.auto_pull --manifest path/to/manifest.json --bare
```

---

## Hydra Integration

Chester can format experiment parameters as Hydra overrides instead of `--key value` CLI args.

```python
run_experiment_lite(
    stub_method_call=run_task,
    variant=v,
    mode='mycluster',
    exp_prefix='hydra_sweep',
    hydra_enabled=True,
    hydra_flags={'multirun': True},
)
```

Set the path to your Hydra config directory in `.chester/config.yaml`:

```yaml
hydra_config_path: configs    # relative to project root
```

Generated command format:

```bash
python script.py lr=0.001 batch_size=32 hydra.run.dir=/path/to/log_dir --multirun
```

---

## Git Snapshot

Chester automatically saves the git state of your repository to the experiment log directory before launching (controlled by `git_snapshot=True`, which is the default). This records everything needed to reproduce the exact code state.

**Files written to `{log_dir}/`:**

| File | Contents |
|---|---|
| `git_info.json` | commit hash, branch, dirty flag, timestamp, submodule status, untracked symlinks |
| `git_diff.patch` | full unified diff of all staged/unstaged changes; untracked file names as comments; dirty submodule diffs in labelled sections |

`git_diff.patch` is only written when there are uncommitted changes (in the parent repo or in any submodule).

**Submodule tracking:** Each submodule entry in `git_info.json` includes `dirty: bool` and `untracked_files: list`. If a submodule has uncommitted changes, its diff is appended to `git_diff.patch` under a `# === Submodule: <path> ===` header.

**Recovery:**

```bash
git checkout <commit>         # from git_info.json
git apply git_diff.patch      # restore parent-repo changes

# For each submodule section in git_diff.patch:
cd <submodule_path>
git checkout <submodule_hash>   # from git_info.json submodules[*].hash
# apply the section manually
```

Disable snapshots for a run:

```python
run_experiment_lite(..., git_snapshot=False)
```

---

## `run_experiment_lite` Reference

```python
run_experiment_lite(
    # ── Required ──────────────────────────────────────────────────────────────
    stub_method_call=None,      # callable: fn(variant, log_dir, exp_name)

    # ── Naming ────────────────────────────────────────────────────────────────
    exp_prefix="experiment",    # prefix for log directory and experiment name
    exp_name=None,              # override the auto-generated name
    log_dir=None,               # override the log directory
    sub_dir='train',            # subdirectory under log_dir

    # ── Execution ─────────────────────────────────────────────────────────────
    mode="local",               # backend name from .chester/config.yaml
    python_command="python",    # base python command (wrapped by package_manager)
    script=None,                # script to run (default: chester worker)
    dry=False,                  # print commands without running them
    variant=None,               # parameter dict for this run
    env=None,                   # extra environment variables {KEY: VALUE}

    # ── SLURM ─────────────────────────────────────────────────────────────────
    slurm_overrides=None,       # dict of SlurmConfig fields to override per-run

    # ── Singularity ───────────────────────────────────────────────────────────
    use_singularity=None,       # True/False to override backend config; None = use config

    # ── Auto-pull ─────────────────────────────────────────────────────────────
    auto_pull=False,            # enable background result polling
    auto_pull_interval=60,      # seconds between polls
    extra_pull_dirs=None,       # additional directories to sync on completion

    # ── Hydra ─────────────────────────────────────────────────────────────────
    hydra_enabled=False,        # use Hydra override format for args
    hydra_flags=None,           # dict of Hydra flags (e.g. {'multirun': True})

    # ── Git ───────────────────────────────────────────────────────────────────
    git_snapshot=True,          # save git state to log dir

    # ── Misc ──────────────────────────────────────────────────────────────────
    confirm=False,              # prompt before remote submission
    fresh=False,                # if True, scan and delete existing exp_prefix dirs
                                # before launching; always prompts for confirmation
    sync_env=None,              # override package_manager sync behaviour
)
```

---

## Contributing a New Backend

Chester backends are classes that implement two abstract methods: `generate_script()` and `submit()`. To add a new backend:

### 1. Create the backend class

```python
# src/chester/backends/mybackend.py
from typing import Any, Dict, Optional
from .base import Backend, BackendConfig


class MyBackend(Backend):
    """My custom execution backend."""

    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return the full bash script as a string."""
        params = task["params"]
        log_dir = params["log_dir"]

        lines = ["#!/usr/bin/env bash", "set -euo pipefail"]

        # Source the per-backend prepare.sh (if configured)
        for cmd in self.get_prepare_commands():
            lines.append(cmd)

        # Build the inner python command
        inner_cmd = self.build_python_command(
            params, script, python_command, env, hydra_enabled, hydra_flags
        )

        # Optionally wrap with Singularity
        if self.config.singularity:
            lines += self.get_overlay_setup_commands()
            sing_cmds = self.get_singularity_prepare_commands() + [inner_cmd]
            lines.append(self.wrap_with_singularity(sing_cmds))
        else:
            lines.append(inner_cmd)

        # Write .done marker so auto-pull knows the job succeeded
        lines.append(f"touch {log_dir}/.done")

        return "\n".join(lines)

    def submit(self, task: Dict[str, Any], script: str, dry: bool = False) -> Any:
        """Submit the script for execution. Return a job identifier."""
        if dry:
            print(script)
            return None

        # ... your submission logic here ...
        return job_id
```

### 2. Register the backend type

In `src/chester/backends/__init__.py`, add your type to the `create_backend` factory:

```python
def create_backend(config: BackendConfig, project_config: Dict[str, Any]) -> Backend:
    if config.type == "local":
        from .local import LocalBackend
        return LocalBackend(config, project_config)
    elif config.type == "ssh":
        from .ssh import SSHBackend
        return SSHBackend(config, project_config)
    elif config.type == "slurm":
        from .slurm import SlurmBackend
        return SlurmBackend(config, project_config)
    elif config.type == "mybackend":          # ← add this
        from .mybackend import MyBackend
        return MyBackend(config, project_config)
    else:
        raise ValueError(f"Unknown backend type: {config.type}")
```

Also add `"mybackend"` to `VALID_BACKEND_TYPES` in `base.py`:

```python
VALID_BACKEND_TYPES = ("local", "ssh", "slurm", "mybackend")
```

### 3. Configure it

```yaml
backends:
  myjob:
    type: mybackend
    host: myhost
    remote_dir: /home/user/project
    prepare: .chester/backends/myjob/prepare.sh
```

### Base class helpers

| Method | What it does |
|---|---|
| `get_prepare_commands()` | Returns `["source /path/to/prepare.sh"]` if configured |
| `get_python_command(base)` | Wraps base with `uv run` if `package_manager=uv` and no singularity |
| `build_python_command(params, script, ...)` | Builds the full `python script.py --key val ...` string |
| `get_overlay_setup_commands()` | Returns shell lines to create the ext3 overlay if missing |
| `get_singularity_prepare_commands()` | Returns source command for the in-container prepare script |
| `wrap_with_singularity(commands)` | Joins commands and wraps with `singularity exec ...` |
| `_get_remote_prepare_commands()` | Like `get_prepare_commands()` but resolves path relative to `remote_dir` |

### Writing tests

Tests live in `tests/`. Use `tmp_path` for isolated directories and mock SSH/SLURM commands with `unittest.mock.patch`. See `tests/test_backend_local.py` and `tests/test_backend_slurm.py` for examples.

```bash
uv run pytest tests/ -v
```

---

## License

MIT License
