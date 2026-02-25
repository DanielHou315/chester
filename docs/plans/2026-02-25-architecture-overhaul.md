# Chester Architecture Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace chester's monolithic, hardcoded execution system with a clean backend abstraction supporting local/SSH/SLURM (each with optional singularity), per-backend `prepare.sh` scripts in `.chester/`, configurable SLURM parameters, and python/conda/uv package managers. Deprecate EC2 and Autobot.

**Architecture:** Three backend types (local, ssh, slurm) × optional singularity wrapping = 6 execution modes. Each backend is configured via YAML in `.chester/config.yaml` with optional per-backend `prepare.sh` scripts. SLURM parameters are configurable per-backend with per-experiment overrides. All project-specific hardcoded values (NCCL settings, CUDA_HOME, ninja install, etc.) move to user-controlled `prepare.sh` scripts.

**Tech Stack:** Python 3.8+, PyYAML, cloudpickle, shlex (stdlib), subprocess, dataclasses, ABC

---

## Design Overview

### New `.chester/` Directory Structure

```
project/
├── .chester/
│   ├── config.yaml                  # Main chester config (replaces chester.yaml at project root)
│   └── backends/
│       ├── greatlakes/
│       │   └── prepare.sh           # Environment setup for Great Lakes cluster
│       ├── my-ssh-server/
│       │   └── prepare.sh           # Environment setup for SSH server
│       └── local/
│           └── prepare.sh           # Optional local env setup
├── chester.yaml                     # DEPRECATED: still loaded for backward compat with warning
└── src/...
```

### New `config.yaml` Format

```yaml
# .chester/config.yaml

# Project paths
project_path: /home/user/my_project  # Auto-detected from config file location if omitted
log_dir: data                         # Relative to project_path

# Package manager: "python", "conda", or "uv"
# Controls how chester wraps the python command:
#   python → "python"
#   conda  → "python" (after conda activate in prepare.sh)
#   uv     → "uv run python"
package_manager: uv

# Code sync patterns (for remote backends)
rsync_include: []
rsync_exclude: []

# Hydra integration
hydra_config_path: configs

# Backend definitions
backends:
  local:
    type: local
    # prepare: .chester/backends/local/prepare.sh  # optional

  greatlakes:
    type: slurm
    host: gl                                        # SSH hostname/alias
    remote_dir: /home/user/my_project
    prepare: .chester/backends/greatlakes/prepare.sh
    modules:                                        # Module system loads
      - singularity
    cuda_module: cuda/12.1.1
    slurm:                                          # SLURM defaults for this backend
      partition: spgpu
      time: "72:00:00"
      nodes: 1
      gpus: 1
      cpus_per_gpu: 4
      mem_per_gpu: 80G
      extra_directives:                             # Any additional #SBATCH lines
        - "--gpu_cmode=shared"
    singularity:                                    # Optional singularity wrapping
      image: /path/to/container.sif
      mounts:
        - /usr/share/glvnd
      gpu: true                                     # Adds --nv flag
      # build_script: .chester/singularity/build.sh # Optional: script to build/update the image

  my-server:
    type: ssh
    host: myserver.example.com
    remote_dir: /home/user/project
    prepare: .chester/backends/my-server/prepare.sh
    singularity:                                    # SSH backends can also use singularity
      image: /opt/containers/ml.sif
      mounts:
        - /data:/data
        - /scratch:/scratch
      gpu: true

  my-server-no-container:
    type: ssh
    host: myserver.example.com
    remote_dir: /home/user/project
    prepare: .chester/backends/my-server/prepare.sh
    # No singularity section → runs natively
```

### Example `prepare.sh`

```bash
#!/usr/bin/env bash
# .chester/backends/greatlakes/prepare.sh
# This script runs on the remote host BEFORE the experiment command.
# It should set up the Python environment, export variables, etc.

# CUDA setup
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))
echo "[chester] CUDA_HOME=$CUDA_HOME"

# Package manager sync (uv)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv &> /dev/null; then
    echo "[chester] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync

# NCCL settings for distributed training
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME=^docker0,lo

# Project-specific build tools
export MAX_JOBS=16
uv tool install ninja 2>/dev/null || true
```

### New Python API

```python
from chester.run_exp import run_experiment_lite, VariantGenerator

vg = VariantGenerator()
vg.add('learning_rate', [0.001, 0.01])
vg.add('batch_size', [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=train,
        variant=v,
        mode='greatlakes',              # Backend name from config
        exp_prefix='my_experiment',
        # Per-experiment SLURM overrides (only for slurm backends)
        slurm_overrides={
            'time': '24:00:00',
            'gpus': 2,
            'partition': 'gpu',
            'mem_per_gpu': '40G',
        },
        # Override singularity (True/False/None=use backend default)
        use_singularity=None,
        auto_pull=True,
    )
```

### New Module Structure

```
src/chester/
├── __init__.py
├── config.py              # REWRITE: loads .chester/config.yaml, validates backends
├── backends/
│   ├── __init__.py         # Backend registry, create_backend() factory
│   ├── base.py             # Backend ABC + BackendConfig dataclass
│   ├── local.py            # LocalBackend
│   ├── ssh.py              # SSHBackend
│   └── slurm.py            # SlurmBackend
├── singularity.py          # SingularityWrapper (wraps any backend's commands)
├── run_exp.py              # REWRITE: uses Backend.submit(), keeps VariantGenerator
├── run_exp_worker.py       # CLEANUP: remove dead rllab code
├── auto_pull.py            # MINOR: use subprocess instead of os.system
├── hydra_utils.py          # KEEP: minor cleanup
├── utils.py                # NEW: shared utilities (shell quoting, etc.)
├── logger.py               # KEEP as-is
├── utils_logger.py         # KEEP as-is
│
│ # DEPRECATED (keep with deprecation warnings, remove in next major):
├── config_ec2.py
├── utils_s3.py
├── setup_ec2_for_chester.py
├── pull_s3_result.py
├── pull_result.py
├── add_variants.py
├── availability_test.py
└── scheduler/              # ENTIRE DIRECTORY DEPRECATED
    ├── remote_scheduler.py
    ├── remote_slurm_launcher.py
    ├── kill_all_jobs.py
    └── list_jobs.py
```

---

## Appendix A: Hardcoded Values That Move to User `prepare.sh`

Every value listed below is currently hardcoded in chester source code. After this overhaul, they become the user's responsibility via their `prepare.sh` scripts. We MUST document these in migration docs so users know what to put in their scripts.

### Environment Variables (from `slurm.py`)

| Current Location | Value | Purpose |
|---|---|---|
| slurm.py:30 | `export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"` | uv/cargo binary paths |
| slurm.py:33-37 | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | Auto-install uv |
| slurm.py:41 | `export CUDA_HOME=$(dirname $(dirname $(which nvcc ...)))` | CUDA detection |
| slurm.py:45 | `export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.0` | curobo build hack |
| slurm.py:47 | `export MAX_JOBS=16` | Parallel CUDA compilation |
| slurm.py:49 | `uv tool install ninja` | Build tool installation |
| slurm.py:51-52 | `uv sync` | Environment sync |
| slurm.py:62 | `source ~/.bashrc` | Conda PATH setup |
| slurm.py:64 | `conda activate {env}` | Conda env activation |
| slurm.py:245 | `export NCCL_DEBUG=INFO` | NCCL debug logging |
| slurm.py:246 | `export PYTHONFAULTHANDLER=1` | Python fault handler |
| slurm.py:247 | `export NCCL_SOCKET_IFNAME=^docker0,lo` | Network interface exclusion |
| slurm.py:250 | `export EGL_GPU=$SLURM_JOB_GRES` | EGL GPU mapping |
| slurm.py:314 | `export PYTHONPATH="$PWD:${PYTHONPATH:-}"` | PYTHONPATH setup |

### Commands (from `run_exp.py`)

| Current Location | Value | Purpose |
|---|---|---|
| run_exp.py:724 | `. ./prepare.sh` (hardcoded in SLURM launcher SSH) | Legacy prepare script |
| run_exp.py:812 | `. ./prepare.sh` (hardcoded in autobot scheduler SSH) | Legacy prepare script |

### Config Defaults Removed

| Current Location | Value | Purpose |
|---|---|---|
| config.py:88 | `package_manager: 'uv'` | Default pkg manager → now explicit |
| config.py:91 | `sync_on_launch: True` | Auto-sync → now in prepare.sh |
| config.py:96 | `prepare_commands: []` | Inline commands → now prepare.sh file |

---

## Appendix B: Deprecated Modules & Features

### EC2 Mode (DEPRECATED)

Files to deprecate with warnings:
- `config_ec2.py` — 134 lines of hardcoded AWS config (AMI IDs, S3 paths, Docker images, security groups)
- `utils_s3.py` — 435 lines (broken `ec2.q()` call, rllab-era Docker/MuJoCo references)
- `setup_ec2_for_chester.py` — EC2 setup utility
- `pull_s3_result.py` — S3 result pulling

Key hardcoded values that would be lost (users who need EC2 should pin old version):
- AMI IDs: `ami-83f26195` (us-east-1), `ami-0ec385d5f98faacc3` (us-east-2), etc.
- S3 paths: `s3://chester-softgym/rllab/experiments`, `s3://chester-softgym/rllab/code`
- Docker images: `dementrock/rllab3-shared-gpu`
- Instance types: `p2.xlarge` (GPU), `c4.4xlarge` (CPU)
- Spot price: `$2.00`
- IAM profile: `rllab`
- Security groups: `rllab-sg`, region-specific SG IDs
- Code sync ignores: 30+ patterns (`.git`, `data/autobot`, etc.)
- MuJoCo key path: `~/.mujoco`

### Autobot Mode (DEPRECATED)

Files to deprecate:
- `scheduler/remote_scheduler.py` — 171 lines with hardcoded autobot node names
- `scheduler/kill_all_jobs.py`
- `scheduler/list_jobs.py`
- `scheduler/__init__.py`

Key hardcoded values that would be lost:
- Node names: `autobot-0-9`, `autobot-0-11`, `autobot-0-13`, ..., `autobot-1-1`, `autobot-1-6`
- Per-node GPU limits: `[:3]` for some nodes, `[:4]` for others
- Check interval: `120` seconds
- GPU state directory structure (pickle files in `GPU_STATE_DIR`)
- Queue directory structure: `CHESTER_QUEUE_DIR/queues/*`
- Custom header format: `#CHESTERNODE`, `#CHESTEROUT`, `#CHESTERERR`, `#CHESTERSCRIPT`

### Config Keys Removed

| Key | Replacement |
|---|---|
| `host_address` | `backends.<name>.host` |
| `ssh_hosts` | `backends.<name>.type: ssh` |
| `remote_dir` | `backends.<name>.remote_dir` |
| `remote_log_dir` | Auto-computed from `remote_dir` + relative path |
| `remote_header` | `backends.<name>.slurm.*` (structured) |
| `simg_path` | `backends.<name>.singularity.image` |
| `cuda_module` | `backends.<name>.cuda_module` |
| `modules` | `backends.<name>.modules` |
| `remote_mount_option` | `backends.<name>.singularity.mounts` |
| `autobot_nodelist` | DEPRECATED |
| `gpu_state_dir` | DEPRECATED |
| `chester_queue_dir` | DEPRECATED |
| `chester_scheduler_log_dir` | DEPRECATED |
| `prepare_commands` | `backends.<name>.prepare` (file path) |
| `sync_on_launch` | User's `prepare.sh` responsibility |
| `conda_env` | User's `prepare.sh` responsibility |
| `conda_command` | User's `prepare.sh` responsibility |

---

## Appendix C: Internal Constants (Stay Hardcoded)

These are internal chester mechanics that should NOT become user-configurable:

| Value | Location | Reason |
|---|---|---|
| `.done` marker file | slurm.py, auto_pull.py | Internal protocol between job and poller |
| `.chester_pid` file | run_exp.py, auto_pull.py | Internal PID tracking |
| `.chester_manifests/` dir | run_exp.py | Internal manifest storage |
| `set -x`, `set -u`, `set -e` | slurm.py | Script safety — always enabled |
| `#!/usr/bin/env bash` | slurm.py | Script shebang |
| `rsync -avzh --delete` | run_exp.py | Code sync flags (reasonable defaults) |
| `sbatch` command | slurm.py | SLURM submission (it's SLURM) |
| `module load` syntax | slurm.py | Standard module system syntax |
| `srun hostname` | slurm.py | SLURM diagnostics |
| `slurm.out`, `slurm.err` | run_exp.py | Standard SLURM log names |
| `stdout.log`, `stderr.log` | run_exp.py | Standard SSH log names |
| SSH/SCP commands | run_exp.py, auto_pull.py | Fundamental remote tools |

---

## Task Breakdown

### Task 1: Shared Utilities Module

**Files:**
- Create: `src/chester/utils.py`
- Modify: `src/chester/slurm.py` (remove duplicates)
- Modify: `src/chester/run_exp.py` (remove duplicates)
- Test: `tests/test_utils.py`

**Step 1: Write tests for shared utilities**

```python
# tests/test_utils.py
import pytest
from chester.utils import shellquote, to_param_val


def test_shellquote_empty_string():
    assert shellquote("") == "''"


def test_shellquote_safe_string():
    assert shellquote("hello") == "hello"


def test_shellquote_unsafe_string():
    # shlex.quote wraps in single quotes
    assert shellquote("hello world") == "'hello world'"


def test_shellquote_single_quotes():
    result = shellquote("it's")
    assert "it" in result and "s" in result  # properly escaped


def test_to_param_val_none():
    assert to_param_val(None) == ""


def test_to_param_val_string():
    assert to_param_val("hello") == "hello"


def test_to_param_val_list():
    result = to_param_val(["a", "b", "c"])
    assert result == "a b c"


def test_to_param_val_number():
    assert to_param_val(42) == "42"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul && uv run pytest tests/test_utils.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chester.utils'`

**Step 3: Implement `utils.py`**

```python
# src/chester/utils.py
"""Shared utility functions for chester."""
import shlex
from typing import Any


def shellquote(s: str) -> str:
    """Shell-escape a string. Uses shlex.quote from stdlib."""
    if not s:
        return "''"
    return shlex.quote(s)


def to_param_val(v: Any) -> str:
    """Convert a parameter value to its shell-safe string representation."""
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(shellquote(str(item)) for item in v)
    else:
        return shellquote(str(v))
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/houhd/code/chester-overhaul && uv run pytest tests/test_utils.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/utils.py tests/test_utils.py
git commit -m "feat: add shared utils module with shellquote and to_param_val"
```

---

### Task 2: Backend Config Dataclasses

**Files:**
- Create: `src/chester/backends/__init__.py`
- Create: `src/chester/backends/base.py`
- Test: `tests/test_backend_config.py`

**Step 1: Write tests for backend config parsing**

```python
# tests/test_backend_config.py
import pytest
from chester.backends.base import (
    BackendConfig,
    SlurmConfig,
    SingularityConfig,
    parse_backend_config,
)


def test_parse_local_backend():
    raw = {"type": "local"}
    cfg = parse_backend_config("local", raw)
    assert cfg.name == "local"
    assert cfg.type == "local"
    assert cfg.host is None
    assert cfg.remote_dir is None
    assert cfg.singularity is None


def test_parse_ssh_backend():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "prepare": ".chester/backends/myserver/prepare.sh",
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.type == "ssh"
    assert cfg.host == "myserver"
    assert cfg.remote_dir == "/home/user/project"
    assert cfg.prepare == ".chester/backends/myserver/prepare.sh"


def test_parse_slurm_backend():
    raw = {
        "type": "slurm",
        "host": "gl",
        "remote_dir": "/home/user/project",
        "modules": ["singularity"],
        "cuda_module": "cuda/12.1.1",
        "slurm": {
            "partition": "spgpu",
            "time": "72:00:00",
            "gpus": 1,
            "cpus_per_gpu": 4,
            "mem_per_gpu": "80G",
        },
    }
    cfg = parse_backend_config("gl", raw)
    assert cfg.type == "slurm"
    assert cfg.slurm is not None
    assert cfg.slurm.partition == "spgpu"
    assert cfg.slurm.time == "72:00:00"
    assert cfg.slurm.gpus == 1


def test_parse_singularity_config():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "singularity": {
            "image": "/path/to/container.sif",
            "mounts": ["/data:/data", "/scratch"],
            "gpu": True,
        },
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.singularity is not None
    assert cfg.singularity.image == "/path/to/container.sif"
    assert cfg.singularity.gpu is True
    assert len(cfg.singularity.mounts) == 2


def test_slurm_override():
    slurm = SlurmConfig(partition="spgpu", time="72:00:00", gpus=1)
    overridden = slurm.with_overrides({"time": "24:00:00", "gpus": 4})
    assert overridden.time == "24:00:00"
    assert overridden.gpus == 4
    assert overridden.partition == "spgpu"  # unchanged


def test_invalid_backend_type():
    with pytest.raises(ValueError, match="Unknown backend type"):
        parse_backend_config("bad", {"type": "ec2"})


def test_slurm_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_backend_config("gl", {"type": "slurm"})


def test_ssh_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_backend_config("srv", {"type": "ssh"})


def test_slurm_generates_header():
    slurm = SlurmConfig(
        partition="spgpu",
        time="72:00:00",
        nodes=1,
        gpus=2,
        cpus_per_gpu=4,
        mem_per_gpu="80G",
        extra_directives=["--gpu_cmode=shared"],
    )
    header = slurm.to_sbatch_header()
    assert "#!/usr/bin/env bash" in header
    assert "#SBATCH --partition=spgpu" in header
    assert "#SBATCH --time=72:00:00" in header
    assert "#SBATCH --gpus=2" in header
    assert "#SBATCH --cpus-per-gpu=4" in header
    assert "#SBATCH --mem-per-gpu=80G" in header
    assert "#SBATCH --gpu_cmode=shared" in header
    assert "#SBATCH --nodes=1" in header
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul && uv run pytest tests/test_backend_config.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement backend base**

```python
# src/chester/backends/__init__.py
"""Chester backend system."""
from .base import BackendConfig, SlurmConfig, SingularityConfig, parse_backend_config

__all__ = ["BackendConfig", "SlurmConfig", "SingularityConfig", "parse_backend_config"]
```

```python
# src/chester/backends/base.py
"""Backend configuration and base classes."""
from __future__ import annotations

import copy
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


VALID_BACKEND_TYPES = ("local", "ssh", "slurm")


@dataclass
class SingularityConfig:
    """Singularity container configuration."""
    image: str
    mounts: List[str] = field(default_factory=list)
    gpu: bool = False
    build_script: Optional[str] = None


@dataclass
class SlurmConfig:
    """SLURM job parameters with per-experiment override support."""
    partition: Optional[str] = None
    time: Optional[str] = None
    nodes: int = 1
    gpus: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    mem_per_gpu: Optional[str] = None
    ntasks_per_node: Optional[int] = None
    extra_directives: List[str] = field(default_factory=list)

    def with_overrides(self, overrides: Dict[str, Any]) -> SlurmConfig:
        """Return a new SlurmConfig with specified fields overridden."""
        new = copy.deepcopy(self)
        valid_fields = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown SLURM override '{key}'. "
                    f"Valid fields: {sorted(valid_fields)}"
                )
            setattr(new, key, value)
        return new

    def to_sbatch_header(self) -> str:
        """Generate #SBATCH header from structured config."""
        lines = ["#!/usr/bin/env bash"]
        if self.nodes is not None:
            lines.append(f"#SBATCH --nodes={self.nodes}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.gpus is not None:
            lines.append(f"#SBATCH --gpus={self.gpus}")
        if self.cpus_per_gpu is not None:
            lines.append(f"#SBATCH --cpus-per-gpu={self.cpus_per_gpu}")
        if self.mem_per_gpu:
            lines.append(f"#SBATCH --mem-per-gpu={self.mem_per_gpu}")
        if self.ntasks_per_node is not None:
            lines.append(f"#SBATCH --ntasks-per-node={self.ntasks_per_node}")
        for directive in self.extra_directives:
            d = directive if directive.startswith("--") else f"--{directive}"
            lines.append(f"#SBATCH {d}")
        return "\n".join(lines)


@dataclass
class BackendConfig:
    """Configuration for a single backend."""
    name: str
    type: str  # "local", "ssh", "slurm"
    host: Optional[str] = None
    remote_dir: Optional[str] = None
    prepare: Optional[str] = None  # Path to prepare.sh
    modules: List[str] = field(default_factory=list)
    cuda_module: Optional[str] = None
    slurm: Optional[SlurmConfig] = None
    singularity: Optional[SingularityConfig] = None


def parse_backend_config(name: str, raw: Dict[str, Any]) -> BackendConfig:
    """Parse a raw dict from config.yaml into a BackendConfig."""
    backend_type = raw.get("type")
    if backend_type not in VALID_BACKEND_TYPES:
        raise ValueError(
            f"Unknown backend type '{backend_type}' for backend '{name}'. "
            f"Valid types: {VALID_BACKEND_TYPES}"
        )

    # Validate required fields for remote backends
    if backend_type in ("ssh", "slurm"):
        if not raw.get("host"):
            raise ValueError(
                f"Backend '{name}' (type={backend_type}) requires 'host'"
            )
        if not raw.get("remote_dir"):
            raise ValueError(
                f"Backend '{name}' (type={backend_type}) requires 'remote_dir'"
            )

    # Parse singularity config
    sing_raw = raw.get("singularity")
    singularity = None
    if sing_raw:
        singularity = SingularityConfig(
            image=sing_raw["image"],
            mounts=sing_raw.get("mounts", []),
            gpu=sing_raw.get("gpu", False),
            build_script=sing_raw.get("build_script"),
        )

    # Parse SLURM config
    slurm_raw = raw.get("slurm")
    slurm = None
    if slurm_raw:
        slurm = SlurmConfig(
            partition=slurm_raw.get("partition"),
            time=slurm_raw.get("time"),
            nodes=slurm_raw.get("nodes", 1),
            gpus=slurm_raw.get("gpus"),
            cpus_per_gpu=slurm_raw.get("cpus_per_gpu"),
            mem_per_gpu=slurm_raw.get("mem_per_gpu"),
            ntasks_per_node=slurm_raw.get("ntasks_per_node"),
            extra_directives=slurm_raw.get("extra_directives", []),
        )

    return BackendConfig(
        name=name,
        type=backend_type,
        host=raw.get("host"),
        remote_dir=raw.get("remote_dir"),
        prepare=raw.get("prepare"),
        modules=raw.get("modules", []),
        cuda_module=raw.get("cuda_module"),
        slurm=slurm,
        singularity=singularity,
    )


class Backend(ABC):
    """Abstract base class for execution backends."""

    def __init__(self, config: BackendConfig, project_config: Dict[str, Any]):
        self.config = config
        self.project_config = project_config

    @abstractmethod
    def generate_script(self, task: Dict[str, Any], **kwargs) -> str:
        """Generate the execution script for a task.

        Returns the full script content as a string.
        """
        ...

    @abstractmethod
    def submit(self, task: Dict[str, Any], script: str, dry: bool = False) -> None:
        """Submit a task for execution."""
        ...

    def get_prepare_commands(self) -> List[str]:
        """Read prepare.sh and return a source command for it."""
        if not self.config.prepare:
            return []
        prepare_path = self.config.prepare
        project_path = self.project_config.get("project_path", "")
        if not os.path.isabs(prepare_path):
            prepare_path = os.path.join(project_path, prepare_path)
        if not os.path.exists(prepare_path):
            raise FileNotFoundError(
                f"prepare script not found: {prepare_path} "
                f"(backend '{self.config.name}')"
            )
        return [f"source {prepare_path}"]

    def get_python_command(self, base: str = "python") -> str:
        """Wrap python command based on package manager."""
        pkg = self.project_config.get("package_manager", "python")
        if pkg == "uv":
            return f"uv run {base}"
        else:
            return base

    def wrap_with_singularity(self, commands: List[str]) -> str:
        """Wrap a list of commands with singularity exec if configured."""
        sing = self.config.singularity
        if not sing:
            # No singularity — join commands with &&
            return " && ".join(commands)

        mount_args = ""
        if sing.mounts:
            mount_args = " ".join(f"-B {m}" for m in sing.mounts)

        gpu_flag = "--nv" if sing.gpu else ""

        inner = " && ".join(commands)
        return (
            f"singularity exec {mount_args} {gpu_flag} "
            f"{sing.image} /bin/bash -c '{inner}'"
        ).strip()
```

**Step 4: Run tests**

Run: `cd /home/houhd/code/chester-overhaul && uv run pytest tests/test_backend_config.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/backends/ tests/test_backend_config.py
git commit -m "feat: add backend config dataclasses with SLURM params and singularity support"
```

---

### Task 3: New Config System

**Files:**
- Create: `src/chester/config_v2.py` (new config loader — we keep old `config.py` for now)
- Test: `tests/test_config_v2.py`

**Step 1: Write tests**

```python
# tests/test_config_v2.py
import os
import pytest
import tempfile
from pathlib import Path


def test_load_config_from_chester_dir(tmp_path):
    """Config loads from .chester/config.yaml."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    config_file = chester_dir / "config.yaml"
    config_file.write_text("""
project_path: /tmp/test_project
log_dir: data
package_manager: uv
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["project_path"] == "/tmp/test_project"
    assert cfg["package_manager"] == "uv"
    assert "local" in cfg["backends"]


def test_load_config_falls_back_to_chester_yaml(tmp_path):
    """Falls back to chester.yaml at project root with deprecation warning."""
    from chester.config_v2 import load_config

    config_file = tmp_path / "chester.yaml"
    config_file.write_text("""
log_dir: data
package_manager: conda
backends:
  local:
    type: local
""")
    with pytest.warns(DeprecationWarning, match="chester.yaml"):
        cfg = load_config(search_from=tmp_path)
    assert cfg["package_manager"] == "conda"


def test_load_config_validates_backends(tmp_path):
    """Invalid backend type raises ValueError."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  bad:
    type: ec2
""")
    with pytest.raises(ValueError, match="Unknown backend type"):
        load_config(search_from=tmp_path)


def test_load_config_auto_detects_project_path(tmp_path):
    """project_path defaults to config directory's parent."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: data
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["project_path"] == str(tmp_path)


def test_load_config_resolves_log_dir(tmp_path):
    """log_dir is resolved relative to project_path."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: my_data
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["log_dir"] == os.path.join(str(tmp_path), "my_data")


def test_load_config_default_local_backend(tmp_path):
    """If no backends defined, a default 'local' backend is created."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: data
""")
    cfg = load_config(search_from=tmp_path)
    assert "local" in cfg["backends"]
    assert cfg["backends"]["local"].type == "local"


def test_load_config_package_manager_validation(tmp_path):
    """Only python, conda, uv are valid package managers."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
package_manager: poetry
backends:
  local:
    type: local
""")
    with pytest.raises(ValueError, match="package_manager"):
        load_config(search_from=tmp_path)


def test_load_config_env_var_override(tmp_path, monkeypatch):
    """CHESTER_CONFIG_PATH env var overrides search."""
    from chester.config_v2 import load_config

    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("""
package_manager: python
backends:
  local:
    type: local
""")
    monkeypatch.setenv("CHESTER_CONFIG_PATH", str(config_file))
    cfg = load_config()
    assert cfg["package_manager"] == "python"


def test_get_backend_returns_parsed_config(tmp_path):
    """get_backend returns a BackendConfig for the named backend."""
    from chester.config_v2 import load_config, get_backend

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  myhost:
    type: ssh
    host: myhost.example.com
    remote_dir: /home/user/project
""")
    cfg = load_config(search_from=tmp_path)
    backend = get_backend("myhost", cfg)
    assert backend.type == "ssh"
    assert backend.host == "myhost.example.com"


def test_get_backend_unknown_raises(tmp_path):
    """get_backend raises KeyError for unknown backend."""
    from chester.config_v2 import load_config, get_backend

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    with pytest.raises(KeyError, match="nonexistent"):
        get_backend("nonexistent", cfg)
```

**Step 2: Run tests — expect FAIL**

Run: `cd /home/houhd/code/chester-overhaul && uv run pytest tests/test_config_v2.py -v`

**Step 3: Implement `config_v2.py`**

The implementation should:
1. Search for `.chester/config.yaml` first, then `chester.yaml` (with deprecation warning)
2. Parse backends into `BackendConfig` objects via `parse_backend_config`
3. Validate `package_manager` is one of `python`, `conda`, `uv`
4. Resolve relative paths against config file location
5. Provide `get_backend()` helper

(Full implementation: approximately 120 lines following the pattern from existing `config.py`)

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/chester/config_v2.py tests/test_config_v2.py
git commit -m "feat: add new config system loading from .chester/config.yaml"
```

---

### Task 4: LocalBackend Implementation

**Files:**
- Create: `src/chester/backends/local.py`
- Test: `tests/test_backend_local.py`

**Step 1: Write tests**

```python
# tests/test_backend_local.py
import os
import pytest
from chester.backends.base import BackendConfig, parse_backend_config
from chester.backends.local import LocalBackend


@pytest.fixture
def local_config():
    return parse_backend_config("local", {"type": "local"})


@pytest.fixture
def project_config():
    return {"project_path": "/tmp/test", "log_dir": "/tmp/test/data", "package_manager": "python"}


def test_generate_command_basic(local_config, project_config):
    backend = LocalBackend(local_config, project_config)
    task = {
        "log_dir": "/tmp/test/data/exp1",
        "exp_name": "test_exp",
        "args_data": "base64data",
    }
    cmd = backend.generate_command(task, script="train.py", python_command="python")
    assert "python" in cmd
    assert "train.py" in cmd
    assert "--log_dir" in cmd


def test_generate_command_uv(project_config):
    project_config["package_manager"] = "uv"
    cfg = parse_backend_config("local", {"type": "local"})
    backend = LocalBackend(cfg, project_config)
    task = {"log_dir": "/tmp/test/data/exp1", "exp_name": "test", "args_data": "data"}
    cmd = backend.generate_command(task, script="train.py")
    assert "uv run python" in cmd


def test_generate_command_conda(project_config):
    project_config["package_manager"] = "conda"
    cfg = parse_backend_config("local", {"type": "local"})
    backend = LocalBackend(cfg, project_config)
    task = {"log_dir": "/tmp/test/data/exp1", "exp_name": "test", "args_data": "data"}
    cmd = backend.generate_command(task, script="train.py")
    assert cmd.startswith("python")  # conda just uses python directly
    assert "uv" not in cmd


def test_local_with_singularity(project_config):
    cfg = parse_backend_config("local", {
        "type": "local",
        "singularity": {
            "image": "/opt/container.sif",
            "mounts": ["/data:/data"],
            "gpu": True,
        },
    })
    backend = LocalBackend(cfg, project_config)
    task = {"log_dir": "/tmp/test/data/exp1", "exp_name": "test", "args_data": "data"}
    cmd = backend.generate_command(task, script="train.py")
    assert "singularity exec" in cmd
    assert "-B /data:/data" in cmd
    assert "--nv" in cmd
    assert "/opt/container.sif" in cmd


def test_local_with_prepare_script(tmp_path, project_config):
    prepare = tmp_path / "prepare.sh"
    prepare.write_text("#!/bin/bash\nexport MY_VAR=1\n")
    project_config["project_path"] = str(tmp_path)
    cfg = parse_backend_config("local", {
        "type": "local",
        "prepare": str(prepare),
    })
    backend = LocalBackend(cfg, project_config)
    task = {"log_dir": "/tmp/test/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "source" in script
    assert "prepare.sh" in script
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement LocalBackend**

LocalBackend generates a command string and submits it via `subprocess.Popen` or `subprocess.call`. It handles:
- Package manager wrapping (python/conda/uv)
- Optional singularity wrapping
- Optional prepare.sh sourcing
- Hydra command format support

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add src/chester/backends/local.py tests/test_backend_local.py
git commit -m "feat: implement LocalBackend with singularity and prepare.sh support"
```

---

### Task 5: SSHBackend Implementation

**Files:**
- Create: `src/chester/backends/ssh.py`
- Test: `tests/test_backend_ssh.py`

**Step 1: Write tests**

Tests should verify:
- Script generation includes `#!/usr/bin/env bash` header
- Script includes `set -x`, `set -u`, `set -e`
- Script includes `cd {remote_dir}`
- Script sources prepare.sh if configured (using remote path, not local)
- Script wraps python command for package manager
- Script creates `.done` marker on success
- Singularity wrapping works (generates `singularity exec ...` around the command)
- `submit()` generates correct SSH nohup command with PID tracking
- `submit()` creates remote directories via SSH
- `submit()` uses `subprocess.run` (not `os.system`)

```python
# tests/test_backend_ssh.py
import pytest
from chester.backends.base import parse_backend_config
from chester.backends.ssh import SSHBackend


@pytest.fixture
def project_config():
    return {"project_path": "/home/user/project", "log_dir": "/home/user/project/data", "package_manager": "uv"}


def test_ssh_script_has_bash_header(project_config):
    cfg = parse_backend_config("srv", {"type": "ssh", "host": "srv", "remote_dir": "/remote/project"})
    backend = SSHBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert script.startswith("#!/usr/bin/env bash")
    assert "set -x" in script
    assert "set -u" in script
    assert "set -e" in script


def test_ssh_script_cds_to_remote_dir(project_config):
    cfg = parse_backend_config("srv", {"type": "ssh", "host": "srv", "remote_dir": "/remote/project"})
    backend = SSHBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "cd /remote/project" in script


def test_ssh_script_creates_done_marker(project_config):
    cfg = parse_backend_config("srv", {"type": "ssh", "host": "srv", "remote_dir": "/remote/project"})
    backend = SSHBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "touch /remote/project/data/exp1/.done" in script


def test_ssh_script_with_singularity(project_config):
    cfg = parse_backend_config("srv", {
        "type": "ssh", "host": "srv", "remote_dir": "/remote/project",
        "singularity": {"image": "/opt/ml.sif", "mounts": ["/data"], "gpu": True},
    })
    backend = SSHBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "singularity exec" in script
    assert "-B /data" in script
    assert "--nv" in script


def test_ssh_script_wraps_python_for_uv(project_config):
    project_config["package_manager"] = "uv"
    cfg = parse_backend_config("srv", {"type": "ssh", "host": "srv", "remote_dir": "/remote/project"})
    backend = SSHBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "uv run python" in script
```

**Step 2–5: Standard TDD cycle + commit**

```bash
git add src/chester/backends/ssh.py tests/test_backend_ssh.py
git commit -m "feat: implement SSHBackend with prepare.sh and singularity support"
```

---

### Task 6: SlurmBackend Implementation

**Files:**
- Create: `src/chester/backends/slurm.py` (new file in backends/, distinct from old `slurm.py`)
- Test: `tests/test_backend_slurm.py`

**Step 1: Write tests**

```python
# tests/test_backend_slurm.py
import pytest
from chester.backends.base import parse_backend_config, SlurmConfig
from chester.backends.slurm import SlurmBackend


@pytest.fixture
def project_config():
    return {"project_path": "/home/user/project", "log_dir": "/home/user/project/data", "package_manager": "uv"}


def test_slurm_script_has_sbatch_header(project_config):
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "slurm": {"partition": "spgpu", "time": "72:00:00", "gpus": 1},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "#SBATCH --partition=spgpu" in script
    assert "#SBATCH --time=72:00:00" in script
    assert "#SBATCH --gpus=1" in script


def test_slurm_script_with_overrides(project_config):
    """Per-experiment SLURM overrides change the header."""
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "slurm": {"partition": "spgpu", "time": "72:00:00", "gpus": 1},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(
        task, script="train.py",
        slurm_overrides={"time": "6:00:00", "gpus": 4},
    )
    assert "#SBATCH --time=6:00:00" in script
    assert "#SBATCH --gpus=4" in script
    assert "#SBATCH --partition=spgpu" in script  # unchanged


def test_slurm_script_loads_modules(project_config):
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "modules": ["singularity", "python/3.10"],
        "cuda_module": "cuda/12.1.1",
        "slurm": {"partition": "gpu"},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "module load singularity" in script
    assert "module load python/3.10" in script
    assert "module load cuda/12.1.1" in script


def test_slurm_script_creates_done_marker(project_config):
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "slurm": {"partition": "gpu"},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "touch /remote/project/data/exp1/.done" in script


def test_slurm_script_with_singularity(project_config):
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "slurm": {"partition": "gpu", "time": "1:00:00"},
        "singularity": {"image": "/opt/ml.sif", "mounts": ["/data"], "gpu": True},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "singularity exec" in script
    assert "-B /data" in script
    assert "--nv" in script
    assert "/opt/ml.sif" in script


def test_slurm_script_has_stdout_stderr_directives(project_config):
    cfg = parse_backend_config("gl", {
        "type": "slurm", "host": "gl", "remote_dir": "/remote/project",
        "slurm": {"partition": "gpu"},
    })
    backend = SlurmBackend(cfg, project_config)
    task = {"log_dir": "/remote/project/data/exp1", "exp_name": "test", "args_data": "data"}
    script = backend.generate_script(task, script="train.py")
    assert "#SBATCH -o /remote/project/data/exp1/slurm.out" in script
    assert "#SBATCH -e /remote/project/data/exp1/slurm.err" in script
```

**Step 2–5: Standard TDD cycle + commit**

```bash
git add src/chester/backends/slurm.py tests/test_backend_slurm.py
git commit -m "feat: implement SlurmBackend with structured params and singularity support"
```

---

### Task 7: Backend Registry & Factory

**Files:**
- Modify: `src/chester/backends/__init__.py`
- Test: `tests/test_backend_registry.py`

**Step 1: Write tests**

```python
# tests/test_backend_registry.py
import pytest
from chester.backends.base import parse_backend_config
from chester.backends import create_backend
from chester.backends.local import LocalBackend
from chester.backends.ssh import SSHBackend
from chester.backends.slurm import SlurmBackend


@pytest.fixture
def project_config():
    return {"project_path": "/tmp/test", "log_dir": "/tmp/test/data", "package_manager": "python"}


def test_create_local_backend(project_config):
    cfg = parse_backend_config("local", {"type": "local"})
    backend = create_backend(cfg, project_config)
    assert isinstance(backend, LocalBackend)


def test_create_ssh_backend(project_config):
    cfg = parse_backend_config("srv", {"type": "ssh", "host": "x", "remote_dir": "/x"})
    backend = create_backend(cfg, project_config)
    assert isinstance(backend, SSHBackend)


def test_create_slurm_backend(project_config):
    cfg = parse_backend_config("gl", {"type": "slurm", "host": "gl", "remote_dir": "/x"})
    backend = create_backend(cfg, project_config)
    assert isinstance(backend, SlurmBackend)
```

**Step 3: Implement factory**

```python
# In src/chester/backends/__init__.py
from .base import Backend, BackendConfig, SlurmConfig, SingularityConfig, parse_backend_config

__all__ = [
    "Backend", "BackendConfig", "SlurmConfig", "SingularityConfig",
    "parse_backend_config", "create_backend",
]


def create_backend(config: BackendConfig, project_config: dict) -> Backend:
    """Create a Backend instance from config."""
    if config.type == "local":
        from .local import LocalBackend
        return LocalBackend(config, project_config)
    elif config.type == "ssh":
        from .ssh import SSHBackend
        return SSHBackend(config, project_config)
    elif config.type == "slurm":
        from .slurm import SlurmBackend
        return SlurmBackend(config, project_config)
    raise ValueError(f"Unknown backend type: {config.type}")
```

**Step 5: Commit**

```bash
git add src/chester/backends/__init__.py tests/test_backend_registry.py
git commit -m "feat: add backend factory for creating backend instances from config"
```

---

### Task 8: Rewrite `run_experiment_lite`

**Files:**
- Modify: `src/chester/run_exp.py`
- Test: `tests/test_run_exp_v2.py`

This is the core refactor. The new `run_experiment_lite` should:

1. **Keep**: `VariantGenerator`, `AttrDict`, `VariantDict`, `variant` decorator, `monitor_processes`
2. **Replace**: The 340-line if/elif mode dispatch with a single `backend.submit()` call
3. **Add**: `slurm_overrides` parameter for per-experiment SLURM config
4. **Add**: `use_singularity` parameter override (True/False/None)
5. **Remove**: hardcoded mode lists (`['gl', 'seuss', 'psc', 'satori']`)
6. **Remove**: `config.HOST_ADDRESS[mode]`, `config.REMOTE_DIR[mode]` lookups (now in backend config)
7. **Replace**: `os.system` calls with `subprocess.run`
8. **Keep**: auto-pull manifest system (works with any backend)

**Step 1: Write tests for the new dispatch**

```python
# tests/test_run_exp_v2.py
import pytest


def test_run_experiment_lite_rejects_ec2_mode():
    """ec2 mode raises ValueError with deprecation message."""
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="ec2",
            exp_prefix="test",
        )


def test_run_experiment_lite_rejects_autobot_mode():
    """autobot mode raises ValueError with deprecation message."""
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="autobot",
            exp_prefix="test",
        )


def test_variant_generator_unchanged():
    """VariantGenerator API is unchanged by the refactor."""
    from chester.run_exp import VariantGenerator
    vg = VariantGenerator()
    vg.add('lr', [0.001, 0.01])
    vg.add('bs', [32, 64])
    variants = vg.variants()
    assert len(variants) == 4
    assert variants[0].get('chester_first_variant')
    assert variants[-1].get('chester_last_variant')
```

**Step 3: Implement the refactored `run_experiment_lite`**

New flow (pseudocode):
```python
def run_experiment_lite(
    stub_method_call=None,
    variant=None,
    mode="local",
    exp_prefix="experiment",
    exp_name=None,
    log_dir=None,
    sub_dir="train",
    script=None,
    python_command="python",
    dry=False,
    env=None,
    use_cloudpickle=True,
    variations=None,
    print_command=True,
    slurm_overrides=None,       # NEW: dict of SLURM param overrides
    use_singularity=None,       # NEW: True/False/None (None=use backend default)
    auto_pull=False,
    auto_pull_interval=60,
    extra_pull_dirs=None,
    hydra_enabled=False,
    hydra_flags=None,
    launch_with_subprocess=True,
    wait_subprocess=True,
    max_num_processes=10,
    **kwargs,
):
    # Reject deprecated modes
    if mode in ("ec2", "autobot"):
        raise ValueError(f"Mode '{mode}' is deprecated. Use 'local', 'ssh', or 'slurm' backends.")

    # Load config, resolve backend
    cfg = _get_project_config()
    backend_config = get_backend(mode, cfg)
    backend = create_backend(backend_config, cfg)

    # Handle singularity override
    if use_singularity is False:
        backend.config.singularity = None
    elif use_singularity is True and not backend.config.singularity:
        raise ValueError(f"use_singularity=True but backend '{mode}' has no singularity config")

    # Task serialization (same logic as current — cloudpickle, base64, etc.)
    # Log dir computation (same logic — local/remote mapping)
    # Experiment naming (same logic)

    # Generate and submit via backend
    script_content = backend.generate_script(
        task, script=script, slurm_overrides=slurm_overrides,
        hydra_enabled=hydra_enabled, hydra_flags=hydra_flags,
    )
    backend.submit(task, script_content, dry=dry)

    # Auto-pull manifest (same logic as current)
```

**Step 5: Commit**

```bash
git add src/chester/run_exp.py tests/test_run_exp_v2.py
git commit -m "refactor: rewrite run_experiment_lite to use backend system"
```

---

### Task 9: Deprecate EC2/Autobot & Clean Dead Code

**Files:**
- Modify: `src/chester/config_ec2.py` (add deprecation warning at import)
- Modify: `src/chester/utils_s3.py` (add deprecation warning at import)
- Modify: `src/chester/scheduler/remote_scheduler.py` (add deprecation warning)
- Modify: `src/chester/run_exp_worker.py` (remove commented-out rllab code)
- Modify: `src/chester/__init__.py` (fix version to use importlib.metadata)
- Delete duplicated `_shellquote`, `_to_param_val` from `run_exp.py` and old `slurm.py`
- Test: `tests/test_deprecation.py`

**Step 1: Write tests**

```python
# tests/test_deprecation.py
import pytest
import warnings


def test_config_ec2_import_warns():
    with pytest.warns(DeprecationWarning, match="EC2"):
        from chester import config_ec2


def test_utils_s3_import_warns():
    with pytest.warns(DeprecationWarning, match="EC2"):
        from chester import utils_s3
```

**Step 3: Changes to make**

- Add `warnings.warn("EC2 support is deprecated...", DeprecationWarning, stacklevel=2)` at top of `config_ec2.py`, `utils_s3.py`
- Remove all commented-out rllab code from `run_exp_worker.py`
- Fix `__init__.py` version: use `importlib.metadata.version("chester-ml")`
- Delete `_shellquote`, `_to_param_val`, `_find_unsafe` from `run_exp.py` and old `slurm.py` — import from `utils.py`
- Fix mutable default args (`env={}` → `env=None`)

**Step 5: Commit**

```bash
git commit -m "refactor: deprecate EC2/autobot, clean dead code, fix version"
```

---

### Task 10: Auto-pull Modernization

**Files:**
- Modify: `src/chester/auto_pull.py`
- Test: `tests/test_auto_pull.py`

**Changes:**
- Replace `os.system(cmd)` spawning in `run_exp.py` with `subprocess.Popen(..., start_new_session=True)`
- Replace shell-string SSH commands with `subprocess.run(["ssh", host, ...])` (list form)
- No architectural changes — the auto_pull design is sound

**Step 5: Commit**

```bash
git commit -m "refactor: modernize auto_pull to use subprocess instead of os.system"
```

---

### Task 11: Update Old `slurm.py` to Thin Wrapper

**Files:**
- Modify: `src/chester/slurm.py` (the old one at package root)

The old `slurm.py` should become a backward-compat wrapper that delegates to the new backend system:
- Remove `get_package_manager_setup_commands` (now in prepare.sh)
- Remove `get_python_command` (now in `Backend.get_python_command`)
- Remove all hardcoded env exports (NCCL, CUDA, ninja, etc.)
- Keep `to_local_command` signature but import `shellquote`/`to_param_val` from `utils.py`
- `to_ssh_command` and `to_slurm_command` can emit deprecation warnings pointing to backends

**Step 5: Commit**

```bash
git commit -m "refactor: simplify old slurm.py, remove hardcoded env setup"
```

---

### Task 12: Migration Guide & Example Configs

**Files:**
- Create: `docs/migration-v1-to-v2.md`
- Create: `examples/chester-v2/.chester/config.yaml`
- Create: `examples/chester-v2/.chester/backends/greatlakes/prepare.sh`
- Create: `examples/chester-v2/.chester/backends/ssh-server/prepare.sh`
- Create: `examples/chester-v2/.chester/backends/local/prepare.sh`
- Create: `examples/chester-v2/launch_example.py`
- Update: `CLAUDE.md` with new architecture

**Content of migration guide:**
1. Config file move: `chester.yaml` → `.chester/config.yaml`
2. Config format changes (flat dicts → structured `backends` section)
3. What goes in `prepare.sh` (all the env exports that were hardcoded — full list from Appendix A)
4. SLURM header changes (raw string → structured `slurm:` config with named fields)
5. Package manager: `sync_on_launch`, `conda_env`, `conda_command` → user's prepare.sh
6. New `slurm_overrides` parameter
7. Deprecated modes (ec2, autobot) — pin `chester-ml<0.5` if needed

**Step 5: Commit**

```bash
git commit -m "docs: add migration guide and v2 example configs"
```

---

### Task 13: Update Tests

**Files:**
- Modify: `tests/test_local.py` (fix `AssertionError` typo, convert to pytest)
- Create: `tests/test_integration.py` (end-to-end local execution test)

**Step 1: Fix existing tests**

- Fix `AssertionError` → `AssertionError` (line 233)
- Convert from custom test runner to standard pytest functions
- Update imports for new module structure

**Step 2: Add integration test**

```python
# tests/test_integration.py
def test_end_to_end_local_dry_run(tmp_path):
    """Full local execution: config + backend + script generation (dry run)."""
    # Set up .chester/config.yaml in tmp_path
    # Set up a simple stub function
    # Call run_experiment_lite(mode='local', dry=True)
    # Verify the command is printed correctly
```

**Step 3: Commit**

```bash
git commit -m "test: fix existing tests, add integration test for new backend system"
```

---

## Execution Order & Dependencies

```
Task 1 (utils) ──────────────────────────────┐
Task 2 (backend config) ────────────────────┤
                                             ├─→ Task 7 (registry) ─→ Task 8 (run_exp rewrite)
Task 4 (LocalBackend) ──────────────────────┤                              │
Task 5 (SSHBackend) ────────────────────────┤                              ├─→ Task 11 (old slurm.py)
Task 6 (SlurmBackend) ──────────────────────┘                              │
                                                                           ├─→ Task 12 (migration docs)
Task 3 (config_v2) ────────────────────────────────────────────────────────┤
                                                                           ├─→ Task 13 (tests)
Task 9 (deprecation) ─────────────────────────────────────────────────────┘
Task 10 (auto_pull) ──────── (independent, can be done anytime)
```

**Parallelizable groups:**
- Tasks 1, 2, 3 can be done in parallel
- Tasks 4, 5, 6 can be done in parallel (after Task 2)
- Task 10 is independent of everything

---

## Summary of Breaking Changes

1. **Config location**: `chester.yaml` → `.chester/config.yaml` (old location still works with deprecation warning)
2. **Config format**: Flat `host_address`, `remote_dir`, `remote_header` dicts → structured `backends` section
3. **Package manager setup**: Hardcoded in chester → user's `prepare.sh` scripts
4. **SLURM headers**: Raw string with `$gpus` substitution → structured `slurm:` config with named fields
5. **EC2 mode**: Removed (raises `ValueError`)
6. **Autobot mode**: Removed (raises `ValueError`)
7. **`prepare_commands`**: Removed from config (use `prepare` script path instead)
8. **`sync_on_launch`**: Removed from config (user does this in `prepare.sh`)
9. **`conda_env`, `conda_command`**: Removed from config (user does this in `prepare.sh`)
10. **`run_experiment_lite` parameters**: New `slurm_overrides` dict replaces inline SLURM control
11. **Mode names**: No more hardcoded `gl`, `seuss`, `psc`, `satori` — any name works if defined in config
