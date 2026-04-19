# Singularity Container Support in Chester

Chester supports running experiments inside Singularity containers across all three backend types: local, SSH, and SLURM. This document covers configuration, runtime behavior, and common patterns.

## Overview

When a backend has a `singularity:` config block, every command (prepare scripts, Python execution, etc.) runs inside the container via `singularity exec`. The `enabled` flag controls whether singularity is active:

- **`enabled: false` (default)** — Singularity is dormant. Activate at runtime with `use_singularity=True` in `run_experiment_lite()` or the `--singularity` CLI flag.
- **`enabled: true`** — Always use singularity for that backend (typical for SLURM clusters where the image is pre-built and staged).

## Full Singularity Config Reference

```yaml
singularity:
  image: /path/to/image.sif     # required; relative paths resolved from project root
  gpu: true                      # --nv flag (NVIDIA GPU passthrough)
  fakeroot: true                 # --fakeroot flag (default: false)
  pid_namespace: true            # --pid (isolate PID namespace)
  writable_tmpfs: true           # --writable-tmpfs (in-memory ephemeral overlay)
  workdir: /workspace            # --pwd inside the container
  enabled: true                  # activate this singularity config
  mounts:                        # bind mounts (--bind)
    - /usr/share/glvnd            # simple host path
    - src:/workspace/src          # relative src → resolved from project root
    - /abs/path:/container/path   # absolute src:dst
    - ~/.cache:/root/.cache       # tilde expanded by bash
    - $HOME/.config:/root/.config # env vars expanded by bash
  prepare: .chester/backends/singularity/prepare.sh   # sourced INSIDE container
  overlay: .containers/myoverlay.ext3                 # persistent ext3 overlay image
  overlay_size: 10240             # MB when creating overlay (default: 10240 = 10 GB)
```

## Option Details

### `image` (required)

Path to the Singularity image (`.sif` file). Relative paths are resolved:
- **Local backend**: relative to project root
- **SSH/SLURM backends**: relative to `remote_dir` on the remote host

Example:
```yaml
image: .containers/base.sif              # local path
image: /nfs/shared/images/myimage.sif    # absolute path
```

### `gpu: true`

Adds the `--nv` flag to `singularity exec` for NVIDIA GPU passthrough. Required for CUDA workloads.

```yaml
gpu: true    # adds --nv
```

### `fakeroot: true`

Adds the `--fakeroot` flag for pseudo-root operations (e.g., installing packages with `apt`, `yum`, etc.). Useful when building overlays or writing to system directories inside the container.

**Warning**: Not available on all HPC clusters. For example, Great Lakes (University of Michigan) disables fakeroot due to security policies.

```yaml
fakeroot: true    # adds --fakeroot
```

### `pid_namespace: true`

Adds the `--pid` flag to isolate the PID namespace. Critical on SLURM clusters where background processes (e.g., Isaac Sim's Omniverse Kit GPU daemon) must be killed when the job ends.

Without `--pid`, background processes can linger and hold GPU memory, preventing subsequent jobs from running.

```yaml
pid_namespace: true    # adds --pid
```

### `writable_tmpfs: true`

Adds the `--writable-tmpfs` flag — an in-memory ephemeral overlay layer. Lets applications write to paths inside the read-only SIF image without requiring a persistent overlay or fakeroot. All writes are lost when the container exits.

Useful for:
- Isaac Sim writing to `/opt/omniverse/` or similar system directories
- Temporary cache/log directories
- Avoiding disk I/O overhead for ephemeral writes

```yaml
writable_tmpfs: true    # adds --writable-tmpfs
```

### `workdir`

Sets the working directory inside the container (`--pwd`). If not specified, defaults to the current directory.

```yaml
workdir: /workspace
```

### `mounts`

List of bind mounts passed as `--bind` flags. Supports multiple formats:

| Format | Example | Behavior |
|--------|---------|----------|
| Simple path | `/usr/share/glvnd` | Bind to the same path in container |
| Explicit mapping | `src:dst` | Bind host `src` to container `dst` |
| Relative `src` | `src:/workspace/src` | Resolve `src` relative to project root (local) or `remote_dir` (remote) |
| Absolute paths | `/abs/path:/container/path` | Use as-is |
| Tilde expansion | `~/.cache:/root/.cache` | Left as-is; bash expands at runtime |
| Env vars | `$HOME/.config:/root/.config` | Left as-is; bash expands at runtime |

Examples:

```yaml
mounts:
  - /usr/share/glvnd                      # GPU/display libraries
  - src:/workspace/src                    # project source code
  - data:/workspace/data                  # data directory
  - /nfs/shared/datasets:/data/datasets   # absolute path on NFS
  - ~/.cache/myapp:/opt/myapp/cache       # user cache, tilde expanded
  - $HOME/.config:/root/.config           # env var expanded
```

### `prepare`

Path to a shell script sourced **inside the container** before the Python command runs. Use for in-container environment setup:
- Activating a conda environment
- Setting LD_LIBRARY_PATH
- Running container-specific initialization

The script is sourced (not executed), so it runs in the same shell as the Python command.

```yaml
prepare: .chester/backends/singularity/prepare.sh
```

Example `prepare.sh`:

```bash
#!/usr/bin/env bash
# Inside the container; activate conda environment from overlay
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

# Set container-specific env vars
export MYAPP_CONFIG=/etc/myapp/config.yaml
```

### `overlay` and `overlay_size`

A persistent ext3 overlay image allows writing to the container without rebuilding the SIF. Chester creates the overlay lazily on first use with `singularity overlay create --size {overlay_size} {overlay}`.

**`overlay`**: Path to the overlay image. Relative paths are resolved from the project root.

**`overlay_size`**: Size in MB when creating the overlay (default: 10240 = 10 GB).

Use overlays to:
- Install packages into a base image without rebuilding
- Persist data/cache across job runs
- Work around read-only filesystem restrictions

Example:

```yaml
singularity:
  image: .containers/base.sif
  overlay: .containers/myoverlay.ext3
  overlay_size: 20480      # 20 GB
  fakeroot: true
  prepare: .chester/backends/singularity/in_container_prepare.sh
```

Then in `in_container_prepare.sh`:

```bash
#!/usr/bin/env bash
# Inside container; overlay is mounted
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv
```

On first run, Chester creates the 20 GB overlay. On subsequent runs, Chester reuses it, preserving any installed packages or data.

## Global vs. Backend-Specific Singularity Config

A top-level `singularity:` block in `.chester/config.yaml` defines shared defaults inherited by all backends:

```yaml
singularity:
  image: .containers/base.sif
  gpu: true
  enabled: false

backends:
  local:
    type: local
    # inherits global singularity config: image=base.sif, gpu=true, enabled=false

  mycluster:
    type: slurm
    singularity:
      enabled: true        # override: always on for this backend
      # still has image, gpu from global
```

**Important**: A backend's `singularity:` block does a **full replacement** of the global block, not a merge. If you set any field at the backend level, re-specify all fields you need:

```yaml
singularity:                 # global defaults
  image: .containers/base.sif
  gpu: true
  enabled: false

backends:
  mycluster:
    type: slurm
    singularity:
      enabled: true
      image: .containers/base.sif  # must re-specify; full replacement
      gpu: true
      fakeroot: false
      pid_namespace: true
      mounts:
        - /usr/share/glvnd
        - src:/workspace/src
```

## Runtime Override: `use_singularity` Parameter

Override singularity on/off for a single run using the `use_singularity` parameter:

```python
from chester import run_experiment_lite

# Force singularity on (requires singularity config in backend)
run_experiment_lite(..., use_singularity=True)

# Force singularity off (ignore backend config)
run_experiment_lite(..., use_singularity=False)

# Respect backend config (default)
run_experiment_lite(..., use_singularity=None)
```

Typical CLI pattern in launchers:

```python
import click
from chester import run_experiment_lite

@click.option("--singularity/--no-singularity", default=None,
              help="Use singularity container (None=respect backend config)")
def main(singularity):
    run_experiment_lite(
        backend="mycluster",
        use_singularity=singularity,
        ...
    )

if __name__ == "__main__":
    main()
```

Usage:

```bash
python launcher.py --singularity          # force on
python launcher.py --no-singularity       # force off
python launcher.py                        # respect config (default)
```

## Common Patterns

### Pattern 1: Opt-In Singularity (Local Development + Remote Cluster)

Use singularity by default only on the cluster; opt in for local testing:

```yaml
singularity:
  image: .containers/base.sif
  gpu: true
  enabled: false              # opt-in by default

backends:
  local:
    type: local
    prepare: .chester/backends/local/prepare.sh
    # singularity is opt-in

  mycluster:
    type: slurm
    singularity:
      enabled: true           # always on for this backend
      gpu: true
      pid_namespace: true
      writable_tmpfs: true
      mounts:
        - /usr/share/glvnd
        - src:/workspace/src
```

Run locally without singularity:

```bash
python launcher.py --backend local
```

Run on cluster with singularity:

```bash
python launcher.py --backend mycluster
```

Run locally with singularity for testing:

```bash
python launcher.py --backend local --singularity
```

### Pattern 2: Persistent Overlay for Package Installation

Build a base image, then install packages into an overlay without rebuilding:

```yaml
singularity:
  image: .containers/base.sif
  overlay: .containers/packages.ext3
  overlay_size: 20480
  fakeroot: true
  gpu: true
  pid_namespace: true
  mounts:
    - /usr/share/glvnd
    - src:/workspace/src
  prepare: .chester/backends/singularity/prepare.sh
```

First run: Chester creates the overlay and sources `prepare.sh` inside it.

To add packages (one-time setup):

```bash
singularity exec --overlay .containers/packages.ext3 --fakeroot \
  .containers/base.sif \
  apt update && apt install -y mypackage
```

Subsequent runs reuse the overlay with all installed packages.

### Pattern 3: Multi-GPU SSH with Singularity Batching

Run parallel jobs on a multi-GPU workstation:

```yaml
backends:
  workstation:
    type: ssh
    host: workstation
    remote_dir: /home/user/myproject
    batch_gpu: 4                 # 4 jobs in parallel (one per GPU)
    singularity:
      enabled: true
      image: .containers/myimage.sif
      gpu: true
      workdir: /workspace
      mounts:
        - src:/workspace/src
        - data:/workspace/data
```

Each job automatically gets `CUDA_VISIBLE_DEVICES` set to a unique GPU.

### Pattern 4: GPU/Display Passthrough for Isaac Sim

For Isaac Sim or Omniverse applications, ensure GPU and display libraries are mounted:

```yaml
singularity:
  image: .containers/isaac-sim.sif
  gpu: true
  pid_namespace: true         # kill background processes on exit
  writable_tmpfs: true        # let app write to /opt/omniverse, etc.
  mounts:
    - /usr/share/glvnd        # NVIDIA OpenGL libraries
    - src:/workspace/src
    - data:/workspace/data
```

## Troubleshooting

### Container processes don't terminate

**Symptom**: Job ends but background GPU processes remain, holding GPU memory.

**Solution**: Set `pid_namespace: true`:

```yaml
singularity:
  pid_namespace: true
```

The `--pid` flag isolates the PID namespace, ensuring all container processes are killed when the job ends.

### Permission denied writing to overlay

**Symptom**: `Permission denied` when writing to paths in the container.

**Solutions**:
- Set `fakeroot: true` (if cluster allows it)
- Use `writable_tmpfs: true` for ephemeral writes
- Check overlay creation: `singularity overlay create --size {size} {overlay}`

### Mount not visible inside container

**Symptom**: A mount path exists on the host but is empty or missing inside the container.

**Solutions**:
1. Check absolute path correctness (for absolute mounts, both sides must exist)
2. For relative mounts, verify resolution: relative to project root (local) or `remote_dir` (remote)
3. Verify bind mount syntax: `src:dst` (no spaces)
4. Check singular vs. plural: mounts must be a list (`-`)

### Image file not found

**Symptom**: `singularity exec: error while loading shared libraries` or "image not found".

**Solutions**:
1. Verify image path: relative to project root (local) or `remote_dir` (remote)
2. Check file permissions: `ls -la /path/to/image.sif`
3. On remote backend, ensure image is synced or mounted (e.g., on NFS)

## See Also

- [Chester Configuration Guide](config.md)
- [Example Configs](examples/)
- [Singularity Documentation](https://sylabs.io/docs/)
