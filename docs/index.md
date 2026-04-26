# Chester Documentation

**Chester** is a Python ML experiment launcher for managing local, remote (SSH), and HPC (SLURM) jobs with unified configuration, parameter sweeps, container support, and reproducible tracking via Git snapshots.

## Features

- **Multiple backends:** local execution, SSH remote hosts, SLURM job submission
- **Container support:** Singularity containers on all backends with overlay mounts and image management
- **Parameter sweeps:** `VariantGenerator` for grid/cartesian sweeps, serial execution steps, dependent SLURM jobs, and randomized ordering
- **Hydra integration:** First-class support for Hydra config framework with `hydra_enabled` mode
- **Git snapshot:** Automatic capture of repo state (`git_info.json` + `git_diff.patch`) at launch time for reproducibility
- **Submodule pinning:** Pin exact submodule commits per job with remote worktrees
- **SSH batch-GPU mode:** Distribute jobs across GPU indices with `batch_gpu` config
- **Extra sync directories:** Sync additional project paths to remote hosts via `extra_sync_dirs`
- **Per-experiment SLURM overrides:** Override cluster settings per job with `slurm_overrides`
- **Prepare scripts:** Backend-specific `prepare.sh` scripts for environment setup
- **rsync filtering:** Include and exclude patterns for granular code sync control

## Installation

```bash
pip install chester-ml
```

Or with `uv`:

```bash
uv add chester-ml
```

## Documentation Index

| Topic | Description |
|-------|-------------|
| [Configuration Reference](configuration.md) | Complete `.chester/config.yaml` guide with examples |
| [Backends](backends.md) | Local, SSH, and SLURM backend options and configuration |
| [Singularity](singularity.md) | Container image setup, overlays, and mount management |
| [Parameter Sweeps](parameter-sweeps.md) | VariantGenerator for grid searches, serial steps, and job dependencies |
| [Hydra Integration](hydra.md) | Hydra override format and `hydra_enabled` mode |
| [Git Snapshot](git-snapshot.md) | Git state tracking and reproducibility features |
| [Submodule Pinning](submodule-pinning.md) | Pinning submodule commits per job with remote worktrees |
| [Examples](examples/) | Annotated real-world configuration examples |

Upgrading from chester 1.x? See [legacy/migration-v1-to-v2.md](legacy/migration-v1-to-v2.md).
