# Chester

Chester (`chester-ml` on PyPI) is a Python experiment launcher for ML workflows. Define your training function and parameter sweep — Chester handles dispatching jobs to local subprocesses, SSH servers, or SLURM clusters, with Singularity container support, code syncing, and reproducibility snapshots baked in.

## Installation

```bash
pip install chester-ml
# or
uv add chester-ml
```

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
    host: myserver                       # SSH alias from ~/.ssh/config
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
      gpus: 1
      cpus_per_gpu: 8
      mem_per_gpu: 32G
```

**2. Write a launcher:**

```python
from chester.run_exp import run_experiment_lite, VariantGenerator, detect_local_gpus, flush_backend

def run_task(variant, log_dir, exp_name):
    print(f"lr={variant['lr']}, batch={variant['batch_size']}")
    # ... your training code ...

vg = VariantGenerator()
vg.add('lr', [1e-3, 1e-4])
vg.add('batch_size', [32, 64])

for v in vg.variants():
    run_experiment_lite(
        stub_method_call=run_task,
        variant=v,
        mode='local',        # or 'myserver', 'mycluster'
        exp_prefix='sweep',
        max_num_processes=max(1, len(detect_local_gpus())),
    )

flush_backend('local')       # no-op for local; required after loop for batch SSH mode
```

**3. Run:**

```bash
python launcher.py           # local
python launcher.py myserver  # SSH
python launcher.py mycluster # SLURM
```

## Features

- **Three backend types**: local subprocess, SSH (`nohup`), SLURM (`sbatch`)
- **Singularity** on all backends: GPU passthrough, persistent overlays, per-container `prepare.sh`
- **VariantGenerator**: cartesian product sweeps, dependent parameters, `order="serial"` (multi-step single job) and `order="dependent"` (chained SLURM jobs)
- **Hydra integration**: pass parameters as `key=value` overrides with OmegaConf interpolation support
- **Git snapshot**: saves `git_info.json` + `git_diff.patch` per run for full reproducibility
- **Submodule commit pinning**: pin specific submodule commits per job via remote git worktrees
- **SSH batch-GPU mode**: accumulate jobs across variants, fire one per GPU on `flush_backend()`
- **Extra sync dirs**: rsync additional paths (datasets, checkpoints) to remote before submission
- **Per-experiment SLURM overrides**: tune `time`, `gpus`, `mem_per_gpu`, etc. per `run_experiment_lite()` call
- **Graceful Ctrl+C**: local kills subprocesses and stops the queue; remote detaches and lets jobs keep running

## Documentation

Full reference in [`docs/`](docs/):

| Doc | What it covers |
|-----|----------------|
| [Configuration](docs/configuration.md) | `.chester/config.yaml` — all fields, global singularity block, YAML anchors |
| [Backends](docs/backends.md) | Local, SSH, SLURM — all options, batch-GPU, extra sync dirs |
| [Singularity](docs/singularity.md) | Mounts, overlays, PID namespace, fakeroot, runtime override |
| [Parameter Sweeps](docs/parameter-sweeps.md) | VariantGenerator, serial/dependent ordering, derive, flush_backend |
| [Hydra](docs/hydra.md) | `hydra_enabled`, flags, OmegaConf interpolations |
| [Git Snapshot](docs/git-snapshot.md) | `git_info.json`, `git_diff.patch`, submodule tracking, recovery |
| [Submodule Pinning](docs/submodule-pinning.md) | Per-job submodule commit pinning via worktrees |
| [Examples](docs/examples/) | Annotated real-world config patterns |

## Example Configs

See [`docs/examples/`](docs/examples/) for annotated configs:

- [`simple.yaml`](docs/examples/simple.yaml) — local + SSH + SLURM, no Singularity
- [`singularity-slurm.yaml`](docs/examples/singularity-slurm.yaml) — production SLURM + Singularity with NFS mounts
- [`multi-gpu-ssh.yaml`](docs/examples/multi-gpu-ssh.yaml) — multi-GPU SSH workstation with batch mode

## Project Layout

```
myproject/
├── .chester/
│   ├── config.yaml                    # Main config
│   └── backends/
│       ├── local/
│       │   └── prepare.sh             # Local env setup
│       ├── mycluster/
│       │   └── prepare.sh             # Cluster setup (modules, paths)
│       └── myserver/
│           └── prepare.sh             # SSH server setup
├── launchers/
│   └── launch_sweep.py
└── src/
```

Chester searches for `.chester/config.yaml` upward from the current directory, stopping at the `.git` root. Override with `$CHESTER_CONFIG_PATH`.

## License

MIT
