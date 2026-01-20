# Chester TODO

## Per-host prepare_commands

Currently `prepare_commands` in `chester.yaml` is global - same commands run on all hosts. Different hosts often need different setup (e.g., ROS on physical robots, module loads on clusters).

**Proposed config:**
```yaml
prepare_commands:
  - export PYTHONPATH=$PWD:$PYTHONPATH  # Global

prepare_commands_per_host:
  armfranka:
    - source /opt/ros/noetic/setup.bash
  gl:
    - module load cuda/12.1
```

**Implementation:**
1. Add `prepare_commands_per_host: {}` to `config.py` defaults and `_ATTR_MAPPING`
2. Modify `slurm.py:get_package_manager_setup_commands()` to accept `host` parameter
3. After global prepare_commands, append host-specific commands

---

## Fix pre_commands consistency

The `run_experiment_lite(pre_commands=[...])` parameter is handled inconsistently:

| Backend | Status |
|---------|--------|
| local | IGNORED (just prints warning) |
| local_singularity | IGNORED |
| SLURM | Works |
| SSH | NOT HANDLED |
| autobot | Works (uses to_slurm_command) |
| EC2 | Works |

**Fix:** Update `to_local_command()` and `to_ssh_command()` to properly handle pre_commands.

---

## Post-commands support

Currently only `.done` marker is created on completion. Users might want cleanup commands.

**Proposed config:**
```yaml
post_commands:
  - rm -rf /tmp/cache_*
```

---

## Pre-rsync hooks

For building assets before syncing code to remote:

**Proposed config:**
```yaml
pre_rsync_commands:
  - python scripts/generate_configs.py
```
