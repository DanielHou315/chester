# Chester 2.0.0 — v1 Deprecation & Cleanup Design

**Date:** 2026-04-26  
**Branch:** `release/2.0.0`  
**Scope:** Remove all v1 legacy code; enforce the v2 backend architecture as the single supported path; publish as a breaking major version.

---

## Background

Chester has carried two parallel systems since the v2 backend rewrite:

- **v1**: Mode-string dispatch (`"ec2"`, `"autobot"`, `"singularity"`) backed by a monolithic `config.py` shim, `slurm.py` command generators, and EC2/Autobot-era utilities.
- **v2**: Named backends declared in `.chester/config.yaml`, dispatched through `BackendConfig` dataclasses and polymorphic `Backend` subclasses (`LocalBackend`, `SSHBackend`, `SlurmBackend`).

All v2 functionality is complete. v1 symbols have been emitting `DeprecationWarning` for several releases. External consumer count is effectively zero. This release removes v1 entirely and cuts a clean major version.

---

## Decision

**Option A: All-at-once cleanup in one branch.** Delete every v1 artifact, rename `config_v2.py` → `config.py` (erasing the "v2" branding), clean up all internal v1 references, and publish as `2.0.0`.

---

## Files to Delete

| File | Reason |
|---|---|
| `src/chester/config.py` | v1 compat `__getattr__` shim — replaced by renamed `config_v2.py` |
| `src/chester/slurm.py` | v1 command generators — replaced by `Backend` subclasses |
| `src/chester/config_ec2.py` | EC2 config module — EC2 mode removed |
| `src/chester/utils_s3.py` | S3 utilities — EC2 era |
| `src/chester/pull_result.py` | v1 rsync helper using `config.REMOTE_DIR` dict |
| `src/chester/pull_s3_result.py` | S3 result pulling |
| `src/chester/setup_ec2_for_chester.py` | EC2 setup script |
| `src/chester/availability_test.py` | Old SLURM availability probe |
| `src/chester/scheduler/` (entire dir) | Autobot GPU scheduler, kill/list jobs, v1 batch launcher |
| `tests/test_deprecation.py` | Tests for deleted symbols |

---

## Files to Rename

| Old | New | Notes |
|---|---|---|
| `src/chester/config_v2.py` | `src/chester/config.py` | Content unchanged; "v2" was a migration label |
| `tests/test_config_v2.py` | `tests/test_config.py` | Update import paths |
| `tests/test_run_exp_v2.py` | `tests/test_run_exp.py` | Remove tests for now-deleted mode guards |

---

## Code Changes in `run_exp.py`

1. Remove `from . import config` (v1 shim import)
2. Update `from chester.config_v2 import` → `from chester.config import`
3. Delete `_DEPRECATED_MODES` frozenset and the guard block at line ~1183
4. Delete `_map_local_to_remote_log_dir()` (v1 version using `config.PROJECT_PATH`/`config.REMOTE_DIR`)
5. Rename `_map_local_to_remote_log_dir_v2` → `_map_local_to_remote_log_dir` at definition and all call sites
6. Delete `query_yes_no()` deprecated alias
7. Delete `rsync_code()` (v1 version using `config.PROJECT_PATH`/`config.RSYNC_*`)
8. Rename `rsync_code_v2` → `rsync_code` at definition and call site

---

## Test Suite Changes

- `test_config.py` (renamed): update all `from chester.config_v2 import` → `from chester.config import`
- `test_run_exp.py` (renamed): remove tests that assert `ValueError` on deprecated mode strings (those code paths are gone); keep any remaining tests
- `test_integration.py`: update `from chester.config_v2 import` → `from chester.config import`

---

## Version Bump

`pyproject.toml`: `1.2.0` → `2.0.0`

This is a semver-correct breaking release. The public API surface changes are:
- `chester.slurm` module removed
- `chester.config` module semantics changed (now `config_v2` content, not the v1 shim)
- `query_yes_no()` removed
- `rsync_code()` signature changed (now takes explicit `project_path`, `rsync_include`, `rsync_exclude` args)
- All deprecated mode strings (`ec2`, `autobot`, `singularity`, `local_singularity`) removed from validation

---

## Parallel Execution Plan

Two agents work in `.worktrees/release/2.0.0` simultaneously on non-overlapping files:

**Agent 1** — deletions + `config_v2` rename  
**Agent 2** — code edits + test renames + version bump

Both agents make file edits only (no git operations). Main thread commits after both complete.

---

## Post-Implementation

```bash
uv run pytest tests/ -v    # verify all tests pass
uv build
uv publish
```
