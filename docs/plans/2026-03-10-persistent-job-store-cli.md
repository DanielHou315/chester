# Persistent Job Store + Chester CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the background-process auto-pull poller with a persistent per-job file store and an on-demand `chester pull-remote` CLI command that survives local reboots.

**Architecture:** Each submitted remote job writes a single JSON file (UUID-named) to `{PROJECT_PATH}/chester/auto_pull_jobs/`. The existing background poller (`_save_and_spawn_auto_pull`) is removed. A new `chester` top-level CLI consolidates all chester entry points; `chester pull-remote` reads the job store, checks status for each job, pulls done/done_orphaned jobs, skips failed ones (no pull), and deletes the job file on success.

**Tech Stack:** Python stdlib only (json, uuid, argparse, pathlib, subprocess); existing rsync-based pull logic in `auto_pull.py` reused unchanged.

---

## Design Notes

### Job store directory
`{config.PROJECT_PATH}/chester/auto_pull_jobs/`
- One file per job: `{uuid}.json`
- Written at submit time in `run_exp.py`
- Deleted by `chester pull-remote` after successful pull
- Failed jobs: status updated in-place to `"failed"`, NOT deleted, NOT pulled (user inspects manually)

### rsync trailing slash invariant (CRITICAL — do not change)
`pull_results()` in `auto_pull.py` already correctly uses:
```
rsync ... {host}:{remote_log_dir}/ {local_log_dir}/
```
Both source AND destination have trailing slashes. This syncs the **contents** of `remote_log_dir` into `local_log_dir`, not `local_log_dir/remote_basename/`. Never remove these trailing slashes.

### Behavior change from old design
| Situation | Old behavior | New behavior |
|-----------|-------------|--------------|
| Job done | Pull + mark manifest | Pull + delete job file |
| Job failed | Pull logs (bare) | Mark failed in file, skip pull |
| After reboot | Poller process dead — nothing | `chester pull-remote` reads store |
| Trigger | Automatic background process | Manual `chester pull-remote` |

### Existing `pull_result.py` bug
`pyproject.toml` declares `chester-pull = "chester.pull_result:main"` but `pull_result.py` has no `main()` function (only `if __name__ == "__main__":`). This entry point is broken. The new `chester pull` subcommand will be the correct replacement.

---

## Task 1: Job store module

**Files:**
- Create: `src/chester/job_store.py`
- Test: `tests/test_job_store.py`

### Step 1: Write failing tests

```python
# tests/test_job_store.py
import json
import pytest
from pathlib import Path
from chester.job_store import write_job_file, load_pending_jobs, delete_job_file, JOB_STATUS_PENDING


def test_write_job_file_creates_file(tmp_path):
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/logs/exp1',
        'local_log_dir': '/local/logs/exp1',
        'exp_name': 'test_exp',
        'exp_prefix': 'my_prefix',
        'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert job_id in files[0].name


def test_write_job_file_content(tmp_path):
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/logs/exp1',
        'local_log_dir': '/local/logs/exp1',
        'exp_name': 'test_exp',
        'exp_prefix': 'my_prefix',
        'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    file_path = tmp_path / f'{job_id}.json'
    data = json.loads(file_path.read_text())
    assert data['host'] == 'gl'
    assert data['job_id'] == job_id
    assert data['status'] == JOB_STATUS_PENDING


def test_load_pending_jobs_all(tmp_path):
    for i in range(3):
        write_job_file(tmp_path, {
            'host': 'gl', 'remote_log_dir': f'/r/{i}',
            'local_log_dir': f'/l/{i}', 'exp_name': f'exp{i}',
            'exp_prefix': 'prefix_a', 'status': JOB_STATUS_PENDING,
        })
    jobs = load_pending_jobs(tmp_path)
    assert len(jobs) == 3


def test_load_pending_jobs_prefix_filter(tmp_path):
    write_job_file(tmp_path, {
        'host': 'gl', 'remote_log_dir': '/r/1', 'local_log_dir': '/l/1',
        'exp_name': 'exp1', 'exp_prefix': 'alpha', 'status': JOB_STATUS_PENDING,
    })
    write_job_file(tmp_path, {
        'host': 'gl', 'remote_log_dir': '/r/2', 'local_log_dir': '/l/2',
        'exp_name': 'exp2', 'exp_prefix': 'beta', 'status': JOB_STATUS_PENDING,
    })
    jobs = load_pending_jobs(tmp_path, prefix='alpha')
    assert len(jobs) == 1
    assert jobs[0]['exp_prefix'] == 'alpha'


def test_load_pending_jobs_skips_non_pending(tmp_path):
    import uuid
    job_id = str(uuid.uuid4())
    path = tmp_path / f'{job_id}.json'
    path.write_text(json.dumps({'status': 'failed', 'job_id': job_id, 'exp_prefix': 'x'}))
    jobs = load_pending_jobs(tmp_path)
    assert len(jobs) == 0


def test_delete_job_file(tmp_path):
    job = {
        'host': 'gl', 'remote_log_dir': '/r/1', 'local_log_dir': '/l/1',
        'exp_name': 'exp1', 'exp_prefix': 'pfx', 'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    delete_job_file(tmp_path, job_id)
    assert not (tmp_path / f'{job_id}.json').exists()


def test_delete_job_file_missing_is_noop(tmp_path):
    delete_job_file(tmp_path, 'nonexistent-uuid')  # should not raise


def test_load_pending_jobs_empty_dir(tmp_path):
    jobs = load_pending_jobs(tmp_path)
    assert jobs == []
```

### Step 2: Run to verify failure
```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
uv run pytest tests/test_job_store.py -v
```
Expected: `ModuleNotFoundError: No module named 'chester.job_store'`

### Step 3: Implement `src/chester/job_store.py`

```python
"""
Chester job store — persistent per-job files for auto-pull tracking.

Each remote job submitted with auto_pull=True writes one JSON file:
    {job_store_dir}/{uuid}.json

The file is deleted after a successful pull.
Failed jobs are marked in-place (status='failed') but never pulled.
"""
import json
import uuid
from pathlib import Path
from typing import Optional

JOB_STATUS_PENDING = 'pending'
JOB_STATUS_FAILED = 'failed'


def get_default_job_store_dir() -> Path:
    """Return {PROJECT_PATH}/chester/auto_pull_jobs/."""
    from chester import config
    return Path(config.PROJECT_PATH) / 'chester' / 'auto_pull_jobs'


def write_job_file(job_store_dir: Path, job: dict) -> str:
    """
    Write a job dict to a new UUID-named file in job_store_dir.

    A 'job_id' key is added automatically. Returns the job_id (UUID string).
    """
    job_store_dir = Path(job_store_dir)
    job_store_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    data = dict(job)
    data['job_id'] = job_id
    (job_store_dir / f'{job_id}.json').write_text(json.dumps(data, indent=2))
    return job_id


def load_pending_jobs(job_store_dir: Path, prefix: Optional[str] = None) -> list:
    """
    Load all jobs with status='pending' from job_store_dir.

    Args:
        job_store_dir: Directory containing per-job JSON files.
        prefix: If given, only return jobs where exp_prefix == prefix.
    """
    job_store_dir = Path(job_store_dir)
    if not job_store_dir.exists():
        return []
    jobs = []
    for path in sorted(job_store_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get('status') != JOB_STATUS_PENDING:
            continue
        if prefix is not None and data.get('exp_prefix') != prefix:
            continue
        jobs.append(data)
    return jobs


def mark_job_failed(job_store_dir: Path, job_id: str):
    """Update job file status to 'failed' in-place. No pull will be attempted."""
    path = Path(job_store_dir) / f'{job_id}.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data['status'] = JOB_STATUS_FAILED
    path.write_text(json.dumps(data, indent=2))


def delete_job_file(job_store_dir: Path, job_id: str):
    """Delete a job file. No-op if already gone."""
    path = Path(job_store_dir) / f'{job_id}.json'
    if path.exists():
        path.unlink()
```

### Step 4: Run tests to verify passing
```bash
uv run pytest tests/test_job_store.py -v
```
Expected: all 8 tests PASS

### Step 5: Commit
```bash
git add src/chester/job_store.py tests/test_job_store.py
git commit -m "feat(job_store): add persistent per-job file store for auto-pull tracking"
```

---

## Task 2: Wire job store into `run_exp.py`

**Files:**
- Modify: `src/chester/run_exp.py`

**What to change:** Replace the 3 manifest functions (`_init_auto_pull_manifest`, `_init_auto_pull_manifest_v2`, `_add_job_to_manifest`) and the poller spawn (`_save_and_spawn_auto_pull`) with a single `_register_job_for_pull()` call. Remove the module-level globals `_auto_pull_manifest_path` and `_auto_pull_jobs`.

### Step 1: Write failing test

Add this class to `tests/test_run_exp_v2.py`:

```python
class TestAutoRegisterJobStore:
    """run_experiment_lite with auto_pull=True writes a job file per submitted job."""

    def test_remote_job_writes_job_file(self, tmp_path, monkeypatch):
        from chester.job_store import load_pending_jobs
        monkeypatch.setattr(
            'chester.run_exp.get_default_job_store_dir',
            lambda: tmp_path / 'jobs'
        )
        # Use an existing remote-submit test fixture or mock the backend submit
        # to confirm a .json file is written without actually SSHing anywhere.
        # See existing test patterns in test_run_exp_v2.py for how to mock
        # backend submission.
        #
        # Minimum check after the call:
        jobs = load_pending_jobs(tmp_path / 'jobs')
        assert len(jobs) == 1
        assert jobs[0]['status'] == 'pending'
        assert jobs[0]['host'] is not None
```

Run first to confirm it fails, then implement.

### Step 2: Add `_register_job_for_pull` to `run_exp.py`

```python
def _register_job_for_pull(
    host: str,
    remote_log_dir: str,
    local_log_dir: str,
    exp_name: str,
    exp_prefix: str,
    extra_pull_dirs: list = None,
    slurm_job_id: int = None,
):
    """Write a single job file to the persistent job store."""
    from chester.job_store import write_job_file, get_default_job_store_dir, JOB_STATUS_PENDING
    job_store_dir = get_default_job_store_dir()
    job = {
        'host': host,
        'remote_log_dir': remote_log_dir,
        'local_log_dir': local_log_dir,
        'exp_name': exp_name,
        'exp_prefix': exp_prefix,
        'extra_pull_dirs': extra_pull_dirs or [],
        'status': JOB_STATUS_PENDING,
    }
    if slurm_job_id is not None:
        job['slurm_job_id'] = slurm_job_id
    job_id = write_job_file(job_store_dir, job)
    print(f'[chester] Registered job for pull: {exp_name} -> {job_store_dir}/{job_id}.json')
```

### Step 3: Update call sites in `run_experiment_lite`

**Remove section 9** (manifest init — no longer needed):
```python
# REMOVE this block:
if is_remote and first_variant and auto_pull:
    _init_auto_pull_manifest_v2(exp_prefix, mode, cfg_log_dir)
```

**Replace `_add_job_to_manifest` call** (inside remote submission loop):
```python
# OLD:
if auto_pull and not dry:
    slurm_job_id = submit_result if backend_config.type == "slurm" else None
    _add_job_to_manifest(
        host=backend_config.host,
        remote_log_dir=remote_log_dir,
        local_log_dir=local_log_dir,
        exp_name=task['exp_name'],
        extra_pull_dirs=resolved_extra_pull_dirs,
        slurm_job_id=slurm_job_id,
    )

# NEW:
if auto_pull and not dry:
    slurm_job_id = submit_result if backend_config.type == "slurm" else None
    _register_job_for_pull(
        host=backend_config.host,
        remote_log_dir=remote_log_dir,
        local_log_dir=local_log_dir,
        exp_name=task['exp_name'],
        exp_prefix=exp_prefix,
        extra_pull_dirs=resolved_extra_pull_dirs,
        slurm_job_id=slurm_job_id,
    )
```

**Remove section 11** (poller spawn):
```python
# REMOVE:
if is_remote and last_variant and auto_pull:
    _save_and_spawn_auto_pull(dry=dry, poll_interval=auto_pull_interval)
```

**Remove `auto_pull_interval` parameter** from `run_experiment_lite` signature and docstring.

**Remove module-level globals** at top of file:
```python
# REMOVE:
_auto_pull_manifest_path = None
_auto_pull_jobs = []
```

**Remove the 4 old functions**: `_init_auto_pull_manifest`, `_init_auto_pull_manifest_v2`, `_add_job_to_manifest`, `_save_and_spawn_auto_pull`.

### Step 4: Run full test suite
```bash
uv run pytest tests/ -v --ignore=tests/live_slurm_validation.py
```
Expected: all existing tests pass; no regressions.

### Step 5: Commit
```bash
git add src/chester/run_exp.py tests/test_run_exp_v2.py
git commit -m "refactor(run_exp): replace manifest+poller with persistent job store registration"
```

---

## Task 3: Add `execute_pull_for_job` to `auto_pull.py`

**Files:**
- Modify: `src/chester/auto_pull.py`
- Modify: `tests/test_auto_pull.py`

### Step 1: Write failing tests

Add to `tests/test_auto_pull.py`:

```python
from chester.auto_pull import (
    check_slurm_job_status, check_job_status, execute_pull_for_job
)

class TestExecutePullForJob:
    def _make_job(self, **overrides):
        job = {
            'job_id': 'test-uuid-1234',
            'host': 'gl',
            'remote_log_dir': '/remote/logs/exp1',
            'local_log_dir': '/local/logs/exp1',
            'exp_name': 'test_exp',
            'exp_prefix': 'pfx',
            'extra_pull_dirs': [],
            'status': 'pending',
        }
        job.update(overrides)
        return job

    def test_done_returns_pulled(self):
        job = self._make_job()
        with mock.patch('chester.auto_pull.check_job_status', return_value='done'), \
             mock.patch('chester.auto_pull.pull_results', return_value=True), \
             mock.patch('chester.auto_pull.pull_extra_dirs', return_value=True):
            assert execute_pull_for_job(job) == 'pulled'

    def test_done_pull_failure_returns_pull_failed(self):
        job = self._make_job()
        with mock.patch('chester.auto_pull.check_job_status', return_value='done'), \
             mock.patch('chester.auto_pull.pull_results', return_value=False):
            assert execute_pull_for_job(job) == 'pull_failed'

    def test_failed_returns_failed_no_pull(self):
        job = self._make_job()
        with mock.patch('chester.auto_pull.check_job_status', return_value='failed'), \
             mock.patch('chester.auto_pull.pull_results') as mock_pull:
            assert execute_pull_for_job(job) == 'failed'
        mock_pull.assert_not_called()

    def test_running_returns_running_no_pull(self):
        job = self._make_job()
        with mock.patch('chester.auto_pull.check_job_status', return_value='running'), \
             mock.patch('chester.auto_pull.pull_results') as mock_pull:
            assert execute_pull_for_job(job) == 'running'
        mock_pull.assert_not_called()

    def test_done_orphans_kills_and_pulls(self):
        job = self._make_job()
        with mock.patch('chester.auto_pull.check_job_status', return_value='done_orphans'), \
             mock.patch('chester.auto_pull.get_remote_pid', return_value=12345), \
             mock.patch('chester.auto_pull.kill_process_tree') as mock_kill, \
             mock.patch('chester.auto_pull.pull_results', return_value=True), \
             mock.patch('chester.auto_pull.pull_extra_dirs', return_value=True):
            assert execute_pull_for_job(job) == 'pulled'
        mock_kill.assert_called_once_with('gl', 12345)
```

### Step 2: Run to verify failure
```bash
uv run pytest tests/test_auto_pull.py::TestExecutePullForJob -v
```
Expected: `ImportError: cannot import name 'execute_pull_for_job'`

### Step 3: Add `execute_pull_for_job` to `auto_pull.py`

Add after `pull_extra_dirs()`:

```python
def execute_pull_for_job(job: dict, bare: bool = False) -> str:
    """
    Check status of one job and pull if complete.

    The caller (CLI) is responsible for deleting or updating the job file
    based on the returned status.

    rsync path safety: pull_results() appends trailing slashes to both
    remote_log_dir and local_log_dir so rsync syncs directory *contents*
    rather than creating a subdirectory. This must not be changed.

    Returns: 'pulled', 'pull_failed', 'failed', or 'running'.
    """
    host = job['host']
    remote_log_dir = job['remote_log_dir']
    local_log_dir = job['local_log_dir']
    exp_name = job.get('exp_name', 'unknown')
    extra_pull_dirs = job.get('extra_pull_dirs', [])

    status = check_job_status(job)

    if status == 'done':
        print(f'[chester] Job completed: {exp_name} on {host}')
        if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
            pull_extra_dirs(host, extra_pull_dirs, bare=bare)
            return 'pulled'
        return 'pull_failed'

    if status == 'done_orphans':
        print(f'[chester] Job completed with orphan processes: {exp_name} on {host}')
        pid = get_remote_pid(host, remote_log_dir)
        if pid:
            kill_process_tree(host, pid)
        if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
            pull_extra_dirs(host, extra_pull_dirs, bare=bare)
            return 'pulled'
        return 'pull_failed'

    if status == 'failed':
        print(f'[chester] Job FAILED: {exp_name} on {host} — skipping pull')
        return 'failed'

    return 'running'
```

**Remove** from `auto_pull.py` (no longer called by anything):
- `poll_and_pull()` function
- `load_manifest()` function
- `save_manifest()` function
- `main()` function and `argparse` block at the bottom

**Keep**: all status-check and pull functions — they are used by `execute_pull_for_job`.

### Step 4: Run tests
```bash
uv run pytest tests/test_auto_pull.py -v
```
Expected: all tests PASS (new + existing).

### Step 5: Commit
```bash
git add src/chester/auto_pull.py tests/test_auto_pull.py
git commit -m "refactor(auto_pull): extract execute_pull_for_job, remove polling loop"
```

---

## Task 4: New `chester` CLI

**Files:**
- Create: `src/chester/cli.py`
- Create: `tests/test_cli.py`

### Step 1: Write failing tests

```python
# tests/test_cli.py
import json
import pytest
from unittest import mock
from pathlib import Path


class TestPullRemoteCommand:

    def _pending_job(self, job_id='abc-123', exp_prefix='pfx'):
        return {
            'job_id': job_id,
            'host': 'gl',
            'remote_log_dir': '/remote/logs/exp1',
            'local_log_dir': '/local/logs/exp1',
            'exp_name': 'test_exp',
            'exp_prefix': exp_prefix,
            'extra_pull_dirs': [],
            'status': 'pending',
        }

    def test_no_jobs(self, tmp_path, capsys):
        from chester.cli import cmd_pull_remote
        cmd_pull_remote(job_store_dir=tmp_path, prefix=None, bare=False, dry_run=False)
        assert 'No pending jobs' in capsys.readouterr().out

    def test_deletes_file_on_pulled(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='pulled'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        assert not (tmp_path / f'{job_id}.json').exists()

    def test_marks_failed_file_no_delete(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file, load_pending_jobs
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='failed'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        path = tmp_path / f'{job_id}.json'
        assert path.exists()
        assert json.loads(path.read_text())['status'] == 'failed'
        assert load_pending_jobs(tmp_path) == []

    def test_keeps_file_on_running(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='running'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        assert (tmp_path / f'{job_id}.json').exists()

    def test_prefix_filter(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        write_job_file(tmp_path, self._pending_job(exp_prefix='alpha'))
        write_job_file(tmp_path, self._pending_job(exp_prefix='beta'))
        calls = []
        def fake_pull(job, bare=False):
            calls.append(job['exp_prefix'])
            return 'running'
        with mock.patch('chester.cli.execute_pull_for_job', side_effect=fake_pull):
            cmd_pull_remote(tmp_path, prefix='alpha', bare=False, dry_run=False)
        assert calls == ['alpha']

    def test_dry_run_no_changes(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job') as mock_pull:
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=True)
        mock_pull.assert_not_called()
        assert (tmp_path / f'{job_id}.json').exists()
```

### Step 2: Run to verify failure
```bash
uv run pytest tests/test_cli.py -v
```
Expected: `ModuleNotFoundError: No module named 'chester.cli'`

### Step 3: Implement `src/chester/cli.py`

```python
"""
Chester CLI — unified command-line interface.

Entry point: `chester <subcommand> [options]`

Subcommands:
    pull-remote   Check all pending remote jobs and pull completed ones.
    pull          Pull a specific remote folder (replaces legacy chester-pull).
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from chester.auto_pull import execute_pull_for_job
from chester.job_store import (
    get_default_job_store_dir,
    load_pending_jobs,
    delete_job_file,
    mark_job_failed,
)


def cmd_pull_remote(
    job_store_dir: Path,
    prefix: Optional[str],
    bare: bool,
    dry_run: bool,
):
    """
    Check all pending jobs and pull completed ones.

    pulled      -> delete job file
    failed      -> mark status='failed', do NOT pull
    pull_failed -> leave as pending (will retry next run)
    running     -> leave as pending
    """
    jobs = load_pending_jobs(job_store_dir, prefix=prefix)

    if not jobs:
        filter_msg = f" (prefix='{prefix}')" if prefix else ""
        print(f'[chester] No pending jobs found in {job_store_dir}{filter_msg}')
        return

    print(f'[chester] Checking {len(jobs)} pending job(s)...')

    if dry_run:
        for job in jobs:
            print(f'  [dry-run] Would check: {job["exp_name"]} on {job["host"]}')
        return

    counts = {'pulled': 0, 'failed': 0, 'pull_failed': 0, 'running': 0}

    for job in jobs:
        job_id = job['job_id']
        result = execute_pull_for_job(job, bare=bare)
        counts[result] = counts.get(result, 0) + 1

        if result == 'pulled':
            delete_job_file(job_store_dir, job_id)
        elif result == 'failed':
            mark_job_failed(job_store_dir, job_id)

    print(
        f'[chester] Done: {counts["pulled"]} pulled, '
        f'{counts["running"]} still running, '
        f'{counts["failed"]} failed (not pulled), '
        f'{counts["pull_failed"]} pull errors (will retry)'
    )


def cmd_pull(host: str, folder: str, bare: bool, dry: bool):
    """
    Pull a specific remote folder to local (replaces legacy chester-pull).

    Uses rsync with trailing slashes on both ends so the remote directory
    contents are merged into the local directory, not nested inside it.
    """
    import os
    from chester import config

    folder = folder.rstrip('/')
    slash_pos = folder.rfind('/')
    if slash_pos != -1:
        local_dir = os.path.join('./data', host, folder[:slash_pos])
    else:
        local_dir = os.path.join('./data', host, folder)

    remote_data_dir = os.path.join(config.REMOTE_DIR[host], 'data', 'local', folder)

    # Trailing slash on source syncs contents into local_dir, not local_dir/basename/
    cmd = [
        'rsync', '-avzh', '--delete', '--progress',
        f'{host}:{remote_data_dir}/', f'{local_dir}/',
    ]
    if bare:
        cmd += [
            '--exclude', '*.pkl', '--exclude', '*.png', '--exclude', '*.gif',
            '--exclude', '*.pth', '--exclude', '*.pt',
            '--include', '*.csv', '--include', '*.json',
        ]

    if dry:
        print(' '.join(cmd))
    else:
        result = subprocess.run(cmd)
        if result.returncode not in (0, 24):  # 24 = "some files vanished", harmless
            sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='chester',
        description='Chester ML experiment launcher CLI',
    )
    sub = parser.add_subparsers(dest='command', required=True)

    pr = sub.add_parser('pull-remote', help='Check pending remote jobs and pull completed ones')
    pr.add_argument('--prefix', type=str, default=None,
                    help='Only process jobs matching this exp_prefix')
    pr.add_argument('--bare', action='store_true',
                    help='Exclude large files (*.pkl, *.pth, etc.) when pulling')
    pr.add_argument('--dry-run', action='store_true',
                    help='Print what would be checked without doing anything')

    p = sub.add_parser('pull', help='Pull a specific remote folder (replaces chester-pull)')
    p.add_argument('host', type=str, help='Remote host name (key in chester.yaml remote_dir)')
    p.add_argument('folder', type=str, help='Remote folder path relative to remote data dir')
    p.add_argument('--bare', action='store_true', help='Exclude large files')
    p.add_argument('--dry', action='store_true', help='Print command without executing')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == 'pull-remote':
        cmd_pull_remote(
            job_store_dir=get_default_job_store_dir(),
            prefix=args.prefix,
            bare=args.bare,
            dry_run=args.dry_run,
        )
    elif args.command == 'pull':
        cmd_pull(host=args.host, folder=args.folder, bare=args.bare, dry=args.dry)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
```

### Step 4: Run tests
```bash
uv run pytest tests/test_cli.py -v
```
Expected: all tests PASS

### Step 5: Commit
```bash
git add src/chester/cli.py tests/test_cli.py
git commit -m "feat(cli): add chester CLI with pull-remote and pull subcommands"
```

---

## Task 5: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`

### Step 1: Replace entry points

Change:
```toml
# OLD:
[project.scripts]
chester-pull = "chester.pull_result:main"

# NEW:
[project.scripts]
chester = "chester.cli:main"
```

> `pull_result.py` had a broken entry point (no `main()` function). It is superseded by `chester pull`. The module file is left in place but the broken script entry is removed.

### Step 2: Reinstall and smoke test
```bash
uv sync
chester --help
chester pull-remote --help
chester pull --help
```
Expected: help text shown for all three.

### Step 3: Commit
```bash
git add pyproject.toml
git commit -m "chore: replace chester-pull entry point with unified chester CLI"
```

---

## Task 6: Full suite verification

### Step 1: Run all tests
```bash
uv run pytest tests/ -v --ignore=tests/live_slurm_validation.py
```
Expected: all tests pass.

### Step 2: Check for old function references
```bash
grep -rn "_save_and_spawn_auto_pull\|_init_auto_pull_manifest\|_add_job_to_manifest\|poll_and_pull\|load_manifest\|save_manifest" src/chester/
```
Expected: no matches.

### Step 3: Verify rsync trailing slashes in auto_pull.py
```bash
grep -n "remote_log_dir" src/chester/auto_pull.py
```
Confirm both occurrences look like:
```
f"{host}:{remote_log_dir}/", f"{local_log_dir}/"
```

### Step 4: Commit cleanup if needed
```bash
git add -p
git commit -m "chore: final cleanup after job store refactor"
```

---

## Summary of changes

| File | Action |
|------|--------|
| `src/chester/job_store.py` | **New** — per-job file store |
| `src/chester/cli.py` | **New** — `chester` CLI with subcommands |
| `src/chester/run_exp.py` | **Modified** — replace manifest/poller with `_register_job_for_pull` |
| `src/chester/auto_pull.py` | **Modified** — add `execute_pull_for_job`, remove polling loop |
| `pyproject.toml` | **Modified** — replace `chester-pull` with `chester` entry |
| `tests/test_job_store.py` | **New** |
| `tests/test_cli.py` | **New** |
| `tests/test_auto_pull.py` | **Modified** — add `TestExecutePullForJob` |
| `tests/test_run_exp_v2.py` | **Modified** — add `TestAutoRegisterJobStore` |

## Files NOT touched

| File | Reason |
|------|--------|
| `src/chester/pull_result.py` | Superseded; kept for any direct module usage |
| `src/chester/backends/*.py` | No change needed |
| `src/chester/config.py` | No change needed |
| `src/chester/slurm.py` | No change needed |
