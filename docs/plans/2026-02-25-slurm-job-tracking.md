# SLURM Job ID Tracking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Capture SLURM job IDs from `sbatch` output and use `sacct` to detect job failure, replacing the current blind `.done`-marker-only polling for SLURM jobs.

**Architecture:** `slurm.py:submit()` parses the job ID from `sbatch` stdout and returns it. The manifest gains a `slurm_job_id` field. `auto_pull.py:check_job_status()` detects SLURM jobs in the manifest and queries `sacct -j <id>` for terminal states (FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED) to detect failure, while still using `.done` for success. `run_exp.py` passes the returned job ID through to the manifest.

**Tech Stack:** Python stdlib (re, subprocess), SLURM CLI (`sacct`), existing chester test infrastructure (pytest, unittest.mock)

---

### Task 1: Parse SLURM job ID in `slurm.py:submit()`

**Files:**
- Modify: `src/chester/backends/slurm.py:180-193`
- Test: `tests/test_backend_slurm.py`

**Step 1: Write the failing test**

Add to `tests/test_backend_slurm.py`:

```python
def test_slurm_submit_returns_job_id(tmp_path):
    """submit() parses job ID from sbatch output and returns it."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }

    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Submitted batch job 98765432\n", stderr=""
        )
        job_id = backend.submit(task, script_content="#!/bin/bash\necho hi\n", dry=False)

    assert job_id == 98765432


def test_slurm_submit_returns_none_on_dry_run():
    """Dry run returns None."""
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    result = backend.submit(task, script_content="#!/bin/bash\necho hi\n", dry=True)
    assert result is None


def test_slurm_submit_returns_none_when_job_id_unparseable(tmp_path):
    """If sbatch output is unexpected, return None instead of crashing."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }

    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Some unexpected output\n", stderr=""
        )
        job_id = backend.submit(task, script_content="#!/bin/bash\necho hi\n", dry=False)

    assert job_id is None
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_backend_slurm.py::test_slurm_submit_returns_job_id test_backend_slurm.py::test_slurm_submit_returns_none_on_dry_run test_backend_slurm.py::test_slurm_submit_returns_none_when_job_id_unparseable -v`
Expected: FAIL (submit currently returns None implicitly)

**Step 3: Implement — parse job ID from sbatch stdout**

In `src/chester/backends/slurm.py`, add `import re` at top, then modify `submit()` (around line 180-193):

```python
        # 4. Submit via sbatch
        print(f"[chester] Submitting SLURM job on {host}: {exp_name}")
        print(f"[chester] Remote script: {remote_script}")
        result = subprocess.run(
            ["ssh", host, f"sbatch {shlex.quote(remote_script)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[chester] sbatch failed: {result.stderr.strip()}")
            raise RuntimeError(
                f"sbatch failed on {host}: {result.stderr.strip()}"
            )
        print(f"[chester] {result.stdout.strip()}")

        # Parse job ID from "Submitted batch job <id>"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = int(match.group(1))
            print(f"[chester] SLURM job ID: {job_id}")
            return job_id
        else:
            print(f"[chester] Warning: could not parse SLURM job ID from: {result.stdout.strip()}")
            return None
```

Also update the `dry` early return to explicitly `return None`.

**Step 4: Run tests to verify they pass**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_backend_slurm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/backends/slurm.py tests/test_backend_slurm.py
git commit -m "feat: parse SLURM job ID from sbatch output in submit()"
```

---

### Task 2: Write SLURM job ID to remote `.chester_slurm_job_id`

**Files:**
- Modify: `src/chester/backends/slurm.py:submit()` (after parsing job ID)
- Test: `tests/test_backend_slurm.py`

**Step 1: Write the failing test**

```python
def test_slurm_submit_writes_job_id_to_remote(tmp_path):
    """submit() writes job ID to .chester_slurm_job_id on remote."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }

    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Submitted batch job 98765432\n", stderr=""
        )
        backend.submit(task, script_content="#!/bin/bash\necho hi\n", dry=False)

    # The 4th call should write the job ID file (after mkdir, scp, sbatch)
    calls = mock_run.call_args_list
    job_id_call = calls[3]  # 0=mkdir, 1=scp, 2=sbatch, 3=write job id
    cmd = job_id_call[0][0]
    assert "ssh" in cmd[0]
    assert "98765432" in " ".join(cmd)
    assert ".chester_slurm_job_id" in " ".join(cmd)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_backend_slurm.py::test_slurm_submit_writes_job_id_to_remote -v`
Expected: FAIL (IndexError — only 3 subprocess calls currently)

**Step 3: Implement — write job ID to remote after sbatch**

In `slurm.py:submit()`, after parsing the job ID successfully, add:

```python
        if match:
            job_id = int(match.group(1))
            print(f"[chester] SLURM job ID: {job_id}")
            # Write job ID to remote for auto-pull tracking
            job_id_file = os.path.join(log_dir, ".chester_slurm_job_id")
            subprocess.run(
                ["ssh", host, f"echo {job_id} > {shlex.quote(job_id_file)}"],
                check=False,  # Non-fatal if this fails
            )
            return job_id
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_backend_slurm.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/backends/slurm.py tests/test_backend_slurm.py
git commit -m "feat: write SLURM job ID to .chester_slurm_job_id on remote"
```

---

### Task 3: Add `slurm_job_id` to the auto-pull manifest

**Files:**
- Modify: `src/chester/run_exp.py:188-201` (`_add_job_to_manifest`)
- Modify: `src/chester/run_exp.py:839-852` (submit call site — capture return value)
- Test: `tests/test_run_exp_v2.py` (or inline verification)

**Step 1: Write the failing test**

Add to `tests/test_run_exp_v2.py` (or a new section):

```python
def test_add_job_to_manifest_includes_slurm_job_id():
    """Manifest entry includes slurm_job_id when provided."""
    from chester.run_exp import _add_job_to_manifest, _auto_pull_jobs
    import chester.run_exp as run_exp_module

    # Reset global state
    run_exp_module._auto_pull_jobs = []

    _add_job_to_manifest(
        host="gl",
        remote_log_dir="/remote/data/exp1",
        local_log_dir="/local/data/exp1",
        exp_name="test_exp",
        slurm_job_id=12345678,
    )

    assert len(run_exp_module._auto_pull_jobs) == 1
    job = run_exp_module._auto_pull_jobs[0]
    assert job['slurm_job_id'] == 12345678

    # Clean up
    run_exp_module._auto_pull_jobs = []


def test_add_job_to_manifest_omits_slurm_job_id_when_none():
    """Manifest entry has no slurm_job_id key when not provided."""
    from chester.run_exp import _add_job_to_manifest
    import chester.run_exp as run_exp_module

    run_exp_module._auto_pull_jobs = []

    _add_job_to_manifest(
        host="armdual",
        remote_log_dir="/remote/data/exp1",
        local_log_dir="/local/data/exp1",
        exp_name="test_exp",
    )

    job = run_exp_module._auto_pull_jobs[0]
    assert job.get('slurm_job_id') is None

    run_exp_module._auto_pull_jobs = []
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_run_exp_v2.py::test_add_job_to_manifest_includes_slurm_job_id -v`
Expected: FAIL (TypeError — `_add_job_to_manifest` doesn't accept `slurm_job_id`)

**Step 3: Implement**

In `src/chester/run_exp.py`, modify `_add_job_to_manifest()`:

```python
def _add_job_to_manifest(host: str, remote_log_dir: str, local_log_dir: str,
                         exp_name: str, extra_pull_dirs: list = None,
                         slurm_job_id: int = None):
    """Add a job to the auto-pull manifest."""
    global _auto_pull_jobs
    entry = {
        'host': host,
        'remote_log_dir': remote_log_dir,
        'local_log_dir': local_log_dir,
        'exp_name': exp_name,
        'extra_pull_dirs': extra_pull_dirs or [],
        'pid_file': os.path.join(remote_log_dir, '.chester_pid'),
        'status': 'pending',
        'submitted_at': datetime.datetime.now().isoformat()
    }
    if slurm_job_id is not None:
        entry['slurm_job_id'] = slurm_job_id
    _auto_pull_jobs.append(entry)
```

Then modify the call site in `run_experiment_lite()` (around line 839-852) to capture the return value from `backend.submit()` and pass it through:

```python
            # Submit via backend
            submit_result = backend.submit(backend_task, script_content, dry=dry)

            # Add job to auto-pull manifest
            if auto_pull and not dry:
                slurm_job_id = submit_result if backend_config.type == "slurm" else None
                _add_job_to_manifest(
                    host=backend_config.host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task.get('exp_name', ''),
                    extra_pull_dirs=_resolve_extra_pull_dirs_v2(
                        extra_pull_dirs, project_path, backend_config.remote_dir
                    ),
                    slurm_job_id=slurm_job_id,
                )
```

**Important:** The submit call must be moved **before** the manifest call so that the job ID is available. Currently manifest is added before submit — swap the order.

**Step 4: Run tests**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_run_exp_v2.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/run_exp.py tests/test_run_exp_v2.py
git commit -m "feat: add slurm_job_id to auto-pull manifest"
```

---

### Task 4: Add `check_slurm_job_status()` to `auto_pull.py`

**Files:**
- Modify: `src/chester/auto_pull.py` (new function)
- Test: `tests/test_auto_pull.py` (new file)

**Step 1: Write the failing tests**

Create `tests/test_auto_pull.py`:

```python
"""Tests for auto_pull SLURM job status checking."""
import unittest.mock as mock
import pytest

from chester.auto_pull import check_slurm_job_status


def test_slurm_completed():
    """sacct COMPLETED → 'completed'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="COMPLETED\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "completed"


def test_slurm_failed():
    """sacct FAILED → 'failed'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="FAILED\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "failed"


def test_slurm_timeout():
    """sacct TIMEOUT → 'failed'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="TIMEOUT\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "failed"


def test_slurm_oom():
    """sacct OUT_OF_MEMORY → 'failed'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="OUT_OF_MEMORY\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "failed"


def test_slurm_cancelled():
    """sacct CANCELLED → 'failed'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="CANCELLED by 12345\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "failed"


def test_slurm_running():
    """sacct RUNNING → 'running'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="RUNNING\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "running"


def test_slurm_pending():
    """sacct PENDING → 'running'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="PENDING\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "running"


def test_slurm_ssh_timeout():
    """SSH timeout → 'unknown' (conservative)."""
    import subprocess as sp
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.side_effect = sp.TimeoutExpired(cmd="ssh", timeout=30)
        assert check_slurm_job_status("gl", 12345) == "unknown"


def test_slurm_empty_output():
    """Empty sacct output → 'unknown'."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "unknown"


def test_slurm_multi_line_sacct():
    """sacct may return multiple lines (job + job steps); use first line."""
    with mock.patch("chester.auto_pull.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="COMPLETED\nCOMPLETED\nCOMPLETED\n", stderr=""
        )
        assert check_slurm_job_status("gl", 12345) == "completed"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_auto_pull.py -v`
Expected: FAIL (ImportError — `check_slurm_job_status` doesn't exist)

**Step 3: Implement `check_slurm_job_status()`**

Add to `src/chester/auto_pull.py`:

```python
# SLURM states that mean the job is terminally dead
_SLURM_FAILED_STATES = frozenset({
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
})
_SLURM_RUNNING_STATES = frozenset({
    "RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "CONFIGURING",
})


def check_slurm_job_status(host: str, slurm_job_id: int) -> str:
    """Query SLURM job status via sacct.

    Args:
        host: Remote host to SSH into.
        slurm_job_id: The SLURM job ID.

    Returns:
        'completed': Job finished successfully.
        'failed': Job terminated with an error state.
        'running': Job is still active or queued.
        'unknown': Could not determine status.
    """
    cmd = [
        "ssh", host,
        f"sacct -j {slurm_job_id} --format=State --noheader -P 2>/dev/null"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        # sacct may return multiple lines (one per job step); use the first
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return "unknown"

        state = lines[0].split()[0]  # Handle "CANCELLED by <uid>"

        if state == "COMPLETED":
            return "completed"
        elif state in _SLURM_FAILED_STATES:
            return "failed"
        elif state in _SLURM_RUNNING_STATES:
            return "running"
        else:
            print(f"[auto_pull] Unknown SLURM state '{state}' for job {slurm_job_id}")
            return "unknown"

    except subprocess.TimeoutExpired:
        print(f"[auto_pull] SSH timeout querying sacct for job {slurm_job_id} on {host}")
        return "unknown"
    except Exception as e:
        print(f"[auto_pull] Error querying sacct for job {slurm_job_id} on {host}: {e}")
        return "unknown"
```

**Step 4: Run tests**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_auto_pull.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/chester/auto_pull.py tests/test_auto_pull.py
git commit -m "feat: add check_slurm_job_status() using sacct"
```

---

### Task 5: Integrate SLURM status into `check_job_status()`

**Files:**
- Modify: `src/chester/auto_pull.py:163-194` (`check_job_status`)
- Test: `tests/test_auto_pull.py` (add integration tests)

**Step 1: Write the failing tests**

Add to `tests/test_auto_pull.py`:

```python
from chester.auto_pull import check_job_status


def test_check_job_status_slurm_failed_detected():
    """SLURM job that failed (no .done, sacct says FAILED) → 'failed'."""
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/data/exp1',
        'slurm_job_id': 12345,
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
         mock.patch("chester.auto_pull.check_slurm_job_status", return_value="failed"):
        assert check_job_status(job) == "failed"


def test_check_job_status_slurm_running():
    """SLURM job still running (no .done, sacct says RUNNING) → 'running'."""
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/data/exp1',
        'slurm_job_id': 12345,
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
         mock.patch("chester.auto_pull.check_slurm_job_status", return_value="running"):
        assert check_job_status(job) == "running"


def test_check_job_status_slurm_done_marker_present():
    """.done exists → 'done' regardless of sacct state."""
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/data/exp1',
        'slurm_job_id': 12345,
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=True):
        assert check_job_status(job) == "done"


def test_check_job_status_slurm_completed_no_done_yet():
    """sacct COMPLETED but no .done yet → 'running' (give it a moment)."""
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/data/exp1',
        'slurm_job_id': 12345,
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
         mock.patch("chester.auto_pull.check_slurm_job_status", return_value="completed"):
        assert check_job_status(job) == "running"


def test_check_job_status_ssh_still_uses_pid():
    """SSH jobs (no slurm_job_id) still use PID-based tracking."""
    job = {
        'host': 'armdual',
        'remote_log_dir': '/remote/data/exp1',
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
         mock.patch("chester.auto_pull.get_remote_pid", return_value=42), \
         mock.patch("chester.auto_pull.check_process_running", return_value=True):
        assert check_job_status(job) == "running"


def test_check_job_status_slurm_sacct_unknown_fallback():
    """If sacct returns 'unknown', treat as running (conservative)."""
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/data/exp1',
        'slurm_job_id': 12345,
    }
    with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
         mock.patch("chester.auto_pull.check_slurm_job_status", return_value="unknown"):
        assert check_job_status(job) == "running"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_auto_pull.py::test_check_job_status_slurm_failed_detected -v`
Expected: FAIL (current `check_job_status` doesn't check `slurm_job_id`)

**Step 3: Implement — update `check_job_status()` to handle SLURM jobs**

Replace `check_job_status()` in `auto_pull.py`:

```python
def check_job_status(job: dict) -> str:
    """
    Check job status based on .done marker and process/SLURM state.

    For SLURM jobs (job has 'slurm_job_id'):
        - .done exists → 'done'
        - sacct says FAILED/TIMEOUT/OOM/CANCELLED → 'failed'
        - sacct says RUNNING/PENDING → 'running'
        - sacct says COMPLETED but no .done → 'running' (race condition grace)
        - sacct unknown → 'running' (conservative)

    For SSH jobs (PID-based):
        - .done + process dead → 'done'
        - .done + process alive → 'done_orphans'
        - no .done + process dead → 'failed'
        - no .done + process alive → 'running'

    Returns:
        'running', 'done', 'done_orphans', or 'failed'
    """
    host = job['host']
    remote_log_dir = job['remote_log_dir']

    done = check_done_marker(host, remote_log_dir)

    # If .done exists, job succeeded — for both SLURM and SSH
    if done:
        # For SSH: check for orphan processes
        slurm_job_id = job.get('slurm_job_id')
        if slurm_job_id is None:
            pid = get_remote_pid(host, remote_log_dir)
            if pid is not None and check_process_running(host, pid):
                return 'done_orphans'
        return 'done'

    # No .done — check what's going on
    slurm_job_id = job.get('slurm_job_id')

    if slurm_job_id is not None:
        # SLURM job: use sacct
        slurm_status = check_slurm_job_status(host, slurm_job_id)
        if slurm_status == 'failed':
            return 'failed'
        # 'completed' without .done = race condition or script error after
        # training but before touch .done; treat as running briefly then
        # the next poll cycle will catch .done or declare failure.
        # 'running', 'unknown' → keep polling
        return 'running'

    else:
        # SSH job: use PID
        pid = get_remote_pid(host, remote_log_dir)
        if pid is None:
            return 'running'  # PID file not written yet
        running = check_process_running(host, pid)
        if not running:
            return 'failed'
        return 'running'
```

**Step 4: Run all auto_pull tests**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest test_auto_pull.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `cd /home/houhd/code/chester-overhaul/tests && uv run pytest -v`
Expected: All PASS (no regressions)

**Step 6: Commit**

```bash
git add src/chester/auto_pull.py tests/test_auto_pull.py
git commit -m "feat: integrate sacct-based SLURM failure detection into check_job_status()"
```

---

### Task 6: Real-world validation against `gl` SLURM host

**Files:**
- No code changes — test against real infrastructure

**Step 1: Verify SSH connectivity to gl**

Run: `ssh gl hostname`
Expected: prints the hostname of the Great Lakes login node

**Step 2: Verify sacct is available on gl**

Run: `ssh gl "sacct --version"`
Expected: prints sacct version (e.g., `slurm 23.x.x`)

**Step 3: Test sacct query format with a real job**

Run: `ssh gl "sacct -u $(ssh gl whoami) --format=JobID,State --noheader -P 2>/dev/null | head -5"`
Expected: lines like `12345678|COMPLETED` — confirms the format parser is correct

**Step 4: Submit a minimal test job and verify tracking**

Create a test script that will succeed quickly on gl, submit it via chester, and verify:
1. `sbatch` output is parsed correctly
2. `.chester_slurm_job_id` is written to remote
3. After job completes, `sacct` returns `COMPLETED`
4. `.done` marker appears

**Step 5: Submit a job designed to fail and verify failure detection**

Submit a script that exits with non-zero status, verify:
1. `sacct` returns `FAILED`
2. `check_job_status()` returns `'failed'` (not stuck in `'running'`)

---
