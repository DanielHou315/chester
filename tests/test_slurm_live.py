#!/usr/bin/env python
"""Live validation of SLURM job tracking against Great Lakes (gl).

Run: cd /home/houhd/code/chester-overhaul/tests && uv run python test_slurm_live.py

This is NOT a pytest test — it requires real SSH access to gl.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

# Add src to path so we can import chester
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chester.backends.base import BackendConfig, SlurmConfig
from chester.backends.slurm import SlurmBackend
from chester.auto_pull import check_slurm_job_status, check_done_marker


HOST = "gl"
REMOTE_BASE = "/home/houhd/chester_slurm_test"


def make_backend(remote_dir=REMOTE_BASE):
    config = BackendConfig(
        name="gl",
        type="slurm",
        host=HOST,
        remote_dir=remote_dir,
        slurm=SlurmConfig(
            partition="standard",
            time="00:02:00",
            nodes=1,
        ),
        modules=[],
        cuda_module=None,
    )
    project_config = {
        "project_path": "/tmp/chester_local",
        "package_manager": "python",
    }
    return SlurmBackend(config, project_config)


def test_success_job():
    """Submit a job that succeeds quickly, verify job ID tracking."""
    print("=" * 60)
    print(" TEST 1: Success job — should detect completion")
    print("=" * 60)

    backend = make_backend()
    remote_log = f"{REMOTE_BASE}/success_test"

    task = {
        "params": {
            "log_dir": remote_log,
            "exp_name": "chester_success_test",
        },
        "_local_log_dir": tempfile.mkdtemp(prefix="chester_test_"),
    }

    # Minimal success script
    script = f"""#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -o {remote_log}/slurm.out
#SBATCH -e {remote_log}/slurm.err
#SBATCH --job-name=chester_success_test
set -e
mkdir -p {remote_log}
echo "Chester test job running on $(hostname)"
sleep 5
touch {remote_log}/.done
echo "Done"
"""

    print(f"\n[1] Submitting success job to {HOST}...")
    job_id = backend.submit(task, script_content=script, dry=False)
    print(f"    Job ID: {job_id}")
    assert job_id is not None, "Failed to parse job ID!"

    # Verify .chester_slurm_job_id was written
    print(f"\n[2] Checking .chester_slurm_job_id on remote...")
    result = subprocess.run(
        ["ssh", HOST, f"cat {remote_log}/.chester_slurm_job_id"],
        capture_output=True, text=True, timeout=15,
    )
    remote_job_id = result.stdout.strip()
    print(f"    Remote job ID file: {remote_job_id}")
    assert remote_job_id == str(job_id), f"Mismatch: {remote_job_id} != {job_id}"

    # Poll sacct until job completes or fails
    print(f"\n[3] Polling sacct for job {job_id}...")
    for i in range(30):
        status = check_slurm_job_status(HOST, job_id)
        print(f"    [{i*5}s] sacct status: {status}")
        if status in ("completed", "failed"):
            break
        time.sleep(5)

    assert status == "completed", f"Expected 'completed', got '{status}'"
    print(f"    sacct reports: COMPLETED")

    # Verify .done marker
    print(f"\n[4] Checking .done marker...")
    done = check_done_marker(HOST, remote_log)
    print(f"    .done exists: {done}")
    assert done, ".done marker not found!"

    print(f"\n    PASS: Success job tracked correctly")
    return True


def test_failure_job():
    """Submit a job that fails, verify sacct detects it."""
    print("\n" + "=" * 60)
    print(" TEST 2: Failure job — should detect failure via sacct")
    print("=" * 60)

    backend = make_backend()
    remote_log = f"{REMOTE_BASE}/failure_test"

    task = {
        "params": {
            "log_dir": remote_log,
            "exp_name": "chester_failure_test",
        },
        "_local_log_dir": tempfile.mkdtemp(prefix="chester_test_"),
    }

    # Script that exits with error before .done
    script = f"""#!/bin/bash
#SBATCH --partition=standard
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH -o {remote_log}/slurm.out
#SBATCH -e {remote_log}/slurm.err
#SBATCH --job-name=chester_failure_test
set -e
mkdir -p {remote_log}
echo "Chester test job running on $(hostname)"
echo "About to fail..."
exit 1
touch {remote_log}/.done
"""

    print(f"\n[1] Submitting failure job to {HOST}...")
    job_id = backend.submit(task, script_content=script, dry=False)
    print(f"    Job ID: {job_id}")
    assert job_id is not None, "Failed to parse job ID!"

    # Poll sacct until job completes
    print(f"\n[2] Polling sacct for job {job_id}...")
    for i in range(30):
        status = check_slurm_job_status(HOST, job_id)
        print(f"    [{i*5}s] sacct status: {status}")
        if status in ("completed", "failed"):
            break
        time.sleep(5)

    assert status == "failed", f"Expected 'failed', got '{status}'"
    print(f"    sacct reports: FAILED")

    # Verify .done marker is NOT present
    print(f"\n[3] Checking .done marker (should be absent)...")
    done = check_done_marker(HOST, remote_log)
    print(f"    .done exists: {done}")
    assert not done, ".done marker should NOT exist for failed job!"

    print(f"\n    PASS: Failure detected correctly via sacct")
    return True


def test_check_job_status_integration():
    """Test the full check_job_status() function with real SLURM jobs."""
    from chester.auto_pull import check_job_status

    print("\n" + "=" * 60)
    print(" TEST 3: check_job_status() integration")
    print("=" * 60)

    # Use the jobs from tests 1 and 2
    # Check a recent completed job
    result = subprocess.run(
        ["ssh", HOST, f"cat {REMOTE_BASE}/success_test/.chester_slurm_job_id"],
        capture_output=True, text=True, timeout=15,
    )
    if result.stdout.strip():
        success_job_id = int(result.stdout.strip())
        job = {
            'host': HOST,
            'remote_log_dir': f"{REMOTE_BASE}/success_test",
            'slurm_job_id': success_job_id,
        }
        status = check_job_status(job)
        print(f"  Success job {success_job_id}: check_job_status = '{status}'")
        assert status == "done", f"Expected 'done', got '{status}'"
    else:
        print("  Skipping success job check (no job ID file)")

    # Check a recent failed job
    result = subprocess.run(
        ["ssh", HOST, f"cat {REMOTE_BASE}/failure_test/.chester_slurm_job_id"],
        capture_output=True, text=True, timeout=15,
    )
    if result.stdout.strip():
        failure_job_id = int(result.stdout.strip())
        job = {
            'host': HOST,
            'remote_log_dir': f"{REMOTE_BASE}/failure_test",
            'slurm_job_id': failure_job_id,
        }
        status = check_job_status(job)
        print(f"  Failure job {failure_job_id}: check_job_status = '{status}'")
        assert status == "failed", f"Expected 'failed', got '{status}'"
    else:
        print("  Skipping failure job check (no job ID file)")

    print(f"\n    PASS: check_job_status() works correctly with real SLURM jobs")
    return True


def cleanup():
    """Clean up remote test files."""
    print(f"\n[cleanup] Removing {REMOTE_BASE} on {HOST}...")
    subprocess.run(
        ["ssh", HOST, f"rm -rf {REMOTE_BASE}"],
        timeout=15,
    )


def main():
    print(f"SLURM Job Tracking — Live Validation")
    print(f"Host: {HOST}")
    print(f"Remote dir: {REMOTE_BASE}")
    print()

    results = {}

    try:
        results["success"] = test_success_job()
        results["failure"] = test_failure_job()
        results["integration"] = test_check_job_status_integration()
    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        results["exception"] = False

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_pass = all(results.values())
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    cleanup()
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
