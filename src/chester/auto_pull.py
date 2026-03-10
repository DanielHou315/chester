"""
Chester Auto Pull - Status checking and result pulling for remote jobs.

Provides helpers to check job status on remote hosts and pull results
back to the local machine when jobs complete.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from . import config


def check_done_marker(host: str, remote_log_dir: str) -> bool:
    """Check if .done marker exists on remote host via SSH."""
    done_file = os.path.join(remote_log_dir, '.done')
    cmd = ["ssh", host, f"test -f {done_file} && echo done"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() == 'done'
    except subprocess.TimeoutExpired:
        print(f"[auto_pull] SSH timeout checking {host}:{done_file}")
        return False
    except Exception as e:
        print(f"[auto_pull] Error checking {host}:{done_file}: {e}")
        return False


def pull_results(host: str, remote_log_dir: str, local_log_dir: str, bare: bool = False) -> bool:
    """Pull results from remote host to local directory."""
    # Create local directory if it doesn't exist
    os.makedirs(os.path.dirname(local_log_dir), exist_ok=True)

    cmd = ["rsync", "-avzh", "--progress",
           f"{host}:{remote_log_dir}/", f"{local_log_dir}/"]

    if bare:
        # Exclude large files
        cmd.extend([
            "--exclude", "*.pkl",
            "--exclude", "*.png",
            "--exclude", "*.gif",
            "--exclude", "*.pth",
            "--exclude", "*.pt",
        ])

    print(f"[auto_pull] Pulling: {host}:{remote_log_dir} -> {local_log_dir}")
    try:
        result = subprocess.run(cmd, timeout=600)  # 10 min timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[auto_pull] Rsync timeout for {host}:{remote_log_dir}")
        return False
    except Exception as e:
        print(f"[auto_pull] Error pulling from {host}:{remote_log_dir}: {e}")
        return False


def pull_extra_dirs(host: str, extra_pull_dirs: list, bare: bool = False) -> bool:
    """
    Pull additional directories from remote host.

    Args:
        host: Remote host address
        extra_pull_dirs: List of dicts with 'remote' and 'local' keys
        bare: If True, exclude large files (*.pkl, *.pth, etc.)

    Returns:
        True if all pulls succeeded, False otherwise
    """
    if not extra_pull_dirs:
        return True

    all_success = True
    for entry in extra_pull_dirs:
        remote_path = entry['remote']
        local_path = entry['local']
        print(f"[auto_pull] Pulling extra dir: {host}:{remote_path} -> {local_path}")
        if not pull_results(host, remote_path, local_path, bare=bare):
            print(f"[auto_pull] Warning: Failed to pull extra dir {remote_path}")
            all_success = False
    return all_success


def execute_pull_for_job(job: dict, bare: bool = False) -> str:
    """
    Check status of one job and pull if complete.

    The caller (CLI) is responsible for deleting or updating the job file
    based on the returned status.

    rsync path safety: pull_results() appends trailing slashes to both
    remote_log_dir and local_log_dir so rsync syncs directory *contents*
    rather than creating a nested subdirectory. Do not remove the trailing
    slashes from pull_results().

    Failed jobs are not pulled: the job file is marked failed and no rsync
    is attempted. To retrieve logs from a failed job, use `chester pull`.

    Returns one of: 'pulled', 'pull_failed', 'failed', 'running'.
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
        else:
            print(f'[chester] Warning: could not read PID for {exp_name} — orphan process may still be running')
        if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
            pull_extra_dirs(host, extra_pull_dirs, bare=bare)
            return 'pulled'
        return 'pull_failed'

    if status == 'failed':
        print(f'[chester] Job FAILED: {exp_name} on {host} — skipping pull')
        return 'failed'

    return 'running'


def get_remote_pid(host: str, remote_log_dir: str) -> Optional[int]:
    """Read the saved PID from .chester_pid file on remote."""
    pid_file = os.path.join(remote_log_dir, '.chester_pid')
    cmd = ["ssh", host, f"cat {pid_file} 2>/dev/null"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if result.stdout.strip():
            return int(result.stdout.strip())
    except subprocess.TimeoutExpired:
        print(f"[auto_pull] SSH timeout reading PID from {host}:{pid_file}")
    except ValueError:
        print(f"[auto_pull] Invalid PID in {host}:{pid_file}")
    except Exception as e:
        print(f"[auto_pull] Error reading PID from {host}:{pid_file}: {e}")
    return None


def check_process_running(host: str, pid: int) -> bool:
    """Check if process with given PID is still running on remote."""
    cmd = ["ssh", host, f"ps -p {pid} -o pid= 2>/dev/null"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() != ''
    except subprocess.TimeoutExpired:
        print(f"[auto_pull] SSH timeout checking process {pid} on {host}")
        return True  # Assume running if can't check (conservative)
    except Exception as e:
        print(f"[auto_pull] Error checking process {pid} on {host}: {e}")
        return True  # Assume running if can't check


def kill_process_tree(host: str, pid: int):
    """Kill process and all its descendants on remote. SIGTERM then SIGKILL."""
    # First, send SIGTERM to allow graceful shutdown
    term_cmd = ["ssh", host,
                f"pkill -TERM -P {pid} 2>/dev/null; kill -TERM {pid} 2>/dev/null || true"]
    try:
        subprocess.run(term_cmd, timeout=30)
        print(f"[auto_pull] Sent SIGTERM to PID {pid} and children on {host}")
    except Exception as e:
        print(f"[auto_pull] Failed to send SIGTERM: {e}")
        return

    # Wait 5 seconds for graceful shutdown
    time.sleep(5)

    # Check if still running, then SIGKILL
    if check_process_running(host, pid):
        kill_cmd = ["ssh", host,
                    f"pkill -KILL -P {pid} 2>/dev/null; kill -KILL {pid} 2>/dev/null || true"]
        try:
            subprocess.run(kill_cmd, timeout=30)
            print(f"[auto_pull] Sent SIGKILL to PID {pid} and children on {host}")
        except Exception as e:
            print(f"[auto_pull] Failed to send SIGKILL: {e}")


_SLURM_FAILED_STATES = frozenset({
    "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
})
_SLURM_RUNNING_STATES = frozenset({
    "RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "CONFIGURING",
})


def check_slurm_job_status(host: str, slurm_job_id: int) -> str:
    """Query SLURM job status via sacct.

    Returns: 'completed', 'failed', 'running', or 'unknown'.
    """
    cmd = [
        "ssh", host,
        f"sacct -j {slurm_job_id} --format=State --noheader -P 2>/dev/null"
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
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


def check_job_status(job: dict) -> str:
    """Check job status based on .done marker and process/SLURM state.

    For SLURM jobs (has 'slurm_job_id'): uses sacct for failure detection.
    For SSH jobs: uses PID-based tracking.

    Returns: 'running', 'done', 'done_orphans', or 'failed'
    """
    host = job['host']
    remote_log_dir = job['remote_log_dir']
    done = check_done_marker(host, remote_log_dir)

    if done:
        slurm_job_id = job.get('slurm_job_id')
        if slurm_job_id is None:
            pid = get_remote_pid(host, remote_log_dir)
            if pid is not None and check_process_running(host, pid):
                return 'done_orphans'
        return 'done'

    slurm_job_id = job.get('slurm_job_id')
    if slurm_job_id is not None:
        slurm_status = check_slurm_job_status(host, slurm_job_id)
        if slurm_status == 'failed':
            return 'failed'
        return 'running'
    else:
        pid = get_remote_pid(host, remote_log_dir)
        if pid is None:
            return 'running'
        running = check_process_running(host, pid)
        if not running:
            return 'failed'
        return 'running'


