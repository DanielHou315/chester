#!/usr/bin/env python
"""
Chester Auto Pull - Background poller for automatic result synchronization.

This script polls remote hosts for .done marker files and automatically
pulls results back to the local machine when jobs complete.

Usage:
    # Run as background process (spawned by chester)
    python chester/auto_pull.py --manifest /path/to/manifest.json

    # Run manually to pull specific jobs
    python chester/auto_pull.py --manifest /path/to/manifest.json --once
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
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
    """
    Check job status based on .done marker and process state.

    Returns:
        'running': Job is still running
        'done': Job completed successfully (.done exists, process exited)
        'done_orphans': Job completed but processes still running
        'failed': Process died without creating .done
    """
    host = job['host']
    remote_log_dir = job['remote_log_dir']

    done = check_done_marker(host, remote_log_dir)
    pid = get_remote_pid(host, remote_log_dir)

    if pid is None:
        # No PID file yet - job may not have started or is a legacy job
        if done:
            return 'done'
        return 'running'

    running = check_process_running(host, pid)

    if done and not running:
        return 'done'
    elif done and running:
        return 'done_orphans'  # Script finished but processes still running
    elif not done and not running:
        return 'failed'  # Process died without creating .done
    else:  # not done and running
        return 'running'


def load_manifest(manifest_path: str) -> list:
    """Load job manifest from JSON file."""
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest_path: str, jobs: list):
    """Save job manifest to JSON file."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(jobs, f, indent=2)


def poll_and_pull(manifest_path: str, poll_interval: int = 60, bare: bool = False, once: bool = False):
    """
    Main polling loop. Checks job status and pulls completed/failed jobs.

    Uses PID-based tracking to detect:
    - Successful completion (.done exists, process exited)
    - Orphan processes (.done exists, process still running -> kill and pull)
    - Failed jobs (process died without .done -> pull logs)

    Args:
        manifest_path: Path to the job manifest JSON file
        poll_interval: Seconds between polls
        bare: If True, exclude large files when pulling
        once: If True, run once and exit instead of looping
    """
    print(f"[auto_pull] Starting poller, manifest: {manifest_path}")
    print(f"[auto_pull] Poll interval: {poll_interval}s, bare mode: {bare}")

    while True:
        jobs = load_manifest(manifest_path)

        if not jobs:
            if once:
                print("[auto_pull] No jobs in manifest, exiting")
                break
            time.sleep(poll_interval)
            continue

        pending_jobs = [j for j in jobs if j.get('status') == 'pending']

        if not pending_jobs:
            # All jobs resolved - print summary and exit
            print("[auto_pull] All jobs resolved:")
            for j in jobs:
                print(f"  {j.get('exp_name', 'unknown')}: {j.get('status')}")
            break

        print(f"[auto_pull] Checking {len(pending_jobs)} pending jobs...")

        for job in pending_jobs:
            host = job['host']
            remote_log_dir = job['remote_log_dir']
            local_log_dir = job['local_log_dir']
            exp_name = job.get('exp_name', 'unknown')

            status = check_job_status(job)

            if status == 'done':
                print(f"[auto_pull] Job completed: {exp_name} on {host}")
                if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
                    # Pull extra directories if configured
                    pull_extra_dirs(host, job.get('extra_pull_dirs', []), bare=bare)
                    job['status'] = 'pulled'
                    job['pulled_at'] = datetime.now().isoformat()
                    print(f"[auto_pull] Successfully pulled: {local_log_dir}")
                else:
                    job['status'] = 'pull_failed'
                    print(f"[auto_pull] Failed to pull: {exp_name}")
                save_manifest(manifest_path, jobs)

            elif status == 'done_orphans':
                print(f"[auto_pull] Job completed with orphan processes: {exp_name} on {host}")
                pid = get_remote_pid(host, remote_log_dir)
                if pid:
                    kill_process_tree(host, pid)
                if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
                    # Pull extra directories if configured
                    pull_extra_dirs(host, job.get('extra_pull_dirs', []), bare=bare)
                    job['status'] = 'pulled'
                    job['pulled_at'] = datetime.now().isoformat()
                    job['had_orphans'] = True
                    print(f"[auto_pull] Successfully pulled (after cleanup): {local_log_dir}")
                else:
                    job['status'] = 'pull_failed'
                    print(f"[auto_pull] Failed to pull: {exp_name}")
                save_manifest(manifest_path, jobs)

            elif status == 'failed':
                print(f"[auto_pull] Job FAILED (process died): {exp_name} on {host}")
                # Pull logs for debugging (always use bare mode for failed jobs)
                pull_results(host, remote_log_dir, local_log_dir, bare=True)
                job['status'] = 'failed'
                job['failed_at'] = datetime.now().isoformat()
                print(f"[auto_pull] Pulled logs from failed job: {local_log_dir}")
                save_manifest(manifest_path, jobs)

            # else: status == 'running' - keep polling

        if once:
            break

        time.sleep(poll_interval)


def main():
    parser = argparse.ArgumentParser(description='Chester Auto Pull - Background result synchronization')
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to job manifest JSON file')
    parser.add_argument('--poll-interval', type=int, default=60,
                        help='Seconds between polls (default: 60)')
    parser.add_argument('--bare', action='store_true',
                        help='Exclude large files (*.pkl, *.pth, etc.)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit instead of continuous polling')

    args = parser.parse_args()

    try:
        poll_and_pull(
            manifest_path=args.manifest,
            poll_interval=args.poll_interval,
            bare=args.bare,
            once=args.once
        )
    except KeyboardInterrupt:
        print("\n[auto_pull] Interrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
