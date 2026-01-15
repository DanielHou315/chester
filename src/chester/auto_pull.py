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

from chester import config


def check_done_marker(host: str, remote_log_dir: str) -> bool:
    """Check if .done marker exists on remote host via SSH."""
    done_file = os.path.join(remote_log_dir, '.done')
    cmd = f'ssh {host} "test -f {done_file} && echo done"'
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
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

    cmd = f'rsync -avzh --progress {host}:{remote_log_dir}/ {local_log_dir}/'

    if bare:
        # Exclude large files
        cmd += " --exclude '*.pkl' --exclude '*.png' --exclude '*.gif' --exclude '*.pth' --exclude '*.pt'"

    print(f"[auto_pull] Pulling: {host}:{remote_log_dir} -> {local_log_dir}")
    try:
        result = subprocess.run(cmd, shell=True, timeout=600)  # 10 min timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[auto_pull] Rsync timeout for {host}:{remote_log_dir}")
        return False
    except Exception as e:
        print(f"[auto_pull] Error pulling from {host}:{remote_log_dir}: {e}")
        return False


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
    Main polling loop. Checks for .done markers and pulls completed jobs.

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
            if once:
                print("[auto_pull] No pending jobs, exiting")
                break
            print(f"[auto_pull] No pending jobs, sleeping {poll_interval}s")
            time.sleep(poll_interval)
            continue

        print(f"[auto_pull] Checking {len(pending_jobs)} pending jobs...")

        for job in pending_jobs:
            host = job['host']
            remote_log_dir = job['remote_log_dir']
            local_log_dir = job['local_log_dir']
            exp_name = job.get('exp_name', 'unknown')

            if check_done_marker(host, remote_log_dir):
                print(f"[auto_pull] Job completed: {exp_name} on {host}")

                if pull_results(host, remote_log_dir, local_log_dir, bare=bare):
                    job['status'] = 'pulled'
                    job['pulled_at'] = datetime.now().isoformat()
                    print(f"[auto_pull] Successfully pulled: {local_log_dir}")
                else:
                    job['status'] = 'pull_failed'
                    print(f"[auto_pull] Failed to pull: {exp_name}")

                # Save manifest after each job status change
                save_manifest(manifest_path, jobs)

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
