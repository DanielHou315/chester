"""
Chester CLI — unified command-line interface.

Entry point: chester <subcommand> [options]

Subcommands:
    pull-remote   Check all pending remote jobs and pull completed ones.
"""

import argparse
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

    Result actions per status:
        pulled      -> delete job file
        failed      -> mark status='failed' in file, do NOT pull
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
        else:
            pass  # pull_failed and running: leave file as pending, no action needed

    print(
        f'[chester] Done: {counts["pulled"]} pulled, '
        f'{counts["running"]} still running, '
        f'{counts["failed"]} failed (not pulled), '
        f'{counts["pull_failed"]} pull errors (will retry)'
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='chester',
        description='Chester ML experiment launcher CLI',
    )
    sub = parser.add_subparsers(dest='command')

    pr = sub.add_parser('pull-remote', help='Check pending remote jobs and pull completed ones')
    pr.add_argument('--prefix', type=str, default=None,
                    help='Only process jobs matching this exp_prefix')
    pr.add_argument('--bare', action='store_true',
                    help='Exclude large files (*.pkl, *.pth, etc.) when pulling')
    pr.add_argument('--dry-run', action='store_true',
                    help='Print what would be checked without doing anything')

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
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
