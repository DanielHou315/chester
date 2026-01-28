#!/usr/bin/env python
"""
Chester launcher to test extra_pull_dirs feature.

This launcher:
1. Syncs chester repo to remote
2. Launches a test task that waits ~1 minute and creates files
3. Uses extra_pull_dirs to pull additional directories back
4. Verifies auto-pull waits for job completion

Usage:
    # Run on armfranka (SSH mode)
    uv run python launch_test_extra_pull.py --mode armfranka

    # Dry run
    uv run python launch_test_extra_pull.py --mode armfranka --dry

    # Local test (quick)
    uv run python launch_test_extra_pull.py --mode local --wait 10
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime

from chester.run_exp import run_experiment_lite, VariantGenerator
from chester import config
from test_extra_pull import run_test_task

# Remote paths for syncing chester repo
# Set via environment variables: CHESTER_REMOTE_DIR_GL, CHESTER_REMOTE_DIR_ARMFRANKA, etc.
def get_remote_chester_dir(mode: str) -> str:
    """Get remote chester directory from environment variable."""
    env_var = f"CHESTER_REMOTE_DIR_{mode.upper()}"
    return os.environ.get(env_var, '')


def sync_chester_to_remote(mode: str, dry: bool = False):
    """Sync the entire chester repo to remote."""
    if mode == 'local':
        return

    remote_dir = get_remote_chester_dir(mode)
    if not remote_dir:
        print(f"Warning: No remote chester directory configured for mode '{mode}'")
        print(f"Set CHESTER_REMOTE_DIR_{mode.upper()} environment variable")
        return

    chester_repo = os.path.dirname(config.PROJECT_PATH)
    remote_host = config.HOST_ADDRESS.get(mode, mode)

    chester_src_dir = os.path.join(chester_repo, 'src', 'chester')
    rsync_exclude = os.path.join(chester_src_dir, 'rsync_exclude')
    if not os.path.exists(rsync_exclude):
        rsync_exclude = os.path.join(config.PROJECT_PATH, 'rsync_exclude')

    cmd = f"rsync -avzh --delete --exclude-from='{rsync_exclude}' {chester_repo}/ {remote_host}:{remote_dir}"

    print(f"\n[test] Syncing chester repo to remote...")
    print(f"[test] Command: {cmd}")

    if dry:
        print("[test] DRY RUN - skipping sync")
    else:
        subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Test extra_pull_dirs feature')
    parser.add_argument('--mode', type=str, default='armfranka',
                        choices=['local', 'armfranka', 'gl'],
                        help='Execution mode')
    parser.add_argument('--dry', action='store_true',
                        help='Dry run - print commands without executing')
    parser.add_argument('--wait', type=int, default=60,
                        help='Seconds to wait (tests auto-pull polling)')
    args = parser.parse_args()

    # Verify config
    config_path = config.get_config_path()
    if config_path is None:
        print("ERROR: chester.yaml not found!")
        sys.exit(1)

    print(f"Using config: {config_path}")
    print(f"Project root: {config.PROJECT_PATH}")

    # Sync chester repo for remote modes
    sync_chester_to_remote(args.mode, dry=args.dry)

    # Generate unique experiment name
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    exp_name = f"test_extra_pull_{timestamp}"

    # Define the extra directory to pull (relative to PROJECT_PATH)
    # This will be: tests/data/generated_data on both local and remote
    extra_dir_relative = 'data/generated_data'

    # Create variant using VariantGenerator (required for auto-pull to work)
    vg = VariantGenerator()
    vg.add('wait_seconds', [args.wait])
    # Pass the absolute extra_dir path to the task
    # On local: PROJECT_PATH/data/generated_data
    # On remote: REMOTE_DIR/data/generated_data
    extra_dir_path = os.path.join(
        config.REMOTE_DIR.get(args.mode, config.PROJECT_PATH),
        extra_dir_relative
    ) if args.mode != 'local' else os.path.join(config.PROJECT_PATH, extra_dir_relative)
    vg.add('extra_dir', [extra_dir_path])

    print(f"\n[test] Launching test experiment: {exp_name}")
    print(f"[test] Mode: {args.mode}")
    print(f"[test] Wait time: {args.wait} seconds")
    print(f"[test] Extra pull dir: {extra_dir_relative}")

    # Launch experiment (using VariantGenerator to set first/last variant flags)
    for variant in vg.variants():
        run_experiment_lite(
            stub_method_call=run_test_task,
            variant=variant,
            mode=args.mode,
            exp_prefix='test_extra_pull',
            exp_name=exp_name,
            dry=args.dry,
            auto_pull=(args.mode != 'local'),
            auto_pull_interval=15,  # Poll every 15 seconds for faster testing
            extra_pull_dirs=[extra_dir_relative],  # Key feature being tested!
        )

    if args.dry:
        print("\n[DRY RUN] No experiment was actually launched.")
    else:
        print(f"\n[test] Experiment launched: {exp_name}")
        if args.mode != 'local':
            print(f"[test] Auto-pull will poll every 15 seconds")
            print(f"[test] Expected completion in ~{args.wait + 30} seconds")
            print(f"\n[test] Files to be pulled:")
            print(f"  1. Log dir: data/test_extra_pull/{exp_name}/")
            print(f"  2. Extra dir: {extra_dir_relative}/")
            print(f"\n[test] Check manifest in data/.chester_manifests/ for status")


if __name__ == '__main__':
    main()
