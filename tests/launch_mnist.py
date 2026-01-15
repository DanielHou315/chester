#!/usr/bin/env python
"""
Chester launcher for MNIST experiments.

This is a standalone test project demonstrating chester usage.
Run from the tests/ directory where chester.yaml is located.

Fully automatic testing:
- First syncs the entire chester repo to remote (so pyproject.toml can reference chester at path="..")
- Then syncs tests/ directory to remote via chester's rsync
- Auto-installs uv if not present on remote
- Runs `uv sync` to install dependencies (chester + PyTorch)
- Trains MNIST model
- Auto-pulls results back when complete

Usage (from tests/ directory):
    # Local test (quick, 1 epoch)
    uv run python launch_mnist.py --mode local --quick

    # Great Lakes SLURM cluster
    uv run python launch_mnist.py --mode gl --quick

    # SSH host armfranka
    uv run python launch_mnist.py --mode armfranka --quick

    # Dry run (print commands without executing)
    uv run python launch_mnist.py --mode armfranka --dry --quick

    # Grid search (8 variants: 2 models x 2 hidden x 2 lr)
    uv run python launch_mnist.py --mode local --grid
"""
import os
import sys
import argparse
import subprocess

from chester.run_exp import run_experiment_lite, VariantGenerator
from chester import config
from train_mnist import run_training

# Remote paths for syncing chester repo (parent of tests/)
# These should match the parent directory of remote_dir in chester.yaml
REMOTE_CHESTER_DIR = {
    'gl': '/home/houhd/code/chester',
    'armfranka': '/home/danielhou/code/chester',
}


def sync_chester_to_remote(mode: str, dry: bool = False):
    """
    Sync the entire chester repo to remote before running tests.

    This ensures that pyproject.toml's `chester = { path = ".." }` works correctly,
    since chester source will be at the parent directory on remote.
    """
    if mode == 'local':
        return  # No need to sync for local mode

    if mode not in REMOTE_CHESTER_DIR:
        print(f"Warning: No remote chester directory configured for mode '{mode}'")
        return

    # Get paths
    chester_repo = os.path.dirname(config.PROJECT_PATH)  # Parent of tests/
    remote_host = config.HOST_ADDRESS.get(mode, mode)
    remote_dir = REMOTE_CHESTER_DIR[mode]

    # Use rsync_exclude from chester package for sensible defaults
    chester_pkg_dir = os.path.dirname(os.path.abspath(__file__))
    chester_src_dir = os.path.join(chester_repo, 'src', 'chester')
    rsync_exclude = os.path.join(chester_src_dir, 'rsync_exclude')

    # Fall back to tests rsync_exclude if chester's doesn't exist
    if not os.path.exists(rsync_exclude):
        rsync_exclude = os.path.join(config.PROJECT_PATH, 'rsync_exclude')

    cmd = f"rsync -avzh --delete --exclude-from='{rsync_exclude}' {chester_repo}/ {remote_host}:{remote_dir}"

    print(f"\n[test] Syncing chester repo to remote...")
    print(f"[test] {chester_repo}/ -> {remote_host}:{remote_dir}")
    print(f"[test] Command: {cmd}")

    if dry:
        print("[test] DRY RUN - skipping sync")
    else:
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"[test] Warning: rsync returned {result.returncode}")
        else:
            print("[test] Chester repo synced successfully")


def create_variants(grid_search=False, quick=False):
    """Create experiment variants."""
    vg = VariantGenerator()

    if quick:
        # Quick test: minimal settings
        vg.add('model', ['mlp'])
        vg.add('hidden_dim', [128])
        vg.add('learning_rate', [0.001])
        vg.add('batch_size', [64])
        vg.add('epochs', [1])
    elif grid_search:
        # Grid search over hyperparameters
        vg.add('model', ['mlp', 'cnn'])
        vg.add('hidden_dim', [128, 256])
        vg.add('learning_rate', [0.001, 0.0001])
        vg.add('batch_size', [64])
        vg.add('epochs', [5])
    else:
        # Default: single run with reasonable settings
        vg.add('model', ['mlp'])
        vg.add('hidden_dim', [256])
        vg.add('learning_rate', [0.001])
        vg.add('batch_size', [64])
        vg.add('epochs', [5])

    return vg.variants()


def main():
    parser = argparse.ArgumentParser(description='Launch MNIST experiments via chester')
    parser.add_argument('--mode', type=str, default='local',
                        choices=['local', 'gl', 'armfranka'],
                        help='Execution mode: local, gl (SLURM), or armfranka (SSH)')
    parser.add_argument('--dry', action='store_true',
                        help='Dry run - print commands without executing')
    parser.add_argument('--grid', action='store_true',
                        help='Run grid search over hyperparameters')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with minimal settings (1 epoch)')
    args = parser.parse_args()

    # Verify config was found
    config_path = config.get_config_path()
    if config_path is None:
        print("ERROR: chester.yaml not found!")
        print("Make sure you're running from a directory with chester.yaml")
        print("or set CHESTER_CONFIG_PATH environment variable.")
        sys.exit(1)

    print(f"Using config: {config_path}")
    print(f"Project root: {config.PROJECT_PATH}")

    # For remote modes, sync chester repo first so pyproject.toml can reference it
    sync_chester_to_remote(args.mode, dry=args.dry)

    # Create variants
    variants = create_variants(grid_search=args.grid, quick=args.quick)
    print(f"\nLaunching {len(variants)} experiment(s) in mode: {args.mode}")

    for i, variant in enumerate(variants):
        print(f"\n--- Variant {i+1}/{len(variants)} ---")
        print(f"  model={variant.get('model')}, lr={variant.get('learning_rate')}, "
              f"hidden={variant.get('hidden_dim')}, epochs={variant.get('epochs')}")

        # Experiment name includes key hyperparameters
        exp_name = f"mnist_{variant.get('model')}_lr{variant.get('learning_rate')}_h{variant.get('hidden_dim')}"

        # Specify local log_dir - chester automatically maps to remote
        # Structure: data/{exp_prefix}/{exp_name}/
        log_dir = os.path.join(config.LOG_DIR, 'mnist_test', exp_name)

        run_experiment_lite(
            stub_method_call=run_training,
            variant=variant,
            mode=args.mode,
            exp_prefix='mnist_test',
            exp_name=exp_name,
            log_dir=log_dir,
            use_gpu=(args.mode != 'local'),
            dry=args.dry,
            auto_pull=(args.mode != 'local'),
            auto_pull_interval=30,
        )

    if args.dry:
        print("\n[DRY RUN] No experiments were actually launched.")
    else:
        print(f"\nLaunched {len(variants)} experiment(s).")


if __name__ == '__main__':
    main()
