#!/usr/bin/env python
"""
Test script for extra_pull_dirs feature.

This script:
1. Waits for ~1 minute to test auto-pull polling
2. Creates files in the log directory (standard behavior)
3. Creates files in an extra directory (tests extra_pull_dirs)

Usage (via chester):
    uv run python launch_test_extra_pull.py --mode armfranka
"""
import os
import time
import json
from datetime import datetime


def run_test_task(variant, log_dir, exp_name):
    """
    Test task that creates files and waits to test auto-pull.

    Args:
        variant: Dictionary of parameters
        log_dir: Directory for logs (standard auto-pull)
        exp_name: Experiment name
    """
    # Extract parameters
    wait_seconds = variant.get('wait_seconds', 60)
    extra_dir = variant.get('extra_dir', None)

    print(f"[test_extra_pull] Starting test task: {exp_name}")
    print(f"[test_extra_pull] Log dir: {log_dir}")
    print(f"[test_extra_pull] Extra dir: {extra_dir}")
    print(f"[test_extra_pull] Will wait {wait_seconds} seconds")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create a file in the standard log directory
    log_file = os.path.join(log_dir, 'test_output.json')
    log_data = {
        'exp_name': exp_name,
        'start_time': datetime.now().isoformat(),
        'variant': variant,
        'status': 'started'
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"[test_extra_pull] Created log file: {log_file}")

    # If extra_dir is specified, create files there too
    if extra_dir:
        os.makedirs(extra_dir, exist_ok=True)

        # Create a "generated dataset" file
        dataset_file = os.path.join(extra_dir, f'{exp_name}_dataset.json')
        dataset_data = {
            'exp_name': exp_name,
            'generated_at': datetime.now().isoformat(),
            'description': 'Test generated dataset file',
            'data': [i * 2 for i in range(10)],  # Some dummy data
        }
        with open(dataset_file, 'w') as f:
            json.dump(dataset_data, f, indent=2)
        print(f"[test_extra_pull] Created dataset file: {dataset_file}")

        # Create another file to verify multiple files are pulled
        metadata_file = os.path.join(extra_dir, f'{exp_name}_metadata.txt')
        with open(metadata_file, 'w') as f:
            f.write(f"Experiment: {exp_name}\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Log dir: {log_dir}\n")
            f.write(f"Extra dir: {extra_dir}\n")
        print(f"[test_extra_pull] Created metadata file: {metadata_file}")

    # Wait to test auto-pull polling
    print(f"[test_extra_pull] Waiting {wait_seconds} seconds...")
    for i in range(wait_seconds):
        if i % 10 == 0:
            print(f"[test_extra_pull] ... {i}/{wait_seconds} seconds elapsed")
        time.sleep(1)

    # Update log file with completion status
    log_data['status'] = 'completed'
    log_data['end_time'] = datetime.now().isoformat()
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    print(f"[test_extra_pull] Task completed!")
    return {'status': 'success'}


if __name__ == '__main__':
    # Standalone test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait_seconds', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='./data/test_extra_pull/test')
    parser.add_argument('--extra_dir', type=str, default='./data/generated_data')
    parser.add_argument('--exp_name', type=str, default='test_standalone')
    args = parser.parse_args()

    variant = {
        'wait_seconds': args.wait_seconds,
        'extra_dir': args.extra_dir,
    }
    run_test_task(variant, args.log_dir, args.exp_name)
