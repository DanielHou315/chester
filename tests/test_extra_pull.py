#!/usr/bin/env python
"""
Test script for extra_pull_dirs feature.

This script:
1. Waits for a specified time to test auto-pull polling
2. Creates files in the log directory (standard behavior)
3. Creates checkpoint files in data/checkpoints/ (tests extra_pull_dirs)
4. Creates artifact files in data/artifacts/ (tests multiple extra_pull_dirs)

Usage (via chester):
    uv run python launch_test_extra_pull.py --mode armfranka
"""
import os
import time
import json
from datetime import datetime


def run_test_task(variant, log_dir, exp_name):
    """
    Test task that creates files in multiple directories and waits to test auto-pull.

    Args:
        variant: Dictionary of parameters
        log_dir: Directory for logs (standard auto-pull)
        exp_name: Experiment name
    """
    # Extract parameters
    wait_seconds = variant.get('wait_seconds', 60)
    checkpoint_dir = variant.get('checkpoint_dir', None)
    artifact_dir = variant.get('artifact_dir', None)

    print(f"[test_extra_pull] Starting test task: {exp_name}")
    print(f"[test_extra_pull] Log dir: {log_dir}")
    print(f"[test_extra_pull] Checkpoint dir: {checkpoint_dir}")
    print(f"[test_extra_pull] Artifact dir: {artifact_dir}")
    print(f"[test_extra_pull] Will wait {wait_seconds} seconds")

    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Create a file in the standard log directory
    log_file = os.path.join(log_dir, 'test_output.json')
    log_data = {
        'exp_name': exp_name,
        'start_time': datetime.now().isoformat(),
        'variant': {k: v for k, v in variant.items() if not k.startswith('chester_')},
        'status': 'started'
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"[test_extra_pull] Created log file: {log_file}")

    # Create checkpoint files (simulating model checkpoints)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Simulate saving checkpoints at different "epochs"
        for epoch in [1, 5, 10]:
            ckpt_file = os.path.join(checkpoint_dir, f'{exp_name}_epoch{epoch}.json')
            ckpt_data = {
                'exp_name': exp_name,
                'epoch': epoch,
                'saved_at': datetime.now().isoformat(),
                'model_weights': [0.1 * epoch * i for i in range(5)],
                'optimizer_state': {'lr': 0.001 / epoch},
                'metrics': {'loss': 1.0 / epoch, 'accuracy': 0.5 + 0.05 * epoch}
            }
            with open(ckpt_file, 'w') as f:
                json.dump(ckpt_data, f, indent=2)
            print(f"[test_extra_pull] Created checkpoint: {ckpt_file}")

    # Create artifact files (simulating generated outputs)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

        # Create a predictions file
        predictions_file = os.path.join(artifact_dir, f'{exp_name}_predictions.json')
        predictions_data = {
            'exp_name': exp_name,
            'generated_at': datetime.now().isoformat(),
            'predictions': [{'input': i, 'output': i * 2} for i in range(10)],
        }
        with open(predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)
        print(f"[test_extra_pull] Created predictions: {predictions_file}")

        # Create a summary report
        report_file = os.path.join(artifact_dir, f'{exp_name}_report.txt')
        with open(report_file, 'w') as f:
            f.write(f"Experiment Report: {exp_name}\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n")
            f.write(f"Log dir: {log_dir}\n")
            f.write(f"Checkpoint dir: {checkpoint_dir}\n")
            f.write(f"Artifact dir: {artifact_dir}\n")
            f.write(f"\nResults:\n")
            f.write(f"  - Final loss: 0.1\n")
            f.write(f"  - Final accuracy: 0.95\n")
        print(f"[test_extra_pull] Created report: {report_file}")

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
    parser.add_argument('--checkpoint_dir', type=str, default='./data/checkpoints')
    parser.add_argument('--artifact_dir', type=str, default='./data/artifacts')
    parser.add_argument('--exp_name', type=str, default='test_standalone')
    args = parser.parse_args()

    variant = {
        'wait_seconds': args.wait_seconds,
        'checkpoint_dir': args.checkpoint_dir,
        'artifact_dir': args.artifact_dir,
    }
    run_test_task(variant, args.log_dir, args.exp_name)
