"""
Example launcher script using Chester v2 backend system.

This script demonstrates how to launch experiments with the new
.chester/config.yaml configuration format.

Usage:
    uv run python launch_example.py
"""
from chester.run_exp import run_experiment_lite, VariantGenerator


def run_task(variant, log_dir, exp_name):
    """Your training function. Chester calls this with the variant dict."""
    print(f"Running {exp_name} in {log_dir}")
    print(f"  learning_rate={variant['learning_rate']}")
    print(f"  batch_size={variant['batch_size']}")


def main():
    vg = VariantGenerator()
    vg.add('learning_rate', [0.001, 0.01])
    vg.add('batch_size', [32, 64])

    for v in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=v,
            mode='local',  # or 'gl', 'armdual', etc. (must match a backends key)
            exp_prefix='my_experiment',
            # Per-experiment SLURM overrides (only used with slurm backends):
            # slurm_overrides={'time': '6:00:00', 'gpus': 2},
        )


if __name__ == '__main__':
    main()
