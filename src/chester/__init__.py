"""
Chester - ML experiment launcher for local, SLURM, SSH, and EC2 environments.

Usage:
    from chester.run_exp import run_experiment_lite, VariantGenerator

    vg = VariantGenerator()
    vg.add('learning_rate', [0.001, 0.01])

    for variant in vg.variants():
        run_experiment_lite(
            stub_method_call=my_train_function,
            variant=variant,
            mode='local',
            exp_prefix='my_experiment',
        )
"""

__version__ = "0.2.0"
__author__ = "Chester Authors"

# Lazy imports to avoid requiring optional dependencies (hydra) at import time
# Users should import directly: from chester.run_exp import run_experiment_lite
__all__ = ["__version__"]


def __getattr__(name):
    """Lazy import for convenience access."""
    if name == "run_experiment_lite":
        from chester.run_exp import run_experiment_lite
        return run_experiment_lite
    elif name == "VariantGenerator":
        from chester.run_exp import VariantGenerator
        return VariantGenerator
    raise AttributeError(f"module 'chester' has no attribute '{name}'")
