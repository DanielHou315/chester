"""Tests for deprecation warnings on legacy modules."""
import importlib
import warnings

import pytest


def test_config_ec2_import_warns():
    import chester.config_ec2
    # Force reload so the module-level warning fires again
    with pytest.warns(DeprecationWarning, match="EC2"):
        importlib.reload(chester.config_ec2)


def test_utils_s3_import_warns():
    import chester.utils_s3
    with pytest.warns(DeprecationWarning, match="EC2"):
        importlib.reload(chester.utils_s3)


def test_version_from_metadata():
    import chester
    # Should be a real version string, not the old hardcoded "0.2.0"
    assert chester.__version__ != "0.2.0"
    # Should look like a version (digits and dots, possibly with pre-release suffix)
    assert chester.__version__[0].isdigit() or chester.__version__ == "0.0.0"


def test_run_exp_worker_no_cloudpickle_raises():
    """Verify the old non-cloudpickle path now raises RuntimeError."""
    from chester.run_exp_worker import run_experiment
    with pytest.raises(RuntimeError, match="no longer supported"):
        run_experiment(["prog", "--use_cloudpickle", "False", "--args_data", "dGVzdA=="])


def test_slurm_shellquote_uses_utils():
    """Verify slurm._shellquote is now imported from utils."""
    from chester.slurm import _shellquote, _to_param_val
    from chester.utils import shellquote, to_param_val
    assert _shellquote is shellquote
    assert _to_param_val is to_param_val


def test_slurm_to_local_command_mutable_default():
    """Verify to_local_command no longer has mutable default arg."""
    import inspect
    from chester.slurm import to_local_command
    sig = inspect.signature(to_local_command)
    default = sig.parameters['env'].default
    assert default is None, f"Expected None default for env, got {default!r}"
