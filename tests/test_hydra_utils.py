# tests/test_hydra_utils.py
"""Tests for Hydra override formatting — especially OmegaConf interpolations."""
import base64
import pickle

import pytest

from chester.hydra_utils import (
    _format_hydra_value,
    build_hydra_args,
    variant_to_hydra_overrides,
)


# ---------------------------------------------------------------------------
# _format_hydra_value
# ---------------------------------------------------------------------------

class TestFormatHydraValue:
    """Unit tests for _format_hydra_value."""

    # --- Interpolations (must pass through unquoted) ---

    def test_simple_interpolation(self):
        assert _format_hydra_value("${db.host}") == "${db.host}"

    def test_eval_resolver(self):
        val = "${eval:'2 ** 10'}"
        assert _format_hydra_value(val) == val

    def test_nested_interpolation(self):
        val = "${oc.select:model.${arch}.layers,4}"
        assert _format_hydra_value(val) == val

    def test_interpolation_with_default(self):
        val = "${oc.env:CUDA_VISIBLE_DEVICES,0}"
        assert _format_hydra_value(val) == val

    def test_interpolation_with_spaces_not_quoted(self):
        """Interpolations containing spaces must NOT be double-quoted."""
        val = "${eval:'len([1, 2, 3])'}"
        assert _format_hydra_value(val) == val
        assert '"' not in _format_hydra_value(val)

    # --- Plain strings ---

    def test_plain_string(self):
        assert _format_hydra_value("adam") == "adam"

    def test_string_with_spaces_quoted(self):
        assert _format_hydra_value("hello world") == '"hello world"'

    def test_string_with_dollar_not_interpolation(self):
        """A leading $ without braces is just a string."""
        assert _format_hydra_value("$HOME/data") == "$HOME/data"

    def test_partial_interpolation_not_special(self):
        """Only full ${...} wrapping counts as interpolation."""
        assert _format_hydra_value("prefix_${x}") == "prefix_${x}"

    # --- Booleans ---

    def test_bool_true(self):
        assert _format_hydra_value(True) == "true"

    def test_bool_false(self):
        assert _format_hydra_value(False) == "false"

    # --- Numerics ---

    def test_int(self):
        assert _format_hydra_value(42) == "42"

    def test_float(self):
        assert _format_hydra_value(0.001) == "0.001"

    # --- Lists ---

    def test_list_of_ints(self):
        assert _format_hydra_value([1, 2, 3]) == "[1,2,3]"

    def test_list_with_interpolation(self):
        result = _format_hydra_value(["${db.host}", 8080])
        assert result == "[${db.host},8080]"

    def test_list_with_comma_string(self):
        result = _format_hydra_value(["a,b", "c"])
        assert result == '["a,b",c]'


# ---------------------------------------------------------------------------
# variant_to_hydra_overrides
# ---------------------------------------------------------------------------

class TestVariantToHydraOverrides:
    """Tests for variant_to_hydra_overrides — full variant → override list."""

    def test_interpolation_in_variant(self):
        variant = {"lr": 0.01, "schedule": "${eval:'[0.1 * i for i in range(10)]'}"}
        overrides = variant_to_hydra_overrides(variant)
        assert "lr=0.01" in overrides
        assert "schedule=${eval:'[0.1 * i for i in range(10)]'}" in overrides

    def test_nested_dict_with_interpolation(self):
        variant = {
            "model": {"_name": "resnet", "layers": "${eval:'2 ** 4'}"},
        }
        overrides = variant_to_hydra_overrides(variant)
        assert "model=resnet" in overrides
        assert "model.layers=${eval:'2 ** 4'}" in overrides

    def test_skips_chester_keys(self):
        variant = {
            "chester_first_variant": True,
            "chester_last_variant": False,
            "is_debug": False,
            "lr": 0.01,
        }
        overrides = variant_to_hydra_overrides(variant)
        assert overrides == ["lr=0.01"]

    def test_env_resolver(self):
        variant = {"data_dir": "${oc.env:DATA_DIR,/tmp/data}"}
        overrides = variant_to_hydra_overrides(variant)
        assert overrides == ["data_dir=${oc.env:DATA_DIR,/tmp/data}"]

    def test_mixed_types(self):
        variant = {
            "seed": 42,
            "use_amp": True,
            "arch": "transformer",
            "hidden": "${eval:'512 * 2'}",
        }
        overrides = variant_to_hydra_overrides(variant)
        assert "seed=42" in overrides
        assert "use_amp=true" in overrides
        assert "arch=transformer" in overrides
        assert "hidden=${eval:'512 * 2'}" in overrides


# ---------------------------------------------------------------------------
# Helper: build a params dict with serialized variant_data
# ---------------------------------------------------------------------------

def _make_params(variant, log_dir="/tmp/logs/exp1"):
    """Build a params dict like run_experiment_lite does."""
    return {
        "variant_data": base64.b64encode(pickle.dumps(variant)).decode("utf-8"),
        "log_dir": log_dir,
        "exp_name": "test_exp",
        "args_data": "dummy",
    }


# ---------------------------------------------------------------------------
# build_hydra_args
# ---------------------------------------------------------------------------

class TestBuildHydraArgs:
    """Tests for build_hydra_args — full params → Hydra args string."""

    def test_basic(self):
        params = _make_params({"lr": 0.01, "seed": 42})
        args = build_hydra_args(params)
        assert "lr=0.01" in args
        assert "seed=42" in args
        assert "hydra.run.dir=/tmp/logs/exp1" in args

    def test_interpolation_passthrough(self):
        params = _make_params({"schedule": "${eval:'2**10'}"})
        args = build_hydra_args(params)
        assert "schedule=${eval:'2**10'}" in args

    def test_hydra_flags_multirun(self):
        params = _make_params({"lr": 0.01})
        args = build_hydra_args(params, hydra_flags={"multirun": True})
        assert "--multirun" in args
        assert "lr=0.01" in args

    def test_hydra_flags_value(self):
        params = _make_params({"lr": 0.01})
        args = build_hydra_args(params, hydra_flags={"config-name": "custom"})
        assert "--config-name=custom" in args

    def test_hydra_flags_false_skipped(self):
        params = _make_params({"lr": 0.01})
        args = build_hydra_args(params, hydra_flags={"multirun": False})
        assert "--multirun" not in args

    def test_no_cli_format(self):
        """Hydra args must NOT contain --key format."""
        params = _make_params({"lr": 0.01, "batch_size": 32})
        args = build_hydra_args(params)
        assert "--lr" not in args
        assert "--batch_size" not in args

    def test_does_not_mutate_params(self):
        params = _make_params({"lr": 0.01})
        original_keys = set(params.keys())
        build_hydra_args(params)
        assert set(params.keys()) == original_keys


# ---------------------------------------------------------------------------
# Backend.build_python_command — hydra vs CLI on all backends
# ---------------------------------------------------------------------------

class TestBuildPythonCommandOnBackends:
    """Verify hydra_enabled works through all backend types."""

    @pytest.fixture()
    def local_backend(self):
        from chester.backends import create_backend, BackendConfig
        cfg = BackendConfig(name="local", type="local")
        return create_backend(cfg, {"package_manager": "uv"})

    @pytest.fixture()
    def ssh_backend(self):
        from chester.backends import create_backend, BackendConfig
        cfg = BackendConfig(
            name="myhost", type="ssh", host="myhost",
            remote_dir="/remote/project",
        )
        return create_backend(cfg, {"package_manager": "uv"})

    @pytest.fixture()
    def slurm_backend(self):
        from chester.backends import create_backend, BackendConfig, SlurmConfig
        cfg = BackendConfig(
            name="gl", type="slurm", host="gl",
            remote_dir="/remote/project",
            slurm=SlurmConfig(partition="gpu", gpus=1),
        )
        return create_backend(cfg, {"package_manager": "uv"})

    def _params(self, variant=None):
        if variant is None:
            variant = {"lr": 0.01, "seed": 42}
        return _make_params(variant)

    # --- CLI mode (hydra_enabled=False) ---

    def test_local_cli_mode(self, local_backend):
        """CLI mode passes raw params (variant_data, log_dir, etc.)."""
        cmd = local_backend.build_python_command(
            self._params(), "main.py", "python",
        )
        assert "--variant_data" in cmd
        assert "--log_dir" in cmd
        assert "lr=0.01" not in cmd  # no hydra format

    def test_ssh_cli_mode(self, ssh_backend):
        cmd = ssh_backend.build_python_command(
            self._params(), "main.py", "python",
        )
        assert "--variant_data" in cmd

    def test_slurm_cli_mode(self, slurm_backend):
        cmd = slurm_backend.build_python_command(
            self._params(), "main.py", "python",
        )
        assert "--variant_data" in cmd

    # --- Hydra mode (hydra_enabled=True) ---

    def test_local_hydra_mode(self, local_backend):
        cmd = local_backend.build_python_command(
            self._params(), "main", "python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in cmd
        assert "seed=42" in cmd
        assert "hydra.run.dir=" in cmd
        assert "--lr" not in cmd
        assert "--seed" not in cmd

    def test_ssh_hydra_mode(self, ssh_backend):
        cmd = ssh_backend.build_python_command(
            self._params(), "main", "python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in cmd
        assert "--lr" not in cmd

    def test_slurm_hydra_mode(self, slurm_backend):
        cmd = slurm_backend.build_python_command(
            self._params(), "main", "python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in cmd
        assert "--lr" not in cmd

    # --- Hydra with flags ---

    def test_hydra_multirun_flag(self, local_backend):
        cmd = local_backend.build_python_command(
            self._params(), "main", "python -m",
            hydra_enabled=True,
            hydra_flags={"multirun": True},
        )
        assert "--multirun" in cmd
        assert "lr=0.01" in cmd

    # --- Hydra with interpolation ---

    def test_hydra_interpolation_in_command(self, local_backend):
        variant = {"layers": "${eval:'2 ** 4'}", "lr": 0.01}
        cmd = local_backend.build_python_command(
            self._params(variant), "main", "python -m",
            hydra_enabled=True,
        )
        assert "layers=${eval:'2 ** 4'}" in cmd
        assert "lr=0.01" in cmd

    # --- Package manager wrapping ---

    def test_uv_wrapping_in_hydra_mode(self, local_backend):
        cmd = local_backend.build_python_command(
            self._params(), "main", "python -m",
            hydra_enabled=True,
        )
        assert cmd.startswith("uv run python -m main")

    # --- Env vars ---

    def test_env_vars_prepended(self, local_backend):
        cmd = local_backend.build_python_command(
            self._params(), "main", "python -m",
            env={"CUDA_VISIBLE_DEVICES": "0"},
            hydra_enabled=True,
        )
        assert "CUDA_VISIBLE_DEVICES=0" in cmd
        assert "lr=0.01" in cmd


# ---------------------------------------------------------------------------
# Full generate_script with hydra — all remote backends
# ---------------------------------------------------------------------------

class TestGenerateScriptHydra:
    """Verify generate_script/generate_command produce Hydra format."""

    def _params_task(self, variant=None, log_dir="/remote/project/data/exp1"):
        if variant is None:
            variant = {"lr": 0.01, "batch_size": 32}
        return {
            "params": {
                **_make_params(variant, log_dir),
                "log_dir": log_dir,
            },
            "exp_name": "test_exp",
        }

    def test_local_generate_command_hydra(self):
        from chester.backends import create_backend, BackendConfig
        cfg = BackendConfig(name="local", type="local")
        backend = create_backend(cfg, {"package_manager": "uv"})
        cmd = backend.generate_command(
            self._params_task(), script="main", python_command="python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in cmd
        assert "batch_size=32" in cmd
        assert "--lr" not in cmd

    def test_ssh_generate_script_hydra(self):
        from chester.backends import create_backend, BackendConfig
        cfg = BackendConfig(
            name="myhost", type="ssh", host="myhost",
            remote_dir="/remote/project",
        )
        backend = create_backend(cfg, {"package_manager": "uv"})
        script = backend.generate_script(
            self._params_task(), script="main", python_command="python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in script
        assert "batch_size=32" in script
        assert "--lr" not in script
        assert "touch" in script  # .done marker still present

    def test_slurm_generate_script_hydra(self):
        from chester.backends import create_backend, BackendConfig, SlurmConfig
        cfg = BackendConfig(
            name="gl", type="slurm", host="gl",
            remote_dir="/remote/project",
            slurm=SlurmConfig(partition="gpu", gpus=1),
        )
        backend = create_backend(cfg, {"package_manager": "uv"})
        script = backend.generate_script(
            self._params_task(), script="main", python_command="python -m",
            hydra_enabled=True,
        )
        assert "lr=0.01" in script
        assert "batch_size=32" in script
        assert "--lr" not in script
        assert "#SBATCH" in script  # SLURM header still present
        assert "touch" in script    # .done marker still present

    def test_slurm_generate_script_cli_mode(self):
        """Verify non-hydra mode still works (--key value for raw params)."""
        from chester.backends import create_backend, BackendConfig, SlurmConfig
        cfg = BackendConfig(
            name="gl", type="slurm", host="gl",
            remote_dir="/remote/project",
            slurm=SlurmConfig(partition="gpu", gpus=1),
        )
        backend = create_backend(cfg, {"package_manager": "uv"})
        script = backend.generate_script(
            self._params_task(), script="main.py", python_command="python",
        )
        assert "--variant_data" in script
        assert "--log_dir" in script
        assert "lr=0.01" not in script  # no hydra format
