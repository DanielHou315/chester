# tests/test_serial_local.py
"""Tests for local backend serial execution path and run_experiment_lite serial integration."""
import base64
import os
import pickle  # noqa: S403 - required by chester's variant serialization protocol
import sys
import pytest

import chester.run_exp as run_exp
from chester.run_exp import run_experiment_lite, VariantGenerator
from chester.backends.base import BackendConfig, SingularityConfig
from chester.backends.local import LocalBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_backend(package_manager="python", singularity=None, prepare=None,
                  project_path="/tmp/myproject"):
    config = BackendConfig(
        name="local",
        type="local",
        singularity=singularity,
        prepare=prepare,
    )
    project_config = {
        "project_path": project_path,
        "package_manager": package_manager,
    }
    return LocalBackend(config, project_config)


def _make_hydra_params(variant_dict, log_dir="/logs/exp1"):
    """Create a params dict with serialized variant_data for hydra mode."""
    encoded = base64.b64encode(pickle.dumps(variant_dict)).decode("utf-8")  # noqa: S301
    return {"variant_data": encoded, "log_dir": log_dir}


def _make_config(tmp_path):
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(f"""
log_dir: {tmp_path}/data
project_path: {tmp_path}
package_manager: python
backends:
  local:
    type: local
""")
    return chester_dir


# ---------------------------------------------------------------------------
# a. generate_command with extra_overrides (hydra mode)
# ---------------------------------------------------------------------------

class TestGenerateCommandExtraOverridesHydra:
    def test_extra_override_appears_as_hydra_arg(self):
        """extra_overrides={"task": "training"} with hydra → task=training in cmd."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({"seed": 42})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
            extra_overrides={"task": "training"},
        )
        assert "task=training" in cmd

    def test_extra_override_merges_with_existing_params(self):
        """extra_overrides should appear alongside existing params."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({"seed": 1})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
            extra_overrides={"task": "evaluate"},
        )
        assert "task=evaluate" in cmd
        assert "seed=1" in cmd

    def test_extra_override_string_value(self):
        """String values in extra_overrides should pass through correctly."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({})}
        cmd = backend.generate_command(
            task,
            script="run.py",
            hydra_enabled=True,
            extra_overrides={"task": "evaluate"},
        )
        assert "task=evaluate" in cmd

    def test_extra_override_numeric_value(self):
        """Numeric values in extra_overrides should appear in command."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({})}
        cmd = backend.generate_command(
            task,
            script="run.py",
            hydra_enabled=True,
            extra_overrides={"num_steps": 1000},
        )
        assert "num_steps=1000" in cmd


# ---------------------------------------------------------------------------
# b. generate_command without extra_overrides
# ---------------------------------------------------------------------------

class TestGenerateCommandNoExtraOverrides:
    def test_no_extra_override_key_absent(self):
        """Without extra_overrides, override key should not appear in command."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({"seed": 42})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
        )
        assert "task=" not in cmd

    def test_no_extra_overrides_basic_command(self):
        """Without extra_overrides, command should still contain the script."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({"lr": 0.01})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
        )
        assert "train.py" in cmd
        assert "python" in cmd
        assert "lr=0.01" in cmd

    def test_extra_overrides_none_is_same_as_omitted(self):
        """Passing extra_overrides=None should produce the same result as not passing it."""
        backend = _make_backend()
        task = {"params": _make_hydra_params({"seed": 7})}
        cmd_none = backend.generate_command(
            task, script="run.py", hydra_enabled=True, extra_overrides=None
        )
        cmd_omitted = backend.generate_command(
            task, script="run.py", hydra_enabled=True
        )
        assert cmd_none == cmd_omitted


# ---------------------------------------------------------------------------
# c. generate_command with singularity + extra_overrides
# ---------------------------------------------------------------------------

class TestGenerateCommandSingularityExtraOverrides:
    def test_singularity_wraps_command_with_override(self):
        """Singularity exec should wrap command that contains the hydra override."""
        sing = SingularityConfig(image="/path/to/container.sif")
        backend = _make_backend(singularity=sing)
        task = {"params": _make_hydra_params({"seed": 1})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
            extra_overrides={"task": "training"},
        )
        assert "singularity exec" in cmd
        assert "task=training" in cmd

    def test_singularity_image_present(self):
        """Generated command should reference the container image."""
        sing = SingularityConfig(image="/path/to/container.sif")
        backend = _make_backend(singularity=sing)
        task = {"params": _make_hydra_params({})}
        cmd = backend.generate_command(
            task,
            script="train.py",
            hydra_enabled=True,
            extra_overrides={"task": "evaluate"},
        )
        assert "/path/to/container.sif" in cmd
        assert "task=evaluate" in cmd

    def test_singularity_gpu_flag_with_override(self):
        """GPU-enabled singularity should include --nv with override present."""
        sing = SingularityConfig(image="/imgs/container.sif", gpu=True)
        backend = _make_backend(singularity=sing)
        task = {"params": _make_hydra_params({})}
        cmd = backend.generate_command(
            task,
            script="main.py",
            hydra_enabled=True,
            extra_overrides={"task": "training"},
        )
        assert "--nv" in cmd
        assert "task=training" in cmd


# ---------------------------------------------------------------------------
# d. run_experiment_lite local serial dry run — command contains both steps
# ---------------------------------------------------------------------------

class TestRunExpLiteSerialDryRun:
    def test_serial_dry_run_command_contains_both_steps(self, tmp_path, capsys):
        """Dry run with serial steps should print a command with both steps joined by &&."""
        chester_dir = _make_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            run_exp.exp_count = -2

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="serial")
            vg.add("seed", [1])
            variants = vg.variants()

            for v in variants:
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=v,
                    mode="local",
                    exp_prefix="serial_test",
                    dry=True,
                    git_snapshot=False,
                    hydra_enabled=True,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

        captured = capsys.readouterr()
        output = captured.out
        # The chained command contains " && " separating two python commands
        assert " && " in output
        # Both step overrides should appear in the printed command
        assert "task=training" in output
        assert "task=evaluate" in output

    def test_serial_dry_run_steps_in_order(self, tmp_path, capsys):
        """Serial steps should appear in order (first step before second)."""
        chester_dir = _make_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            run_exp.exp_count = -2

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="serial")
            vg.add("seed", [1])
            variants = vg.variants()

            for v in variants:
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=v,
                    mode="local",
                    exp_prefix="serial_order_test",
                    dry=True,
                    git_snapshot=False,
                    hydra_enabled=True,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

        captured = capsys.readouterr()
        output = captured.out
        idx_training = output.find("task=training")
        idx_evaluate = output.find("task=evaluate")
        assert idx_training != -1
        assert idx_evaluate != -1
        assert idx_training < idx_evaluate, "training step should come before evaluate step"


# ---------------------------------------------------------------------------
# e. run_experiment_lite local serial variant count
# ---------------------------------------------------------------------------

class TestRunExpLiteSerialVariantCount:
    def test_serial_collapses_to_seed_count_only(self, tmp_path):
        """order='serial' on task (2 values) with seed (2 values) → 2 variants, not 4."""
        chester_dir = _make_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        call_count = 0
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            run_exp.exp_count = -2

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="serial")
            vg.add("seed", [1, 2])
            variants = vg.variants()

            for v in variants:
                call_count += 1
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=v,
                    mode="local",
                    exp_prefix="serial_count_test",
                    dry=True,
                    git_snapshot=False,
                    hydra_enabled=True,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

        assert call_count == 2, (
            f"Expected 2 iterations (one per seed), got {call_count}. "
            "Serial tasks should be collapsed, not cross-producted."
        )

    def test_serial_variant_count_via_vg(self):
        """VariantGenerator.variants() with serial key produces seed-count variants."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        assert len(variants) == 2

    def test_serial_each_variant_has_serial_steps_metadata(self):
        """Each collapsed variant should carry _chester_serial_steps metadata."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_serial_steps" in v
            steps = v["_chester_serial_steps"]
            assert len(steps) == 1
            key, vals = steps[0]
            assert key == "task"
            assert vals == ["training", "evaluate"]


# ---------------------------------------------------------------------------
# f. Serial on local doesn't raise dependency error
# ---------------------------------------------------------------------------

class TestSerialLocalNoDependencyError:
    def test_serial_on_local_does_not_raise(self, tmp_path):
        """order='serial' should work on local backends without skip_dependency_check."""
        chester_dir = _make_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            run_exp.exp_count = -2

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="serial")
            vg.add("seed", [1])
            variants = vg.variants()

            # Should not raise
            for v in variants:
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=v,
                    mode="local",
                    exp_prefix="serial_no_error_test",
                    skip_dependency_check=False,
                    dry=True,
                    git_snapshot=False,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

    def test_dependent_on_local_raises_without_skip(self, tmp_path):
        """order='dependent' on local backend should raise ValueError (contrast with serial)."""
        chester_dir = _make_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            run_exp.exp_count = -2

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="dependent")
            variants = vg.variants()

            with pytest.raises(ValueError, match="order='dependent'"):
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=variants[0],
                    mode="local",
                    exp_prefix="dependent_error_test",
                    skip_dependency_check=False,
                    dry=True,
                    git_snapshot=False,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

    def test_serial_on_local_no_seq_identity_metadata(self):
        """Serial variants should not carry _chester_seq_identity (which triggers dep check)."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" not in v
            assert "_chester_pred_identities" not in v
