# tests/test_serial_script_gen.py
"""Tests for serial step script generation in SLURM and SSH backends.

Covers:
  a. SLURM serial_steps generates multiple python commands
  b. SSH serial_steps generates multiple python commands
  c. SLURM serial_steps with singularity wraps each step independently
  d. SSH serial_steps with singularity wraps each step independently
  e. No serial_steps generates a single python command
  f. .done marker appears after all commands (last substantive line)
"""
import base64
import importlib

import pytest

from chester.backends.base import BackendConfig, SlurmConfig, SingularityConfig
from chester.backends.slurm import SlurmBackend
from chester.backends.ssh import SSHBackend

# Mirror the serialization used by run_experiment_lite when encoding variant_data.
# This module is the standard library serialization module used throughout chester.
_serial = importlib.import_module("pickle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_variant(variant: dict) -> str:
    """Encode a variant dict exactly as run_experiment_lite does."""
    return base64.b64encode(_serial.dumps(variant)).decode("utf-8")


def _make_task(variant: dict = None, log_dir: str = "/logs/exp1",
               exp_name: str = "test_job") -> dict:
    """Build a minimal task dict with a serialized variant_data payload."""
    if variant is None:
        variant = {"lr": 0.01}
    return {
        "params": {
            "log_dir": log_dir,
            "exp_name": exp_name,
            "variant_data": _encode_variant(variant),
        }
    }


def _make_slurm_backend(singularity=None, package_manager="python") -> SlurmBackend:
    config = BackendConfig(
        name="test",
        type="slurm",
        host="server",
        remote_dir="/home/user/project",
        singularity=singularity,
        slurm=SlurmConfig(partition="gpu", time="1:00:00", gpus=1),
    )
    project_config = {
        "project_path": "/local/project",
        "package_manager": package_manager,
    }
    return SlurmBackend(config, project_config)


def _make_ssh_backend(singularity=None, package_manager="python") -> SSHBackend:
    config = BackendConfig(
        name="test",
        type="ssh",
        host="server",
        remote_dir="/home/user/project",
        singularity=singularity,
    )
    project_config = {
        "project_path": "/local/project",
        "package_manager": package_manager,
    }
    return SSHBackend(config, project_config)


def _make_singularity() -> SingularityConfig:
    return SingularityConfig(
        image="/path/to/image.sif",
        gpu=True,
        workdir="/workspace",
    )


SERIAL_STEPS = [("task", ["training", "evaluate"])]
SCRIPT = "train.py"


# ---------------------------------------------------------------------------
# a. SLURM serial_steps generates multiple commands
# ---------------------------------------------------------------------------

class TestSlurmSerialStepsMultipleCommands:
    def test_contains_training_override(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "task=training" in script

    def test_contains_evaluate_override(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "task=evaluate" in script

    def test_exactly_two_python_commands(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        # Each python command references the script name; count those lines
        python_lines = [
            line for line in script.splitlines()
            if SCRIPT in line and not line.strip().startswith("#")
        ]
        assert len(python_lines) == 2

    def test_exactly_one_done_marker(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert script.count(".done") == 1

    def test_both_overrides_on_distinct_lines(self):
        """training and evaluate must appear in separate lines (not on same line)."""
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        lines_with_training = [l for l in script.splitlines() if "task=training" in l]
        lines_with_evaluate = [l for l in script.splitlines() if "task=evaluate" in l]
        assert len(lines_with_training) >= 1
        assert len(lines_with_evaluate) >= 1
        # They must not be the same line
        assert not set(lines_with_training) & set(lines_with_evaluate)


# ---------------------------------------------------------------------------
# b. SSH serial_steps generates multiple commands
# ---------------------------------------------------------------------------

class TestSSHSerialStepsMultipleCommands:
    def test_contains_training_override(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "task=training" in script

    def test_contains_evaluate_override(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "task=evaluate" in script

    def test_exactly_two_python_commands(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        python_lines = [
            line for line in script.splitlines()
            if SCRIPT in line and not line.strip().startswith("#")
        ]
        assert len(python_lines) == 2

    def test_exactly_one_done_marker(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert script.count(".done") == 1

    def test_both_overrides_on_distinct_lines(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        lines_with_training = [l for l in script.splitlines() if "task=training" in l]
        lines_with_evaluate = [l for l in script.splitlines() if "task=evaluate" in l]
        assert len(lines_with_training) >= 1
        assert len(lines_with_evaluate) >= 1
        assert not set(lines_with_training) & set(lines_with_evaluate)


# ---------------------------------------------------------------------------
# c. SLURM serial_steps with singularity — each step wrapped independently
# ---------------------------------------------------------------------------

class TestSlurmSerialStepsSingularity:
    def test_two_singularity_exec_invocations(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert script.count("singularity exec") == 2

    def test_training_step_inside_singularity(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        # The training override must appear on the same line as singularity exec
        lines_with_singularity_and_training = [
            l for l in script.splitlines()
            if "singularity exec" in l and "task=training" in l
        ]
        assert len(lines_with_singularity_and_training) == 1

    def test_evaluate_step_inside_singularity(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        lines_with_singularity_and_evaluate = [
            l for l in script.splitlines()
            if "singularity exec" in l and "task=evaluate" in l
        ]
        assert len(lines_with_singularity_and_evaluate) == 1

    def test_singularity_uses_configured_image(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "/path/to/image.sif" in script

    def test_singularity_gpu_flag_present(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "--nv" in script

    def test_done_marker_not_inside_singularity_call(self):
        """The .done marker should be on its own host line, not inside a container call."""
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_line = next(
            (l for l in script.splitlines() if ".done" in l), None
        )
        assert done_line is not None
        assert "singularity exec" not in done_line


# ---------------------------------------------------------------------------
# d. SSH serial_steps with singularity — each step wrapped independently
# ---------------------------------------------------------------------------

class TestSSHSerialStepsSingularity:
    def test_two_singularity_exec_invocations(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert script.count("singularity exec") == 2

    def test_training_step_inside_singularity(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        lines_with_singularity_and_training = [
            l for l in script.splitlines()
            if "singularity exec" in l and "task=training" in l
        ]
        assert len(lines_with_singularity_and_training) == 1

    def test_evaluate_step_inside_singularity(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        lines_with_singularity_and_evaluate = [
            l for l in script.splitlines()
            if "singularity exec" in l and "task=evaluate" in l
        ]
        assert len(lines_with_singularity_and_evaluate) == 1

    def test_singularity_uses_configured_image(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "/path/to/image.sif" in script

    def test_singularity_gpu_flag_present(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        assert "--nv" in script

    def test_done_marker_not_inside_singularity_call(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_line = next(
            (l for l in script.splitlines() if ".done" in l), None
        )
        assert done_line is not None
        assert "singularity exec" not in done_line


# ---------------------------------------------------------------------------
# e. No serial_steps generates a single python command
# ---------------------------------------------------------------------------

class TestNoSerialStepsSingleCommand:
    def test_slurm_single_python_command(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        python_lines = [
            line for line in script.splitlines()
            if SCRIPT in line and not line.strip().startswith("#")
        ]
        assert len(python_lines) == 1

    def test_ssh_single_python_command(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        python_lines = [
            line for line in script.splitlines()
            if SCRIPT in line and not line.strip().startswith("#")
        ]
        assert len(python_lines) == 1

    def test_slurm_no_task_override_in_single_mode(self):
        backend = _make_slurm_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        assert "task=training" not in script
        assert "task=evaluate" not in script

    def test_ssh_no_task_override_in_single_mode(self):
        backend = _make_ssh_backend()
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        assert "task=training" not in script
        assert "task=evaluate" not in script

    def test_slurm_singularity_single_invocation(self):
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        assert script.count("singularity exec") == 1

    def test_ssh_singularity_single_invocation(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task()
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
        )
        assert script.count("singularity exec") == 1


# ---------------------------------------------------------------------------
# f. Script structure: .done marker is the last substantive line
# ---------------------------------------------------------------------------

class TestDoneMarkerIsLast:
    def test_slurm_done_is_last_non_empty_line(self):
        backend = _make_slurm_backend()
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        non_empty_lines = [l for l in script.rstrip("\n").splitlines() if l.strip()]
        assert non_empty_lines[-1] == "touch /logs/exp1/.done"

    def test_ssh_done_is_last_non_empty_line(self):
        backend = _make_ssh_backend()
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        non_empty_lines = [l for l in script.rstrip("\n").splitlines() if l.strip()]
        assert non_empty_lines[-1] == "touch /logs/exp1/.done"

    def test_slurm_python_commands_before_done(self):
        """Both python commands must appear before the .done marker."""
        backend = _make_slurm_backend()
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_pos = script.index("touch /logs/exp1/.done")
        training_pos = script.index("task=training")
        evaluate_pos = script.index("task=evaluate")
        assert training_pos < done_pos
        assert evaluate_pos < done_pos

    def test_ssh_python_commands_before_done(self):
        backend = _make_ssh_backend()
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_pos = script.index("touch /logs/exp1/.done")
        training_pos = script.index("task=training")
        evaluate_pos = script.index("task=evaluate")
        assert training_pos < done_pos
        assert evaluate_pos < done_pos

    def test_slurm_singularity_done_after_both_container_calls(self):
        """With singularity, both container invocations appear before .done."""
        backend = _make_slurm_backend(singularity=_make_singularity())
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_pos = script.index("touch /logs/exp1/.done")
        first_sing_pos = script.index("singularity exec")
        second_sing_pos = script.index("singularity exec", first_sing_pos + 1)
        assert first_sing_pos < done_pos
        assert second_sing_pos < done_pos

    def test_ssh_singularity_done_after_both_container_calls(self):
        backend = _make_ssh_backend(singularity=_make_singularity())
        task = _make_task(log_dir="/logs/exp1")
        script = backend.generate_script(
            task, script=SCRIPT,
            hydra_enabled=True,
            serial_steps=SERIAL_STEPS,
        )
        done_pos = script.index("touch /logs/exp1/.done")
        first_sing_pos = script.index("singularity exec")
        second_sing_pos = script.index("singularity exec", first_sing_pos + 1)
        assert first_sing_pos < done_pos
        assert second_sing_pos < done_pos
