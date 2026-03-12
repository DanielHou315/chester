import base64
import pickle
import pytest


def _make_params(variant: dict, log_dir: str = "/logs/exp1") -> dict:
    """Encode a variant into a params dict matching chester's internal format."""
    return {
        "variant_data": base64.b64encode(pickle.dumps(variant)).decode("utf-8"),
        "log_dir": log_dir,
    }


class TestBuildHydraArgsExtraOverrides:
    def test_no_extra_overrides_unchanged(self):
        from chester.hydra_utils import build_hydra_args
        params = _make_params({"seed": 1, "experiment.tasks": ["training"]})
        result = build_hydra_args(params)
        assert "seed=1" in result
        assert "experiment.tasks=[training]" in result

    def test_extra_overrides_replaces_key(self):
        from chester.hydra_utils import build_hydra_args
        params = _make_params({"seed": 1, "experiment.tasks": ["training"]})
        result = build_hydra_args(
            params, extra_overrides={"experiment.tasks": ["evaluate"]}
        )
        assert "experiment.tasks=[evaluate]" in result
        assert "experiment.tasks=[training]" not in result

    def test_extra_overrides_adds_new_key(self):
        from chester.hydra_utils import build_hydra_args
        params = _make_params({"seed": 1})
        result = build_hydra_args(
            params,
            extra_overrides={"experiment.evaluate.training_dir": "/some/path"},
        )
        assert "experiment.evaluate.training_dir=/some/path" in result
        assert "seed=1" in result

    def test_extra_overrides_none_is_default(self):
        from chester.hydra_utils import build_hydra_args
        params = _make_params({"seed": 1})
        assert build_hydra_args(params) == build_hydra_args(params, extra_overrides=None)

    def test_extra_overrides_empty_dict_is_noop(self):
        from chester.hydra_utils import build_hydra_args
        params = _make_params({"seed": 1})
        assert build_hydra_args(params) == build_hydra_args(params, extra_overrides={})


class TestBuildPythonCommandExtraOverrides:
    def _make_backend(self):
        from chester.backends.local import LocalBackend
        from chester.backends.base import BackendConfig
        cfg = BackendConfig(name="local", type="local")
        return LocalBackend(cfg, {"package_manager": "python"})

    def test_no_extra_overrides_unchanged(self):
        backend = self._make_backend()
        params = _make_params({"seed": 1, "experiment.tasks": ["training"]})
        cmd = backend.build_python_command(params, "main", hydra_enabled=True)
        assert "experiment.tasks=[training]" in cmd

    def test_extra_overrides_replaces_key(self):
        backend = self._make_backend()
        params = _make_params({"seed": 1, "experiment.tasks": ["training"]})
        cmd = backend.build_python_command(
            params, "main", hydra_enabled=True,
            extra_overrides={"experiment.tasks": ["evaluate"]},
        )
        assert "experiment.tasks=[evaluate]" in cmd
        assert "experiment.tasks=[training]" not in cmd

    def test_extra_overrides_none_unchanged(self):
        backend = self._make_backend()
        params = _make_params({"seed": 1})
        assert (
            backend.build_python_command(params, "main", hydra_enabled=True)
            == backend.build_python_command(
                params, "main", hydra_enabled=True, extra_overrides=None
            )
        )


class TestSSHSequentialSteps:
    def _make_task(self, log_dir="/remote/logs/exp1"):
        return {
            "params": {**_make_params({"seed": 1, "experiment.tasks": ["training"]}),
                       "exp_name": "test_exp", "log_dir": log_dir}
        }

    def _make_backend(self):
        from chester.backends.ssh import SSHBackend
        from chester.backends.base import BackendConfig
        cfg = BackendConfig(
            name="myhost", type="ssh",
            host="myhost", remote_dir="/remote/project",
        )
        return SSHBackend(cfg, {"package_manager": "python"})

    def test_no_sequential_steps_unchanged(self):
        backend = self._make_backend()
        task = self._make_task()
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=None,
        )
        assert script.count("python -m main") == 1
        assert ".done" in script

    def test_two_steps_two_commands(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert script.count("python -m main") == 2

    def test_done_marker_appears_once_after_last_step(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert script.count(".done") == 1
        last_cmd_pos = script.rfind("python -m main")
        done_pos = script.find(".done")
        assert done_pos > last_cmd_pos

    def test_step_overrides_applied(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"], "experiment.evaluate.training_dir": "/ckpt"},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert "experiment.tasks=[training]" in script
        assert "experiment.tasks=[evaluate]" in script
        assert "experiment.evaluate.training_dir=/ckpt" in script

    def test_step_comments_present(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [{"experiment.tasks": ["training"]}, {"experiment.tasks": ["evaluate"]}]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert "# step 1/2" in script
        assert "# step 2/2" in script

class TestSlurmSequentialSteps:
    def _make_task(self, log_dir="/remote/logs/exp1"):
        return {
            "params": {**_make_params({"seed": 1, "experiment.tasks": ["training"]}),
                       "exp_name": "test_exp", "log_dir": log_dir}
        }

    def _make_backend(self):
        from chester.backends.slurm import SlurmBackend
        from chester.backends.base import BackendConfig, SlurmConfig
        cfg = BackendConfig(
            name="gl", type="slurm",
            host="gl", remote_dir="/remote/project",
            slurm=SlurmConfig(partition="gpu", time="4:00:00", gpus=1),
        )
        return SlurmBackend(cfg, {"package_manager": "python"})

    def test_no_sequential_steps_unchanged(self):
        backend = self._make_backend()
        task = self._make_task()
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=None,
        )
        assert script.count("python -m main") == 1
        assert ".done" in script

    def test_two_steps_two_commands(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert script.count("python -m main") == 2

    def test_done_marker_appears_once_after_last_step(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert script.count(".done") == 1
        last_cmd_pos = script.rfind("python -m main")
        done_pos = script.find(".done")
        assert done_pos > last_cmd_pos

    def test_step_overrides_applied(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"], "experiment.evaluate.training_dir": "/ckpt"},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert "experiment.tasks=[training]" in script
        assert "experiment.tasks=[evaluate]" in script
        assert "experiment.evaluate.training_dir=/ckpt" in script

class TestLocalSequentialSteps:
    def _make_task(self, log_dir="/local/logs/exp1"):
        return {
            "params": {**_make_params({"seed": 1, "experiment.tasks": ["training"]}),
                       "exp_name": "test_exp", "log_dir": log_dir}
        }

    def _make_backend(self):
        from chester.backends.local import LocalBackend
        from chester.backends.base import BackendConfig
        cfg = BackendConfig(name="local", type="local")
        return LocalBackend(cfg, {"package_manager": "python"})

    def test_no_sequential_steps_unchanged(self):
        backend = self._make_backend()
        task = self._make_task()
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=None,
        )
        assert script.count("python -m main") == 1

    def test_two_steps_in_script(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert script.count("python -m main") == 2

    def test_step_overrides_applied_in_script(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        script = backend.generate_script(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert "experiment.tasks=[training]" in script
        assert "experiment.tasks=[evaluate]" in script

    def test_generate_command_two_steps_joined_with_and(self):
        backend = self._make_backend()
        task = self._make_task()
        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        cmd = backend.generate_command(
            task, script="main", python_command="python -m",
            hydra_enabled=True, sequential_steps=steps,
        )
        assert " && " in cmd
        assert cmd.count("python -m main") == 2


class TestRunExpSequentialSteps:
    def _make_mock_cfg(self, tmp_path):
        return {
            "project_path": str(tmp_path),
            "log_dir": str(tmp_path / "data"),
            "rsync_include": [],
            "rsync_exclude": [],
            "hydra_config_path": str(tmp_path / "configs"),
            "backends": {},
        }

    def test_empty_sequential_steps_raises(self, tmp_path, monkeypatch):
        import pytest
        from unittest import mock
        from chester.run_exp import run_experiment_lite
        monkeypatch.chdir(tmp_path)
        with mock.patch("chester.run_exp.load_config", return_value=self._make_mock_cfg(tmp_path)), \
             mock.patch("chester.run_exp.get_backend"), \
             mock.patch("chester.run_exp.create_backend"):
            with pytest.raises(ValueError, match="sequential_steps"):
                run_experiment_lite(
                    script="main", mode="local",
                    variant={"seed": 1},
                    sequential_steps=[],
                )

    def test_sequential_steps_forwarded_to_backend(self, tmp_path, monkeypatch):
        from unittest import mock
        from chester.run_exp import run_experiment_lite
        from chester.backends.base import BackendConfig
        monkeypatch.chdir(tmp_path)

        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        mock_backend = mock.MagicMock()
        mock_backend.config = BackendConfig(name="local", type="local")
        mock_backend.generate_command.return_value = "echo ok"
        mock_backend.submit.return_value = None

        with mock.patch("chester.run_exp.load_config", return_value=self._make_mock_cfg(tmp_path)), \
             mock.patch("chester.run_exp.get_backend"), \
             mock.patch("chester.run_exp.create_backend", return_value=mock_backend):
            variant = {"seed": 1, "chester_first_variant": True, "chester_last_variant": True}
            run_experiment_lite(
                script="main", mode="local", variant=variant,
                sequential_steps=steps, dry=True, hydra_enabled=True,
            )

        all_calls = (
            mock_backend.generate_command.call_args_list
            + mock_backend.generate_script.call_args_list
        )
        assert any(
            call.kwargs.get("sequential_steps") == steps
            for call in all_calls
        ), "sequential_steps not forwarded to backend"

    def test_multi_step_forces_subprocess(self, tmp_path, monkeypatch):
        """Multi-step sequential_steps must use subprocess even when launch_with_subprocess=False.

        Hydra can only be initialized once per process, and each step must be a
        separate Python process (Isaac Sim constraint), so in-process execution
        is not valid for multi-step runs.
        """
        import subprocess
        from unittest import mock
        from chester.run_exp import run_experiment_lite
        from chester.backends.base import BackendConfig
        monkeypatch.chdir(tmp_path)

        steps = [
            {"experiment.tasks": ["training"]},
            {"experiment.tasks": ["evaluate"]},
        ]
        mock_backend = mock.MagicMock()
        mock_backend.config = BackendConfig(name="local", type="local")
        mock_backend.generate_command.return_value = "echo step1 && echo step2"
        mock_backend.generate_script.return_value = "#!/bin/bash\necho step1 && echo step2"

        from chester.backends.base import BackendConfig as BC
        local_cfg = BC(name="local", type="local")
        with mock.patch("chester.run_exp.load_config", return_value=self._make_mock_cfg(tmp_path)), \
             mock.patch("chester.run_exp.get_backend", return_value=local_cfg), \
             mock.patch("chester.run_exp.create_backend", return_value=mock_backend), \
             mock.patch("chester.run_exp.subprocess.call") as mock_call:
            variant = {"seed": 1, "chester_first_variant": True, "chester_last_variant": True}
            run_experiment_lite(
                script="main", mode="local", variant=variant,
                sequential_steps=steps, hydra_enabled=True,
                launch_with_subprocess=False,  # debug mode — would normally go in-process
                wait_subprocess=True,
                auto_pull=False,
            )

        # Must have gone through subprocess.call, not run_hydra_command
        mock_call.assert_called_once()
