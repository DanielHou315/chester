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
