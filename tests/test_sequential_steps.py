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
