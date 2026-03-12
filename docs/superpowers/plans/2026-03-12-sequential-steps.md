# Sequential Steps Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `sequential_steps: list[dict] | None` to `run_experiment_lite` so multiple Python commands (e.g. training then evaluate) run in series in one SLURM/SSH job.

**Architecture:** Each backend's `generate_script()` is refactored to call a step-command loop that merges per-step overrides into the variant before building each command. `build_python_command()` in `base.py` gains an `extra_overrides` param that merges a step's dict on top of the decoded variant before generating Hydra args. A single `.done` marker is written after all steps complete.

**Tech Stack:** Python, pytest. All files in `src/chester/`. Tests in `tests/`.

**Spec:** `docs/superpowers/specs/2026-03-12-sequential-steps-design.md`

---

## Chunk 1: Extend base command builder to accept per-step overrides

### Task 1: Add `extra_overrides` to `build_hydra_args` and `build_python_command`

**Files:**
- Modify: `src/chester/backends/base.py` (`build_python_command`)
- Modify: `src/chester/hydra_utils.py` (`build_hydra_args`)
- Test: `tests/test_sequential_steps.py` (new file)

Chester already uses cloudpickle/pickle internally for variant serialization (existing
pattern). Test helpers mirror that pattern. The goal is to allow a step's override dict
(e.g. `{"experiment.tasks": ["evaluate"]}`) to be merged on top of the base variant when
generating a command, without re-encoding anything.

- [ ] **Step 1: Write failing tests for `build_hydra_args` with `extra_overrides`**

Create `tests/test_sequential_steps.py` (note: `_make_params` uses the same
base64+pickle encoding that chester uses internally for `variant_data`):

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
uv run pytest tests/test_sequential_steps.py -v 2>&1 | head -40
```

Expected: `TypeError` or `FAILED` — `build_hydra_args` does not accept `extra_overrides`.

- [ ] **Step 3: Add `extra_overrides` to `build_hydra_args` in `hydra_utils.py`**

Change signature:
```python
def build_hydra_args(
    params: Dict[str, Any],
    hydra_flags: Dict[str, Any] = None,
    extra_overrides: Dict[str, Any] = None,
) -> str:
```

After `variant_data = pickle.loads(base64.b64decode(params["variant_data"]))`, add:
```python
    if extra_overrides:
        variant_data.update(extra_overrides)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
uv run pytest tests/test_sequential_steps.py::TestBuildHydraArgsExtraOverrides -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Write failing tests for `build_python_command` with `extra_overrides`**

Add to `tests/test_sequential_steps.py`:

```python
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
```

- [ ] **Step 6: Run tests — verify they fail**

```bash
uv run pytest tests/test_sequential_steps.py::TestBuildPythonCommandExtraOverrides -v
```

Expected: `TypeError` — `build_python_command` does not accept `extra_overrides`.

- [ ] **Step 7: Add `extra_overrides` to `build_python_command` in `base.py`**

Change signature to add `extra_overrides: Optional[Dict[str, Any]] = None` as the last param.

In the `hydra_enabled` branch, pass it through:
```python
        if hydra_enabled:
            from ..hydra_utils import build_hydra_args
            args = build_hydra_args(params, hydra_flags, extra_overrides=extra_overrides)
```

- [ ] **Step 8: Run all sequential_steps tests — verify they pass**

```bash
uv run pytest tests/test_sequential_steps.py -v
```

Expected: 8 PASSED.

- [ ] **Step 9: Run full test suite — verify no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 10: Commit**

```bash
git add src/chester/hydra_utils.py src/chester/backends/base.py tests/test_sequential_steps.py
git commit -m "feat(sequential-steps): add extra_overrides to build_hydra_args and build_python_command"
```

---

## Chunk 2: SSH backend — multi-step `generate_script`

### Task 2: SSH backend generates N commands for N steps

**Files:**
- Modify: `src/chester/backends/ssh.py` (`generate_script`)
- Test: `tests/test_sequential_steps.py` (extend)

The `.done` marker must appear after the LAST step only. `set -e` aborts on first failure.

- [ ] **Step 1: Write failing SSH multi-step tests**

Add to `tests/test_sequential_steps.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_sequential_steps.py::TestSSHSequentialSteps -v
```

Expected: `TypeError` — `generate_script` does not accept `sequential_steps`.

- [ ] **Step 3: Refactor `ssh.py` `generate_script` to support sequential steps**

Add `sequential_steps: Optional[List[Dict[str, Any]]] = None` to the signature.

Replace the single-command block (the `inner` list construction and `command` call) with:

```python
        steps = sequential_steps if sequential_steps is not None else [None]

        inner: List[str] = []
        if self.config.singularity:
            inner.extend(self.get_singularity_prepare_commands())

        for i, step_overrides in enumerate(steps):
            if len(steps) > 1:
                inner.append(f"# step {i + 1}/{len(steps)}")
            command = self.build_python_command(
                params, script, python_command, env,
                hydra_enabled, hydra_flags,
                extra_overrides=step_overrides,
            )
            inner.append(command)

        # .done marker — written once, after all steps complete
        inner.append(f"touch {log_dir}/.done")

        if self.config.singularity:
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.extend(inner)
```

- [ ] **Step 4: Run SSH tests — verify they pass**

```bash
uv run pytest tests/test_sequential_steps.py::TestSSHSequentialSteps -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Run full suite — no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/chester/backends/ssh.py tests/test_sequential_steps.py
git commit -m "feat(sequential-steps): SSH backend generates N commands for N steps"
```

---

## Chunk 3: SLURM backend — multi-step `generate_script`

### Task 3: SLURM backend generates N commands for N steps

**Files:**
- Modify: `src/chester/backends/slurm.py` (`generate_script`)
- Test: `tests/test_sequential_steps.py` (extend)

Identical pattern to SSH. `.done` after last step only.

- [ ] **Step 1: Write failing SLURM multi-step tests**

Add to `tests/test_sequential_steps.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_sequential_steps.py::TestSlurmSequentialSteps -v
```

Expected: `TypeError` — `generate_script` does not accept `sequential_steps`.

- [ ] **Step 3: Refactor `slurm.py` `generate_script` to support sequential steps**

Add `sequential_steps: Optional[List[Dict[str, Any]]] = None` to the signature.

Replace the single-command block with:

```python
        steps = sequential_steps if sequential_steps is not None else [None]

        inner: List[str] = []
        if self.config.singularity:
            inner.extend(self.get_singularity_prepare_commands())

        for i, step_overrides in enumerate(steps):
            if len(steps) > 1:
                inner.append(f"# step {i + 1}/{len(steps)}")
            command = self.build_python_command(
                params, script, python_command, env,
                hydra_enabled, hydra_flags,
                extra_overrides=step_overrides,
            )
            inner.append(command)

        # .done marker — written once, after all steps complete
        inner.append(f"touch {log_dir}/.done")

        if self.config.singularity:
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.extend(inner)
```

- [ ] **Step 4: Run SLURM tests — verify they pass**

```bash
uv run pytest tests/test_sequential_steps.py::TestSlurmSequentialSteps -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Run full suite — no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 6: Commit**

```bash
git add src/chester/backends/slurm.py tests/test_sequential_steps.py
git commit -m "feat(sequential-steps): SLURM backend generates N commands for N steps"
```

---

## Chunk 4: Local backend — multi-step commands

### Task 4: Local backend generates N commands for N steps

**Files:**
- Modify: `src/chester/backends/local.py` (`generate_script`, `generate_command`)
- Test: `tests/test_sequential_steps.py` (extend)

Local `generate_script` already has `set -e`. No `.done` marker for local execution.
`generate_command` chains with `&&`.

- [ ] **Step 1: Write failing local multi-step tests**

Add to `tests/test_sequential_steps.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_sequential_steps.py::TestLocalSequentialSteps -v
```

Expected: `TypeError`.

- [ ] **Step 3: Refactor `local.py` `generate_script` to support sequential steps**

Add `sequential_steps: Optional[List[Dict[str, Any]]] = None` to `generate_script` signature.

Replace the single-command block:

```python
        params = task.get("params", {})
        steps = sequential_steps if sequential_steps is not None else [None]
        commands = [
            self.build_python_command(
                params, script, python_command, env,
                hydra_enabled, hydra_flags,
                extra_overrides=step_overrides,
            )
            for step_overrides in steps
        ]

        if self.config.singularity:
            inner: List[str] = list(self.get_singularity_prepare_commands())
            inner.extend(commands)
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.extend(commands)
```

- [ ] **Step 4: Refactor `local.py` `generate_command` to support sequential steps**

Add `sequential_steps: Optional[List[Dict[str, Any]]] = None` to `generate_command` signature.

Replace the single `command = self.build_python_command(...)` call with:

```python
        params = task.get("params", {})
        steps = sequential_steps if sequential_steps is not None else [None]
        commands = [
            self.build_python_command(
                params, script, python_command, env,
                hydra_enabled, hydra_flags,
                extra_overrides=step_overrides,
            )
            for step_overrides in steps
        ]
        command = " && ".join(commands)
```

Then pass `command` into the existing singularity-wrapping logic unchanged.

- [ ] **Step 5: Run local tests — verify they pass**

```bash
uv run pytest tests/test_sequential_steps.py::TestLocalSequentialSteps -v
```

Expected: 4 PASSED.

- [ ] **Step 6: Run full suite — no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

- [ ] **Step 7: Commit**

```bash
git add src/chester/backends/local.py tests/test_sequential_steps.py
git commit -m "feat(sequential-steps): local backend generates N commands for N steps"
```

---

## Chunk 5: Thread `sequential_steps` through `run_experiment_lite`

### Task 5: Accept and pass through `sequential_steps` in `run_experiment_lite`

**Files:**
- Modify: `src/chester/run_exp.py` (`run_experiment_lite`)
- Test: `tests/test_sequential_steps.py` (extend)

- [ ] **Step 1: Write failing integration tests**

Add to `tests/test_sequential_steps.py`:

```python
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
uv run pytest tests/test_sequential_steps.py::TestRunExpSequentialSteps -v
```

Expected: `TypeError` or assertion errors.

- [ ] **Step 3: Add `sequential_steps` to `run_experiment_lite` signature**

Add after `fresh=False,`:

```python
        sequential_steps: list = None,
```

Add validation immediately after the `if variations is None` block:

```python
    if sequential_steps is not None and len(sequential_steps) == 0:
        raise ValueError(
            "sequential_steps must be None or a non-empty list. "
            "Pass None (default) for single-step behavior."
        )
```

Pass to local backend `generate_command` call:

```python
            command = backend.generate_command(
                backend_task,
                script=script,
                python_command=python_command,
                env=merged_env or None,
                hydra_enabled=hydra_enabled,
                hydra_flags=hydra_flags,
                sequential_steps=sequential_steps,
            )
```

Pass to remote backend `generate_script` call (add to `gen_kwargs` before the call):

```python
            gen_kwargs["sequential_steps"] = sequential_steps
            script_content = backend.generate_script(backend_task, **gen_kwargs)
```

- [ ] **Step 4: Run all sequential_steps tests**

```bash
uv run pytest tests/test_sequential_steps.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Run full test suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/chester/run_exp.py tests/test_sequential_steps.py
git commit -m "feat(sequential-steps): thread sequential_steps through run_experiment_lite"
```

---

## Chunk 6: Final cleanup and launcher example

### Task 6: Add usage example to launcher

**Files:**
- Modify: `launchers/launch_cotrain_ppo.py` (cotrain_dynamics repo)

- [ ] **Step 1: Add usage comment to `launch_cotrain_ppo.py`**

After the existing module docstring in
`/home/houhd/code/cotrain_dynamics/launchers/launch_cotrain_ppo.py`, add:

```python
# To run training then evaluate in one job (single SLURM/SSH allocation):
#   sequential_steps=[
#       {"experiment.tasks": ["training"]},
#       {"experiment.tasks": ["evaluate"],
#        "experiment.evaluate.training_dir": "${hydra:runtime.output_dir}"},
#   ]
# Pass this as sequential_steps=... to run_experiment_lite.
```

- [ ] **Step 2: Run full test suite one final time**

```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
uv run pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 3: Commit launcher comment (cotrain_dynamics repo)**

```bash
cd /home/houhd/code/cotrain_dynamics
git add launchers/launch_cotrain_ppo.py
git commit -m "docs: add sequential_steps usage example in launch_cotrain_ppo.py"
```

- [ ] **Step 4: Verify chester repo is fully committed**

```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
git status
```

Expected: clean working tree.
