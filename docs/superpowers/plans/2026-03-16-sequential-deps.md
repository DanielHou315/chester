# Sequential SLURM Dependencies Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `sequential=True` flag to `vg.add()` so variants within a sequential field form SLURM job dependency chains (`sbatch --dependency=afterok:<jobid>`), and unify all confirmation prompts behind `confirm_action()`.

**Architecture:** `VariantGenerator` gains metadata about which fields are sequential. Variants carry `_chester_seq_identity` and `_chester_pred_identities` metadata so `run_experiment_lite` can look up predecessor job IDs from a module-level registry. The SLURM backend's `submit()` accepts an optional `dependency_job_ids` list to add `--dependency=afterok:...` to the sbatch command. Non-SLURM modes raise `ValueError` (unless `skip_dependency_check=True`). The registry is cleared on `first_variant` to prevent cross-sweep leakage.

**Tech Stack:** Python, sbatch CLI, pytest

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/chester/run_exp.py` | `VariantGenerator.add()` sequential flag, `confirm_action()`, dependency tracking in `run_experiment_lite`, remove `sequential_steps` param |
| `src/chester/backends/slurm.py` | `submit()` accepts `dependency_job_ids`, remove `sequential_steps` from `generate_script()` |
| `src/chester/backends/local.py` | Remove `sequential_steps` from `generate_command()` and `generate_script()` |
| `src/chester/backends/ssh.py` | Remove `sequential_steps` from `generate_script()`, fix syntax error |
| `src/chester/backends/base.py` | Remove `sequential_steps` from `build_python_command()` docstring if mentioned (keep `extra_overrides` — it's general-purpose) |
| `tests/test_sequential_deps.py` | New test file for sequential dependency feature |
| `tests/test_confirm_action.py` | New test file for `confirm_action()` |
| `tests/test_sequential_steps.py` | Delete entirely |

---

## Chunk 1: Cleanup & confirm_action()

### Task 1: Fix SSH backend syntax error

The `else:` on line 114 of `ssh.py` is at the wrong indentation level — it's a dangling else after an already-closed if/else block. This is a pre-existing bug causing all SSH-related tests to fail with `SyntaxError`.

**Files:**
- Modify: `src/chester/backends/ssh.py:101-115`

- [ ] **Step 1: Fix the indentation bug**

The block at lines 101-115 currently reads:
```python
        else:
            inner: List[str] = []
            if self.config.singularity:
                inner.extend(self.get_singularity_prepare_commands())
            for i, step_overrides in enumerate(steps):
                command = self.build_python_command(...)
                inner.append(command)
            if self.config.singularity:
                lines.append(self.wrap_with_singularity(inner))
        else:                          # <-- BUG: dangling else
            lines.extend(inner)
```

Fix: the second `else:` should be part of the inner `if self.config.singularity:` block:
```python
        else:
            inner: List[str] = []
            if self.config.singularity:
                inner.extend(self.get_singularity_prepare_commands())
            for i, step_overrides in enumerate(steps):
                command = self.build_python_command(...)
                inner.append(command)
            if self.config.singularity:
                lines.append(self.wrap_with_singularity(inner))
            else:
                lines.extend(inner)
```

- [ ] **Step 2: Verify SSH imports work**

Run: `uv run python -c "from chester.backends.ssh import SSHBackend; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/chester/backends/ssh.py
git commit -m "fix(ssh): fix dangling else syntax error in generate_script"
```

### Task 2: Add `confirm_action()` and migrate callers

**Files:**
- Modify: `src/chester/run_exp.py` (add `confirm_action`, update `_fresh_start_v2`, update `run_experiment_lite`)
- Create: `tests/test_confirm_action.py`

- [ ] **Step 1: Write tests for `confirm_action()`**

```python
# tests/test_confirm_action.py
from unittest.mock import patch
from chester.run_exp import confirm_action


def test_confirm_action_skip():
    """When skip=True, return True without prompting."""
    assert confirm_action("Delete?", skip=True) is True


def test_confirm_action_yes(monkeypatch):
    """User types 'yes' -> returns True."""
    monkeypatch.setattr('builtins.input', lambda _="": 'yes')
    assert confirm_action("Delete?") is True


def test_confirm_action_y(monkeypatch):
    """User types 'y' -> returns True."""
    monkeypatch.setattr('builtins.input', lambda _="": 'y')
    assert confirm_action("Delete?") is True


def test_confirm_action_no(monkeypatch):
    """User types 'no' -> returns False."""
    monkeypatch.setattr('builtins.input', lambda _="": 'no')
    assert confirm_action("Delete?") is False


def test_confirm_action_default_yes(monkeypatch):
    """Empty input with default='yes' -> returns True."""
    monkeypatch.setattr('builtins.input', lambda _="": '')
    assert confirm_action("Delete?", default="yes") is True


def test_confirm_action_default_no(monkeypatch):
    """Empty input with default='no' -> returns False."""
    monkeypatch.setattr('builtins.input', lambda _="": '')
    assert confirm_action("Delete?", default="no") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_confirm_action.py -v`
Expected: FAIL (confirm_action not yet defined or different signature)

- [ ] **Step 3: Implement `confirm_action()`**

Add to `run_exp.py`, replacing `query_yes_no`:

```python
def confirm_action(message: str, default: str = "yes", skip: bool = False) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        message: The question to display.
        default: Presumed answer on empty input ("yes", "no", or None for required).
        skip: If True, return True without prompting.

    Returns:
        True if confirmed, False if denied.
    """
    if skip:
        return True

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: '{default}'")

    while True:
        choice = input(message + prompt).lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def query_yes_no(question, default="yes"):
    """Deprecated: use confirm_action() instead."""
    import warnings
    warnings.warn("query_yes_no() is deprecated, use confirm_action()", DeprecationWarning, stacklevel=2)
    return confirm_action(question, default=default)
```

- [ ] **Step 4: Migrate `_fresh_start_v2` to use `confirm_action()`**

Replace the raw `input()` block (lines 731-735):
```python
    # OLD:
    print("WARNING: This will permanently delete ALL directories listed above.")
    answer = input("Type 'yes' or 'y' to confirm: ").strip().lower()
    if answer not in ('yes', 'y'):
        print("Aborted.")
        sys.exit(0)

    # NEW:
    if not confirm_action("WARNING: This will permanently delete ALL directories listed above."):
        print("Aborted.")
        sys.exit(0)
```

- [ ] **Step 5: Migrate remote confirmation in `run_experiment_lite` to use `confirm_action()`**

Replace lines 1026-1030:
```python
    # OLD:
    if is_remote and not remote_confirmed and not dry and not confirm:
        remote_confirmed = query_yes_no(
            "Running in (non-dry) mode %s. Confirm?" % mode)
        if not remote_confirmed:
            sys.exit(1)

    # NEW:
    if is_remote and not remote_confirmed and not dry:
        remote_confirmed = confirm_action(
            f"Running in (non-dry) mode {mode}. Confirm?",
            skip=confirm,
        )
        if not remote_confirmed:
            sys.exit(1)
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_confirm_action.py tests/test_fresh_start.py -v`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/chester/run_exp.py tests/test_confirm_action.py
git commit -m "refactor: unify confirmation prompts with confirm_action()"
```

### Task 3: Remove `sequential_steps` from all backends and `run_experiment_lite`

**Files:**
- Modify: `src/chester/backends/local.py`
- Modify: `src/chester/backends/ssh.py`
- Modify: `src/chester/backends/slurm.py`
- Modify: `src/chester/run_exp.py`
- Delete: `tests/test_sequential_steps.py`

- [ ] **Step 1: Remove `sequential_steps` from `LocalBackend.generate_command()`**

Remove the `sequential_steps` parameter. Simplify to always build a single command:
```python
    def generate_command(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        params = task.get("params", {})
        command = self.build_python_command(
            params, script, python_command, env,
            hydra_enabled, hydra_flags,
        )
        # ... rest stays the same but without steps loop
```

Do the same for `LocalBackend.generate_script()`.

- [ ] **Step 2: Remove `sequential_steps` from `SSHBackend.generate_script()`**

Remove the parameter and simplify the body. The singularity multi-step logic goes away entirely:
```python
    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        remote_dir = self.config.remote_dir or "./"

        lines: List[str] = []
        lines.append("#!/usr/bin/env bash")
        lines.append(f"exec 19>{log_dir}/chester_xtrace.log")
        lines.append("BASH_XTRACEFD=19")
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")
        lines.append(f"cd {remote_dir}")

        prepare_cmds = self.get_prepare_commands()
        lines.extend(prepare_cmds)
        lines.extend(self.get_overlay_setup_commands())

        command = self.build_python_command(
            params, script, python_command, env,
            hydra_enabled, hydra_flags,
        )

        if self.config.singularity:
            inner = list(self.get_singularity_prepare_commands()) + [command]
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.append(command)

        lines.append(f"touch {log_dir}/.done")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 3: Remove `sequential_steps` from `SlurmBackend.generate_script()`**

Same simplification — remove the parameter and multi-step loop:
```python
    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        slurm_overrides: Optional[Dict[str, Any]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        # ... header and preamble unchanged ...

        prepare_cmds = self.get_prepare_commands()
        lines.extend(prepare_cmds)
        lines.extend(self.get_overlay_setup_commands())

        command = self.build_python_command(
            params, script, python_command, env,
            hydra_enabled, hydra_flags,
        )

        if self.config.singularity:
            inner = list(self.get_singularity_prepare_commands()) + [command]
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.append(command)

        lines.append(f"touch {log_dir}/.done")
        return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Remove `sequential_steps` from `run_experiment_lite()`**

Remove these from `run_experiment_lite()`:
1. The `sequential_steps` parameter (line 814)
2. The validation block (lines 863-867)
3. The `sequential_steps=sequential_steps` kwarg passed to `backend.generate_command()` (line 1077)
4. The `gen_kwargs["sequential_steps"] = sequential_steps` line (line 1128)
5. The `multi_step` logic that forces subprocess mode (lines 1090-1091) — simplify to:
   ```python
   use_subprocess = launch_with_subprocess or backend.config.singularity
   ```

- [ ] **Step 5: Delete `tests/test_sequential_steps.py`**

```bash
git rm tests/test_sequential_steps.py
```

- [ ] **Step 6: Run all tests**

Run: `uv run pytest tests/ -v --ignore=tests/test_sequential_steps.py`
Expected: All previously-passing tests still pass. SSH tests should now pass too (after Task 1 fix).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: remove sequential_steps from all backends and run_experiment_lite"
```

---

## Chunk 2: Sequential dependency feature

### Task 4: Add `sequential` flag to `VariantGenerator.add()` and dependency computation

**Files:**
- Modify: `src/chester/run_exp.py` (VariantGenerator class)
- Create: `tests/test_sequential_deps.py`

- [ ] **Step 1: Write tests for VariantGenerator sequential metadata**

```python
# tests/test_sequential_deps.py
import pytest
from chester.run_exp import VariantGenerator


class TestSequentialMetadata:
    def test_get_sequential_keys_none(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        vg.add("seed", [1, 2])
        assert vg.get_sequential_keys() == []

    def test_get_sequential_keys_one(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        assert vg.get_sequential_keys() == ["task"]

    def test_get_sequential_keys_multiple(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        vg.add("phase", ["warmup", "finetune"], sequential=True)
        assert set(vg.get_sequential_keys()) == {"task", "phase"}

    def test_sequential_single_value_raises(self):
        """sequential=True with a single value is pointless and likely a mistake."""
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="at least 2 values"):
            vg.add("task", ["training"], sequential=True)

    def test_sequential_with_callable_raises(self):
        """sequential=True with a callable (lambda dependency) is not supported."""
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="cannot be used with callable"):
            vg.add("task", lambda seed: ["train", "eval"], sequential=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sequential_deps.py::TestSequentialMetadata -v`
Expected: FAIL

- [ ] **Step 3: Implement `get_sequential_keys()` and validation in `add()`**

In `VariantGenerator`:

```python
    def add(self, key, vals, **kwargs):
        if kwargs.get("sequential"):
            if callable(vals) and not isinstance(vals, list):
                raise ValueError(
                    f"sequential=True on '{key}' cannot be used with callable values. "
                    f"Provide a concrete list instead."
                )
            if isinstance(vals, list) and len(vals) < 2:
                raise ValueError(
                    f"sequential=True on '{key}' requires at least 2 values, got {len(vals)}"
                )
        self._variants.append((key, vals, kwargs))

    def get_sequential_keys(self) -> list:
        """Return keys marked with sequential=True."""
        return [k for k, _, cfg in self._variants if cfg.get("sequential")]
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sequential_deps.py::TestSequentialMetadata -v`
Expected: All PASS

- [ ] **Step 5: Write tests for `get_dependency_map()`**

```python
class TestDependencyMap:
    def test_no_sequential_keys(self):
        """No sequential keys -> empty dependency map."""
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)
        assert dep_map == {}

    def test_single_sequential_field(self):
        """task=evaluate depends on task=training (same seed)."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        # Find indices
        def find(task, seed):
            for i, v in enumerate(variants):
                if v["task"] == task and v["seed"] == seed:
                    return i
            raise ValueError(f"Not found: task={task}, seed={seed}")

        # training variants have no dependencies
        assert find("training", 1) not in dep_map
        assert find("training", 2) not in dep_map
        # evaluate variants depend on corresponding training
        assert dep_map[find("evaluate", 1)] == [find("training", 1)]
        assert dep_map[find("evaluate", 2)] == [find("training", 2)]

    def test_two_sequential_fields(self):
        """Per-field independent chains with two sequential fields."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("phase", ["warmup", "finetune"], sequential=True)
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(**kw):
            for i, v in enumerate(variants):
                if all(v[k] == val for k, val in kw.items()):
                    return i
            raise ValueError(f"Not found: {kw}")

        # (training, warmup) -> no deps
        assert find(task="training", phase="warmup") not in dep_map

        # (evaluate, warmup) -> depends on (training, warmup)
        assert dep_map[find(task="evaluate", phase="warmup")] == [
            find(task="training", phase="warmup")
        ]

        # (training, finetune) -> depends on (training, warmup)
        assert dep_map[find(task="training", phase="finetune")] == [
            find(task="training", phase="warmup")
        ]

        # (evaluate, finetune) -> depends on (training, finetune) AND (evaluate, warmup)
        deps = set(dep_map[find(task="evaluate", phase="finetune")])
        assert deps == {
            find(task="training", phase="finetune"),
            find(task="evaluate", phase="warmup"),
        }

    def test_three_value_chain(self):
        """a -> b -> c: b depends on a, c depends on b (not a)."""
        vg = VariantGenerator()
        vg.add("stage", ["prep", "train", "eval"], sequential=True)
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(stage):
            for i, v in enumerate(variants):
                if v["stage"] == stage:
                    return i

        assert find("prep") not in dep_map
        assert dep_map[find("train")] == [find("prep")]
        assert dep_map[find("eval")] == [find("train")]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `uv run pytest tests/test_sequential_deps.py::TestDependencyMap -v`
Expected: FAIL

- [ ] **Step 7: Implement `get_dependency_map()`**

In `VariantGenerator`:

```python
    def get_dependency_map(self, variants: list) -> dict:
        """Compute inter-variant dependency map based on sequential fields.

        For each sequential key, a variant's predecessor is the variant that is
        identical except the sequential field has the previous value in the list.

        Args:
            variants: List of variant dicts (from self.variants()).

        Returns:
            Dict mapping variant index -> list of predecessor variant indices.
            Variants with no predecessors are omitted.
        """
        seq_keys = self.get_sequential_keys()
        if not seq_keys:
            return {}

        # Build value ordering for each sequential key
        seq_val_order = {}
        seq_val_list = {}
        for key, vals, cfg in self._variants:
            if cfg.get("sequential") and isinstance(vals, list):
                seq_val_order[key] = {v: i for i, v in enumerate(vals)}
                seq_val_list[key] = vals

        # Index variants by their identity tuple (all field values)
        all_keys = [k for k, _, _ in self._variants]
        variant_index = {}
        for i, v in enumerate(variants):
            identity = tuple(v.get(k) for k in all_keys)
            variant_index[identity] = i

        dep_map = {}
        for i, v in enumerate(variants):
            predecessors = []
            identity = list(v.get(k) for k in all_keys)

            for seq_key in seq_keys:
                val = v[seq_key]
                order = seq_val_order[seq_key]
                val_idx = order.get(val, 0)
                if val_idx == 0:
                    continue  # first value, no predecessor for this key

                # Find the previous value
                prev_val = seq_val_list[seq_key][val_idx - 1]

                # Build predecessor identity: same as current but with prev_val for this key
                key_pos = all_keys.index(seq_key)
                pred_identity = list(identity)
                pred_identity[key_pos] = prev_val
                pred_idx = variant_index.get(tuple(pred_identity))
                if pred_idx is not None:
                    predecessors.append(pred_idx)

            if predecessors:
                dep_map[i] = predecessors

        return dep_map
```

- [ ] **Step 8: Run tests**

Run: `uv run pytest tests/test_sequential_deps.py -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/chester/run_exp.py tests/test_sequential_deps.py
git commit -m "feat(VariantGenerator): add sequential=True flag and dependency map computation"
```

### Task 5: Add `_chester_seq_identity` to variants and variant naming

When `sequential=True` fields exist, each variant needs an identity tuple so `run_experiment_lite` can match variants to their predecessors' SLURM job IDs across separate calls.

**Files:**
- Modify: `src/chester/run_exp.py` (VariantGenerator.variants())
- Modify: `tests/test_sequential_deps.py`

- [ ] **Step 1: Write tests for variant metadata**

```python
class TestVariantSequentialMetadata:
    def test_variants_carry_seq_identity(self):
        """Variants should carry _chester_seq_identity when sequential keys exist."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" in v
            # Identity is a tuple of (key, value) pairs for all non-hidden keys
            assert isinstance(v["_chester_seq_identity"], tuple)

    def test_variants_carry_pred_identities(self):
        """Variants should carry _chester_pred_identities listing predecessor identities."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_pred_identities" in v

        # training variants have empty pred list
        for v in variants:
            if v["task"] == "training":
                assert v["_chester_pred_identities"] == []
            else:
                assert len(v["_chester_pred_identities"]) == 1

    def test_no_seq_metadata_when_no_sequential(self):
        """No sequential keys -> no _chester metadata on variants."""
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" not in v
            assert "_chester_pred_identities" not in v
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sequential_deps.py::TestVariantSequentialMetadata -v`
Expected: FAIL

- [ ] **Step 3: Implement metadata injection in `variants()`**

Modify `VariantGenerator.variants()` to attach metadata after derivations:

```python
    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        ret = list(map(AttrDict, ret))
        for variant in ret:
            for key, fn in self._derivations:
                variant[key] = fn(variant)

        # Sequential dependency metadata
        seq_keys = self.get_sequential_keys()
        if seq_keys:
            dep_map = self.get_dependency_map(ret)
            all_keys = [k for k, _, _ in self._variants]
            for i, v in enumerate(ret):
                identity = tuple((k, v.get(k)) for k in all_keys)
                v["_chester_seq_identity"] = identity
                v["_chester_pred_identities"] = [
                    tuple((k, ret[j].get(k)) for k in all_keys)
                    for j in dep_map.get(i, [])
                ]

        # Reject randomized=True when sequential keys exist
        if seq_keys and randomized:
            raise ValueError(
                "variants(randomized=True) cannot be used with sequential fields. "
                "Sequential dependencies require deterministic variant ordering."
            )

        ret[0]['chester_first_variant'] = True
        ret[-1]['chester_last_variant'] = True
        return ret
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sequential_deps.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/chester/run_exp.py tests/test_sequential_deps.py
git commit -m "feat(VariantGenerator): attach dependency metadata to variants"
```

### Task 6: SLURM backend `submit()` with dependency support

**Files:**
- Modify: `src/chester/backends/slurm.py`
- Modify: `tests/test_backend_slurm.py`

- [ ] **Step 1: Write test for sbatch dependency flag**

Add to `tests/test_backend_slurm.py`:

```python
class TestSlurmDependency:
    def test_submit_with_dependency_ids(self, slurm_backend, tmp_path, monkeypatch):
        """submit() should pass --dependency=afterok:id1:id2 to sbatch."""
        task = {
            "params": {"log_dir": "/remote/logs/exp1", "exp_name": "test_job"},
            "_local_log_dir": str(tmp_path / "local_logs"),
        }

        captured_commands = []

        def fake_run(cmd, **kwargs):
            captured_commands.append(cmd)
            result = type('Result', (), {
                'returncode': 0,
                'stdout': 'Submitted batch job 12345\n',
                'stderr': '',
            })()
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        job_id = slurm_backend.submit(
            task, "#!/bin/bash\necho hi\n",
            dependency_job_ids=[100, 200],
        )

        # Find the sbatch command
        sbatch_cmd = [c for c in captured_commands if any("sbatch" in str(x) for x in c)]
        assert len(sbatch_cmd) == 1
        sbatch_str = " ".join(str(x) for x in sbatch_cmd[0])
        assert "--dependency=afterok:100:200" in sbatch_str

    def test_submit_without_dependency(self, slurm_backend, tmp_path, monkeypatch):
        """submit() without dependency_job_ids should not add --dependency."""
        task = {
            "params": {"log_dir": "/remote/logs/exp1", "exp_name": "test_job"},
            "_local_log_dir": str(tmp_path / "local_logs"),
        }

        captured_commands = []

        def fake_run(cmd, **kwargs):
            captured_commands.append(cmd)
            result = type('Result', (), {
                'returncode': 0,
                'stdout': 'Submitted batch job 12345\n',
                'stderr': '',
            })()
            return result

        monkeypatch.setattr("subprocess.run", fake_run)

        slurm_backend.submit(task, "#!/bin/bash\necho hi\n")

        sbatch_cmd = [c for c in captured_commands if any("sbatch" in str(x) for x in c)]
        sbatch_str = " ".join(str(x) for x in sbatch_cmd[0])
        assert "--dependency" not in sbatch_str
```

Note: this test may need a `slurm_backend` fixture. Check `tests/test_backend_slurm.py` for an existing fixture and reuse it. If none exists, create one:

```python
@pytest.fixture
def slurm_backend():
    from chester.backends.base import BackendConfig, SlurmConfig
    from chester.backends.slurm import SlurmBackend
    config = BackendConfig(
        name="test_slurm", type="slurm",
        host="testhost", remote_dir="/remote/project",
        slurm=SlurmConfig(partition="gpu", time="1:00:00"),
    )
    return SlurmBackend(config, {"project_path": "/local/project"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backend_slurm.py::TestSlurmDependency -v`
Expected: FAIL

- [ ] **Step 3: Implement `dependency_job_ids` in `SlurmBackend.submit()`**

Modify `submit()` to accept the new parameter and modify the sbatch command:

```python
    def submit(
        self,
        task: Dict[str, Any],
        script_content: str,
        dry: bool = False,
        dependency_job_ids: Optional[List[int]] = None,
    ) -> Optional[int]:
        # ... (existing code until step 4. Submit via sbatch) ...

        # 4. Submit via sbatch
        sbatch_cmd = f"sbatch"
        if dependency_job_ids:
            dep_str = ":".join(str(jid) for jid in dependency_job_ids)
            sbatch_cmd += f" --dependency=afterok:{dep_str}"
            print(f"[chester] Job depends on SLURM jobs: {dependency_job_ids}")
        sbatch_cmd += f" {shlex.quote(remote_script)}"

        print(f"[chester] Submitting SLURM job on {host}: {exp_name}")
        print(f"[chester] Remote script: {remote_script}")
        result = subprocess.run(
            ["ssh", host, sbatch_cmd],
            capture_output=True,
            text=True,
        )
        # ... rest unchanged ...
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_backend_slurm.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/chester/backends/slurm.py tests/test_backend_slurm.py
git commit -m "feat(slurm): support --dependency=afterok in submit()"
```

### Task 7: Wire dependencies into `run_experiment_lite()`

**Files:**
- Modify: `src/chester/run_exp.py`
- Modify: `tests/test_sequential_deps.py`

- [ ] **Step 1: Write tests for dependency wiring**

```python
class TestRunExpDependencyWiring:
    def test_non_slurm_with_sequential_raises(self):
        """Non-SLURM mode with sequential deps should raise ValueError."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        variants = vg.variants()

        # The first variant should trigger the check and raise
        with pytest.raises(ValueError, match="sequential dependencies"):
            run_experiment_lite(
                stub_method_call=lambda v, log_dir, exp_name: None,
                variant=variants[0],
                mode="local",
                exp_prefix="test",
                skip_dependency_check=False,
                dry=True,
            )

    def test_non_slurm_with_skip_does_not_exit(self):
        """skip_dependency_check=True should suppress the error."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        variants = vg.variants()

        # Should not raise
        run_experiment_lite(
            stub_method_call=lambda v, log_dir, exp_name: None,
            variant=variants[0],
            mode="local",
            exp_prefix="test",
            skip_dependency_check=True,
            dry=True,
        )

    def test_no_sequential_no_warning(self):
        """No sequential fields -> no dependency check, no exit."""
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()

        # Should not raise regardless of skip_dependency_check
        run_experiment_lite(
            stub_method_call=lambda v, log_dir, exp_name: None,
            variant=variants[0],
            mode="local",
            exp_prefix="test",
            skip_dependency_check=False,
            dry=True,
        )
```

Note: These tests require a `.chester/config.yaml` in the test directory. Check existing test fixtures for how config is set up (likely via `monkeypatch` or test `chester.yaml`). Adapt accordingly.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sequential_deps.py::TestRunExpDependencyWiring -v`
Expected: FAIL

- [ ] **Step 3: Implement dependency wiring in `run_experiment_lite()`**

Add module-level job ID registry:
```python
# Module-level registry: (exp_prefix, seq_identity) -> slurm_job_id
_slurm_job_registry: dict = {}
```

Add `skip_dependency_check=False` parameter to `run_experiment_lite()`.

After variant bookkeeping (step 4), add registry clearing and the dependency check:
```python
    # ----------------------------------------------------------------
    # 4.1. Clear dependency registry on first variant (prevent cross-sweep leakage)
    # ----------------------------------------------------------------
    if first_variant:
        _slurm_job_registry.clear()

    # ----------------------------------------------------------------
    # 4.2. Sequential dependency check (non-SLURM guard)
    # ----------------------------------------------------------------
    has_sequential = "_chester_seq_identity" in variant

    if has_sequential and backend_config.type != "slurm" and not skip_dependency_check:
        raise ValueError(
            "[chester] sequential dependencies (sequential=True in vg.add()) "
            f"are only enforced on SLURM backends. Current mode: '{mode}'.\n"
            "Pass skip_dependency_check=True to run_experiment_lite() "
            "to suppress this check when you deliberately want unordered execution."
        )
```

Pop the metadata from the variant before it gets serialized:
```python
    seq_identity = variant.pop("_chester_seq_identity", None)
    pred_identities = variant.pop("_chester_pred_identities", None)
```

Before submission, resolve predecessor job IDs:
```python
    dependency_job_ids = None
    if backend_config.type == "slurm" and pred_identities:
        dependency_job_ids = []
        for pred_identity in pred_identities:
            jid = _slurm_job_registry.get((exp_prefix, pred_identity))
            if jid is not None:
                dependency_job_ids.append(jid)
            else:
                print(f"[chester] Warning: predecessor job not found in registry "
                      f"for identity {pred_identity}. Submitting without this dependency.")
        if not dependency_job_ids:
            dependency_job_ids = None
```

After SLURM submission, register the job ID:
```python
    # After submit returns job_id for SLURM:
    if backend_config.type == "slurm" and seq_identity is not None and submit_result is not None:
        _slurm_job_registry[(exp_prefix, seq_identity)] = submit_result
```

Pass `dependency_job_ids` to `backend.submit()` for SLURM:
```python
    submit_result = backend.submit(
        backend_task, script_content, dry=dry,
        **({"dependency_job_ids": dependency_job_ids} if dependency_job_ids else {}),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_sequential_deps.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass (no regressions)

- [ ] **Step 6: Commit**

```bash
git add src/chester/run_exp.py tests/test_sequential_deps.py
git commit -m "feat: wire sequential SLURM dependencies into run_experiment_lite"
```

---

## Chunk 3: Integration test and cleanup

### Task 8: End-to-end integration test

**Files:**
- Modify: `tests/test_sequential_deps.py`

- [ ] **Step 1: Write integration test**

```python
class TestSequentialIntegration:
    """End-to-end test simulating a launcher loop with sequential deps."""

    def test_slurm_dependency_chain(self, monkeypatch, tmp_path):
        """Simulate launching sequential variants on SLURM and verify
        dependency flags are passed correctly."""
        import chester.run_exp as run_exp

        # Reset module-level state
        run_exp.exp_count = -2
        run_exp.remote_confirmed = False
        run_exp._slurm_job_registry.clear()

        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1])
        variants = vg.variants()

        submitted_jobs = []

        # Mock the backend submit to capture dependency_job_ids
        original_submit = None

        def mock_submit(self, task, script_content, dry=False, dependency_job_ids=None):
            job_id = len(submitted_jobs) + 100
            submitted_jobs.append({
                "exp_name": task.get("exp_name", ""),
                "dependency_job_ids": dependency_job_ids,
                "job_id": job_id,
            })
            return job_id

        # Patch SlurmBackend.submit
        from chester.backends.slurm import SlurmBackend
        monkeypatch.setattr(SlurmBackend, "submit", mock_submit)

        # Patch rsync and other remote operations
        monkeypatch.setattr(run_exp, "rsync_code_v2", lambda **kw: None)

        # Need a chester config that returns a slurm backend
        # Use test config if available, or mock load_config
        # This will depend on existing test infrastructure
        # ... (adapt to existing test patterns)

        # Launch all variants
        for v in variants:
            run_experiment_lite(
                stub_method_call=lambda v, log_dir, exp_name: None,
                variant=v,
                mode="gl",  # or whatever SLURM mode exists in test config
                exp_prefix="seq_test",
                confirm=True,
                dry=False,
            )

        # First job (training) should have no dependencies
        assert submitted_jobs[0]["dependency_job_ids"] is None

        # Second job (evaluate) should depend on first job
        assert submitted_jobs[1]["dependency_job_ids"] == [100]
```

Note: This test needs adaptation to work with the actual config/backend mocking pattern used in the existing test suite. Check `tests/test_run_exp_v2.py` and `tests/test_integration.py` for patterns. The key assertion is that the second sbatch call includes `dependency_job_ids=[first_job_id]`.

- [ ] **Step 2: Run integration test**

Run: `uv run pytest tests/test_sequential_deps.py::TestSequentialIntegration -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_sequential_deps.py
git commit -m "test: add end-to-end integration test for sequential SLURM deps"
```

### Task 9: Final cleanup and documentation

**Files:**
- Modify: `CLAUDE.md` (if sequential_steps is mentioned)
- Modify: `CHANGELOG.md` (if it exists)

- [ ] **Step 1: Check for any remaining references to `sequential_steps`**

Run: `grep -r "sequential_steps" src/ tests/`
Expected: No results

- [ ] **Step 2: Check for any remaining references in docs**

Run: `grep -r "sequential_steps" .`
Expected: No results (or only in git history)

- [ ] **Step 3: Run full test suite one final time**

Run: `uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit any remaining cleanup**

```bash
git add -A
git commit -m "chore: final cleanup of sequential_steps references"
```
