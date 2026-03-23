# Submodule Commit Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `submodule_commits` parameter to `run_experiment_lite()` that pins submodule git commits by creating temporary worktrees on the remote host and redirecting singularity mounts to them.

**Architecture:** Pure Python at generation time — validate refs locally, compute worktree paths, inject bash commands into the SLURM/SSH script that create worktrees before and clean up after the singularity exec. Mount rewriting happens in Python before `wrap_with_singularity()` is called. No config schema changes.

**Tech Stack:** Python 3.10+, `subprocess` (for `git rev-parse`), `secrets` (random hex), existing chester backend infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-20-submodule-commit-pinning-design.md`

---

## File Map

| File | Change |
|------|--------|
| `src/chester/run_exp.py` | Add `submodule_commits` param; add `_validate_submodule_commits()`, `_build_worktree_paths()`; wire into variant loop and `_register_job_for_pull()` |
| `src/chester/backends/base.py` | Add `mounts_override` to `wrap_with_singularity()`; add `_rewrite_mounts_for_worktrees()`, `_build_worktree_setup_commands()`, `_build_worktree_cleanup_commands()` |
| `src/chester/backends/slurm.py` | Add `submodule_worktrees` + `submodule_resolved_commits` to `generate_script()` |
| `src/chester/backends/ssh.py` | Same as slurm.py |
| `tests/test_submodule_worktrees.py` | New — unit tests for all pure helpers in base.py |
| `tests/test_backend_slurm.py` | Add tests for `generate_script()` with worktrees |
| `tests/test_backend_ssh.py` | Add tests for `generate_script()` with worktrees |
| `tests/test_run_exp_v2.py` | Add tests for `_validate_submodule_commits()` and singularity guard |

---

## Task 1: Pure helper — `_rewrite_mounts_for_worktrees()`

**Files:**
- Create: `tests/test_submodule_worktrees.py`
- Modify: `src/chester/backends/base.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_submodule_worktrees.py`:

```python
import pytest
from chester.backends.base import _rewrite_mounts_for_worktrees


REMOTE_DIR = "/home/user/project"


def test_rewrites_subdirectory_mount():
    # source subpath of the submodule — should get wt_path/source
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/home/user/project/IsaacLabTactile/.worktrees/wt0/source:/workspace/IsaacLabTactile/source"]


def test_rewrites_bare_submodule_mount():
    mounts = ["third_party/rl_games:/workspace/third_party/rl_games"]
    worktrees = {"third_party/rl_games": "/home/user/project/third_party/rl_games/.worktrees/wt1"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/home/user/project/third_party/rl_games/.worktrees/wt1:/workspace/third_party/rl_games"]


def test_does_not_rewrite_non_submodule_mount():
    mounts = ["configs:/workspace/configs"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["configs:/workspace/configs"]


def test_no_prefix_collision():
    # IsaacLabTactile_v2 must NOT be rewritten when submodule is IsaacLabTactile
    mounts = ["IsaacLabTactile_v2/source:/workspace/foo"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["IsaacLabTactile_v2/source:/workspace/foo"]


def test_dollar_prefixed_mount_not_rewritten():
    mounts = ["$ISAAC_KIT_DATA:/opt/isaac-sim/kit/data"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["$ISAAC_KIT_DATA:/opt/isaac-sim/kit/data"]


def test_tilde_prefixed_mount_not_rewritten():
    mounts = ["~/.isaac-cache:/opt/isaac-sim/kit/cache"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["~/.isaac-cache:/opt/isaac-sim/kit/cache"]


def test_bare_mount_no_dst():
    mounts = ["/usr/share/glvnd"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/usr/share/glvnd"]


def test_multiple_submodules_rewritten():
    mounts = [
        "IsaacLabTactile/source:/workspace/IsaacLabTactile/source",
        "third_party/rl_games:/workspace/third_party/rl_games",
        "configs:/workspace/configs",
    ]
    worktrees = {
        "IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0",
        "third_party/rl_games": "/home/user/project/third_party/rl_games/.worktrees/wt1",
    }
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result[0] == "/home/user/project/IsaacLabTactile/.worktrees/wt0/source:/workspace/IsaacLabTactile/source"
    assert result[1] == "/home/user/project/third_party/rl_games/.worktrees/wt1:/workspace/third_party/rl_games"
    assert result[2] == "configs:/workspace/configs"


def test_does_not_mutate_input():
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    original = list(mounts)
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert mounts == original


def test_empty_worktrees_returns_mounts_unchanged():
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    result = _rewrite_mounts_for_worktrees(mounts, {}, REMOTE_DIR)
    assert result == mounts
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
uv run pytest tests/test_submodule_worktrees.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError` — `_rewrite_mounts_for_worktrees` not defined.

- [ ] **Step 3: Implement `_rewrite_mounts_for_worktrees()` in `base.py`**

Add this function as a module-level standalone function in `src/chester/backends/base.py`, after the `Backend` class definition:

```python
def _rewrite_mounts_for_worktrees(
    mounts: List[str],
    submodule_worktrees: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Rewrite mount sources that fall under a pinned submodule worktree.

    For each mount whose host-side source resolves to a path under a pinned
    submodule, replaces the submodule root prefix with the worktree path.

    The ``/`` suffix guard in the prefix check prevents false matches
    (e.g. submodule ``IsaacLabTactile`` does NOT match ``IsaacLabTactile_v2``).

    Mounts whose source starts with ``~`` or ``$`` are left untouched
    (shell-expanded at runtime, cannot be statically resolved).

    Args:
        mounts: List of mount strings in ``src:dst`` or bare ``src`` format.
        submodule_worktrees: Mapping of submodule path (relative to project
            root) to absolute remote worktree path.
        remote_dir: Absolute path of the remote project root.

    Returns:
        New list of mount strings with sources rewritten where applicable.
        The input list is not mutated.
    """
    import os as _os

    result = []
    # Pre-compute absolute submodule paths once
    abs_submodules = {
        sub: _os.path.normpath(_os.path.join(remote_dir, sub))
        for sub in submodule_worktrees
    }

    for mount in mounts:
        if ":" in mount:
            src, dst = mount.split(":", 1)
        else:
            src, dst = mount, None

        # Leave shell-expanded paths untouched
        if src.startswith(("~", "$")):
            result.append(mount)
            continue

        # Resolve relative src to absolute remote path
        if not _os.path.isabs(src):
            abs_src = _os.path.normpath(_os.path.join(remote_dir, src))
        else:
            abs_src = _os.path.normpath(src)

        new_src = abs_src
        for sub_path, wt_path in submodule_worktrees.items():
            abs_sub = abs_submodules[sub_path]
            if abs_src == abs_sub:
                new_src = wt_path
                break
            # Explicit /sep guard prevents IsaacLabTactile_v2 matching IsaacLabTactile
            if abs_src.startswith(abs_sub + _os.sep):
                suffix = abs_src[len(abs_sub):]  # includes leading sep
                new_src = wt_path + suffix
                break
        # NOTE: new_src is the absolute worktree path (not a $CHESTER_WT_N bash variable).
        # The spec describes using bash variable references in mounts, but using
        # the absolute path directly is functionally equivalent and simpler — the
        # path is fully known at script generation time. The CHESTER_WT_N variables
        # are still emitted for git worktree add and cleanup; they just aren't needed
        # in the singularity -B flags.

        if dst is not None:
            result.append(f"{new_src}:{dst}")
        else:
            result.append(new_src)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_submodule_worktrees.py -v
```

Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/houhd/code/cotrain_dynamics/third_party/chester
git add tests/test_submodule_worktrees.py src/chester/backends/base.py
git commit -m "feat(worktrees): add _rewrite_mounts_for_worktrees() helper"
```

---

## Task 2: Pure helpers — setup/cleanup command builders

**Files:**
- Modify: `tests/test_submodule_worktrees.py`
- Modify: `src/chester/backends/base.py`

- [ ] **Step 1: Add failing tests to `tests/test_submodule_worktrees.py`**

```python
from chester.backends.base import (
    _rewrite_mounts_for_worktrees,
    _build_worktree_setup_commands,
    _build_worktree_cleanup_commands,
)


def test_setup_commands_variable_assignments():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "abc" * 13 + "abcd"}  # 40 chars
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "CHESTER_WT_0=/remote/IsaacLabTactile/.worktrees/wt0" in combined


def test_setup_commands_trap():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "trap '_chester_wt_cleanup' EXIT" in combined
    assert "trap 'trap - EXIT; _chester_wt_cleanup; exit 130' INT" in combined
    assert "trap 'trap - EXIT; _chester_wt_cleanup; exit 143' TERM" in combined


def test_setup_commands_git_worktree_add():
    sha = "abc1234def5678abc1234def5678abc1234def56"
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": sha}
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "git -C /remote/IsaacLabTactile worktree add" in combined
    assert sha in combined
    # The worktree add must use the shell variable reference, not the literal path
    assert '"$CHESTER_WT_0"' in combined


def test_setup_commands_multiple_submodules():
    worktrees = {
        "IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0",
        "third_party/rl_games": "/remote/third_party/rl_games/.worktrees/wt1",
    }
    commits = {
        "IsaacLabTactile": "a" * 40,
        "third_party/rl_games": "b" * 40,
    }
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "CHESTER_WT_0=" in combined
    assert "CHESTER_WT_1=" in combined
    assert "git -C /remote/IsaacLabTactile" in combined
    assert "git -C /remote/third_party/rl_games" in combined


def test_cleanup_commands_or_true():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    cmds = _build_worktree_cleanup_commands(worktrees, "/remote")
    combined = "\n".join(cmds)
    # Must use || true so cleanup doesn't fail on non-existent worktrees
    assert "|| true" in combined


def test_cleanup_commands_git_worktree_remove():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    cmds = _build_worktree_cleanup_commands(worktrees, "/remote")
    combined = "\n".join(cmds)
    assert "git -C /remote/IsaacLabTactile worktree remove --force" in combined
    assert '"$CHESTER_WT_0"' in combined
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_submodule_worktrees.py::test_setup_commands_variable_assignments tests/test_submodule_worktrees.py::test_setup_commands_trap tests/test_submodule_worktrees.py::test_cleanup_commands_or_true -v 2>&1 | head -20
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3: Implement the two builders in `base.py`**

Add after `_rewrite_mounts_for_worktrees()`:

```python
def _build_worktree_setup_commands(
    submodule_worktrees: Dict[str, str],
    resolved_commits: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Return bash lines for worktree variable assignments, trap, and git worktree add.

    Emits CHESTER_WT_0, CHESTER_WT_1, ... in dict insertion order.
    The same ordering is relied upon by _rewrite_mounts_for_worktrees()
    and _build_worktree_cleanup_commands().

    Args:
        submodule_worktrees: {submodule_path: abs_remote_worktree_path}
        resolved_commits: {submodule_path: full_40char_sha}
        remote_dir: Absolute remote project root path.

    Returns:
        List of bash lines to inject into the host-side script.
    """
    import os as _os

    lines = ["# --- chester: submodule worktree setup ---"]

    # Variable assignments
    for i, (sub, wt_path) in enumerate(submodule_worktrees.items()):
        lines.append(f"CHESTER_WT_{i}={wt_path}")

    # Cleanup function
    lines.append("")
    lines.append("_chester_wt_cleanup() {")
    for i, (sub, wt_path) in enumerate(submodule_worktrees.items()):
        abs_sub = _os.path.normpath(_os.path.join(remote_dir, sub))
        lines.append(
            f'    git -C {abs_sub} worktree remove --force "$CHESTER_WT_{i}" 2>/dev/null || true'
        )
    lines.append("}")

    # Traps: EXIT always fires; INT/TERM suppress EXIT re-fire then call cleanup
    lines.append("trap '_chester_wt_cleanup' EXIT")
    lines.append("trap 'trap - EXIT; _chester_wt_cleanup; exit 130' INT")
    lines.append("trap 'trap - EXIT; _chester_wt_cleanup; exit 143' TERM")
    lines.append("")

    # Worktree creation
    for i, (sub, wt_path) in enumerate(submodule_worktrees.items()):
        sha = resolved_commits[sub]
        abs_sub = _os.path.normpath(_os.path.join(remote_dir, sub))
        lines.append(
            f'git -C {abs_sub} worktree add "$CHESTER_WT_{i}" {sha}'
        )

    return lines


def _build_worktree_cleanup_commands(
    submodule_worktrees: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Return the body of _chester_wt_cleanup() as bash lines (without the function wrapper).

    Useful for inspection and testing of the cleanup logic in isolation.

    Args:
        submodule_worktrees: {submodule_path: abs_remote_worktree_path}
        remote_dir: Absolute remote project root path.

    Returns:
        List of bash lines for the cleanup body.
    """
    import os as _os

    lines = []
    for i, (sub, _wt_path) in enumerate(submodule_worktrees.items()):
        abs_sub = _os.path.normpath(_os.path.join(remote_dir, sub))
        lines.append(
            f'git -C {abs_sub} worktree remove --force "$CHESTER_WT_{i}" 2>/dev/null || true'
        )
    return lines
```

- [ ] **Step 4: Run all worktree tests**

```bash
uv run pytest tests/test_submodule_worktrees.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_submodule_worktrees.py src/chester/backends/base.py
git commit -m "feat(worktrees): add worktree setup/cleanup command builders"
```

---

## Task 3: `wrap_with_singularity()` — add `mounts_override` parameter

**Files:**
- Modify: `tests/test_submodule_worktrees.py`
- Modify: `src/chester/backends/base.py`

- [ ] **Step 1: Add failing test**

Add to `tests/test_submodule_worktrees.py`:

```python
from chester.backends.base import (
    _rewrite_mounts_for_worktrees,
    _build_worktree_setup_commands,
    _build_worktree_cleanup_commands,
    Backend, BackendConfig, SingularityConfig,
)


def _make_backend_with_singularity(mounts=None):
    """Helper: returns a minimal Backend instance with singularity configured."""
    from chester.backends.slurm import SlurmBackend
    from chester.backends.base import SlurmConfig
    sing = SingularityConfig(
        image="myimage.sif",
        mounts=mounts or ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"],
        gpu=False,
        fakeroot=False,
    )
    cfg = BackendConfig(
        name="gl", type="slurm", host="gl",
        remote_dir="/remote/project", singularity=sing,
    )
    return SlurmBackend(cfg, {"project_path": "/local/project", "package_manager": "python"})


def test_wrap_with_singularity_mounts_override():
    backend = _make_backend_with_singularity(
        mounts=["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    )
    override = ["/remote/project/IsaacLabTactile/.worktrees/wt0/source:/workspace/IsaacLabTactile/source"]
    result = backend.wrap_with_singularity(["echo hello"], mounts_override=override)
    # Overridden mount must appear
    assert "/worktrees/wt0/source:/workspace/IsaacLabTactile/source" in result
    # Original mount must NOT appear
    assert "IsaacLabTactile/source:/workspace/IsaacLabTactile/source" not in result.split("-B")[1] if "-B" in result else True


def test_wrap_with_singularity_no_override_uses_config_mounts():
    backend = _make_backend_with_singularity(
        mounts=["configs:/workspace/configs"]
    )
    result = backend.wrap_with_singularity(["echo hello"])
    assert "configs" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_submodule_worktrees.py::test_wrap_with_singularity_mounts_override tests/test_submodule_worktrees.py::test_wrap_with_singularity_no_override_uses_config_mounts -v 2>&1 | head -20
```

Expected: `TypeError` — `wrap_with_singularity()` takes no `mounts_override` kwarg.

- [ ] **Step 3: Add `mounts_override` to `wrap_with_singularity()` in `base.py`**

Find the `wrap_with_singularity` method signature:
```python
def wrap_with_singularity(self, commands: List[str]) -> str:
```

Change to:
```python
def wrap_with_singularity(
    self,
    commands: List[str],
    mounts_override: Optional[List[str]] = None,
) -> str:
```

Then find the line inside `wrap_with_singularity` that reads `sing.mounts`:
```python
        for m in sing.mounts:
```

Change to:
```python
        effective_mounts = mounts_override if mounts_override is not None else sing.mounts
        for m in effective_mounts:
```

- [ ] **Step 4: Run all worktree tests**

```bash
uv run pytest tests/test_submodule_worktrees.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_submodule_worktrees.py src/chester/backends/base.py
git commit -m "feat(worktrees): add mounts_override param to wrap_with_singularity()"
```

---

## Task 4: `SlurmBackend.generate_script()` — worktree injection

**Files:**
- Modify: `tests/test_backend_slurm.py`
- Modify: `src/chester/backends/slurm.py`

- [ ] **Step 1: Add failing tests to `tests/test_backend_slurm.py`**

```python
from chester.backends.base import SingularityConfig


def _make_singularity_backend(mounts=None):
    sing = SingularityConfig(
        image="myimage.sif",
        mounts=mounts or [
            "IsaacLabTactile/source:/workspace/IsaacLabTactile/source",
            "IsaacLabTactile/apps:/workspace/IsaacLabTactile/apps",
            "configs:/workspace/configs",
        ],
        gpu=True,
        fakeroot=False,
    )
    return _make_backend(singularity=sing)


def test_slurm_generate_script_with_worktrees_injects_setup():
    backend = _make_singularity_backend()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    worktrees = {"IsaacLabTactile": "/remote/project/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    script = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=worktrees,
        submodule_resolved_commits=commits,
    )
    assert "CHESTER_WT_0=" in script
    assert "_chester_wt_cleanup" in script
    assert "trap '_chester_wt_cleanup' EXIT" in script
    assert "git -C" in script and "worktree add" in script


def test_slurm_generate_script_with_worktrees_rewrites_mounts():
    backend = _make_singularity_backend()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    script = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=worktrees,
        submodule_resolved_commits=commits,
    )
    # Rewritten mount must appear (worktree path)
    assert "/worktrees/wt0/source" in script
    # configs (non-submodule) mount must still appear unchanged
    assert "configs:/workspace/configs" in script


def test_slurm_generate_script_without_worktrees_unchanged():
    backend = _make_singularity_backend()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    script_with = backend.generate_script(task, script="train.py")
    script_without = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=None,
        submodule_resolved_commits=None,
    )
    assert script_with == script_without
    assert "CHESTER_WT" not in script_with


def test_slurm_generate_script_worktrees_setup_before_singularity():
    backend = _make_singularity_backend()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    worktrees = {"IsaacLabTactile": "/remote/project/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    script = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=worktrees,
        submodule_resolved_commits=commits,
    )
    wt_pos = script.index("CHESTER_WT_0=")
    sing_pos = script.index("singularity exec")
    assert wt_pos < sing_pos
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_backend_slurm.py::test_slurm_generate_script_with_worktrees_injects_setup tests/test_backend_slurm.py::test_slurm_generate_script_with_worktrees_rewrites_mounts -v 2>&1 | head -20
```

Expected: `TypeError` — `generate_script()` got unexpected keyword argument.

- [ ] **Step 3: Extend `SlurmBackend.generate_script()` in `slurm.py`**

Add the two new parameters to the signature:
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
    serial_steps: Optional[List[Tuple[str, list]]] = None,
    submodule_worktrees: Optional[Dict[str, str]] = None,
    submodule_resolved_commits: Optional[Dict[str, str]] = None,
) -> str:
```

In the imports at the top of the method body (or file), ensure `_build_worktree_setup_commands` and `_rewrite_mounts_for_worktrees` are imported from `.base`.

After the prepare commands section (after `lines.extend(prepare_cmds)`), add:

```python
        # ---- Submodule worktree setup (before overlay and singularity) ----
        if submodule_worktrees:
            from .base import _build_worktree_setup_commands, _rewrite_mounts_for_worktrees
            lines.extend(_build_worktree_setup_commands(
                submodule_worktrees, submodule_resolved_commits, remote_dir
            ))
            rewritten_mounts = _rewrite_mounts_for_worktrees(
                self.config.singularity.mounts if self.config.singularity else [],
                submodule_worktrees,
                remote_dir,
            )
        else:
            rewritten_mounts = None
```

Then in each `wrap_with_singularity()` call within `generate_script()`, pass `mounts_override=rewritten_mounts`:

```python
# Change every occurrence of:
lines.append(self.wrap_with_singularity(inner))
# To:
lines.append(self.wrap_with_singularity(inner, mounts_override=rewritten_mounts))
```

There are two such calls (serial_steps branch and non-serial branch).

- [ ] **Step 4: Run slurm backend tests**

```bash
uv run pytest tests/test_backend_slurm.py -v
```

Expected: all tests PASS (new and existing).

- [ ] **Step 5: Commit**

```bash
git add tests/test_backend_slurm.py src/chester/backends/slurm.py
git commit -m "feat(worktrees): extend SlurmBackend.generate_script() with worktree injection"
```

---

## Task 5: `SSHBackend.generate_script()` — worktree injection

**Files:**
- Modify: `tests/test_backend_ssh.py`
- Modify: `src/chester/backends/ssh.py`

- [ ] **Step 1: Add failing tests to `tests/test_backend_ssh.py`**

Read the existing `tests/test_backend_ssh.py` first to understand the `_make_backend()` helper there, then add:

```python
from chester.backends.base import SingularityConfig


def _make_ssh_backend_with_singularity(remote_dir="/remote/project"):
    """Make an SSH backend with singularity configured."""
    from chester.backends.ssh import SSHBackend
    from chester.backends.base import BackendConfig, SingularityConfig
    sing = SingularityConfig(
        image="myimage.sif",
        mounts=[
            "IsaacLabTactile/source:/workspace/IsaacLabTactile/source",
            "configs:/workspace/configs",
        ],
        gpu=True,
        fakeroot=False,
    )
    cfg = BackendConfig(
        name="armdual", type="ssh", host="armdual",
        remote_dir=remote_dir, singularity=sing,
    )
    return SSHBackend(cfg, {"project_path": "/local/project", "package_manager": "python"})


def test_ssh_generate_script_with_worktrees_injects_setup():
    backend = _make_ssh_backend_with_singularity()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    worktrees = {"IsaacLabTactile": "/remote/project/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    script = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=worktrees,
        submodule_resolved_commits=commits,
    )
    assert "CHESTER_WT_0=" in script
    assert "_chester_wt_cleanup" in script
    assert "trap '_chester_wt_cleanup' EXIT" in script


def test_ssh_generate_script_with_worktrees_rewrites_mounts():
    backend = _make_ssh_backend_with_singularity()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    worktrees = {"IsaacLabTactile": "/remote/project/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    script = backend.generate_script(
        task, script="train.py",
        submodule_worktrees=worktrees,
        submodule_resolved_commits=commits,
    )
    assert "/worktrees/wt0/source" in script
    assert "configs:/workspace/configs" in script


def test_ssh_generate_script_without_worktrees_unchanged():
    backend = _make_ssh_backend_with_singularity()
    task = {"params": {"log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "CHESTER_WT" not in script
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_backend_ssh.py::test_ssh_generate_script_with_worktrees_injects_setup -v 2>&1 | head -20
```

Expected: `TypeError` — unexpected keyword argument.

- [ ] **Step 3: Extend `SSHBackend.generate_script()` in `ssh.py`**

Same pattern as Task 4. Add to the signature:
```python
    submodule_worktrees: Optional[Dict[str, str]] = None,
    submodule_resolved_commits: Optional[Dict[str, str]] = None,
```

After `lines.extend(prepare_cmds)`, add the worktree setup block (identical to the SLURM backend):
```python
        # ---- Submodule worktree setup (before overlay and singularity) ----
        if submodule_worktrees:
            from .base import _build_worktree_setup_commands, _rewrite_mounts_for_worktrees
            lines.extend(_build_worktree_setup_commands(
                submodule_worktrees, submodule_resolved_commits, remote_dir
            ))
            rewritten_mounts = _rewrite_mounts_for_worktrees(
                self.config.singularity.mounts if self.config.singularity else [],
                submodule_worktrees,
                remote_dir,
            )
        else:
            rewritten_mounts = None
```

Pass `mounts_override=rewritten_mounts` to both `wrap_with_singularity()` calls.

- [ ] **Step 4: Run SSH backend tests**

```bash
uv run pytest tests/test_backend_ssh.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_backend_ssh.py src/chester/backends/ssh.py
git commit -m "feat(worktrees): extend SSHBackend.generate_script() with worktree injection"
```

---

## Task 6: Validation helpers — `_validate_submodule_commits()` and `_build_worktree_paths()`

**Files:**
- Modify: `tests/test_run_exp_v2.py`
- Modify: `src/chester/run_exp.py`

- [ ] **Step 1: Add failing tests to `tests/test_run_exp_v2.py`**

```python
import os
import subprocess
from unittest.mock import patch, MagicMock
import pytest


def test_validate_submodule_commits_resolves_sha(tmp_path):
    from chester.run_exp import _validate_submodule_commits

    # Create a bare fake git repo
    sub_path = tmp_path / "MySub"
    sub_path.mkdir()
    subprocess.run(["git", "init"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub_path, check=True)
    (sub_path / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=sub_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sub_path, check=True, capture_output=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sub_path, text=True
    ).strip()

    result = _validate_submodule_commits({"MySub": sha[:8]}, str(tmp_path))
    assert result["MySub"] == sha  # resolved to full 40-char SHA


def test_validate_submodule_commits_raises_on_bad_ref(tmp_path):
    from chester.run_exp import _validate_submodule_commits

    sub_path = tmp_path / "MySub"
    sub_path.mkdir()
    subprocess.run(["git", "init"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub_path, check=True)
    (sub_path / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=sub_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sub_path, check=True, capture_output=True)

    with pytest.raises(ValueError, match="MySub"):
        _validate_submodule_commits({"MySub": "deadbeef0000"}, str(tmp_path))


def test_validate_submodule_commits_raises_on_missing_path(tmp_path):
    from chester.run_exp import _validate_submodule_commits
    with pytest.raises(ValueError, match="not found"):
        _validate_submodule_commits({"nonexistent": "abc"}, str(tmp_path))


def test_build_worktree_paths_format():
    from chester.run_exp import _build_worktree_paths
    commits = {"IsaacLabTactile": "abc1234def5678" + "a" * 26}
    paths = _build_worktree_paths(commits, "/remote/project", "03_23_10_00")
    wt = paths["IsaacLabTactile"]
    assert wt.startswith("/remote/project/IsaacLabTactile/.worktrees/")
    assert "03_23_10_00" in wt
    assert "abc1234def5" in wt  # short SHA present


def test_build_worktree_paths_unique():
    from chester.run_exp import _build_worktree_paths
    commits = {"IsaacLabTactile": "a" * 40}
    p1 = _build_worktree_paths(commits, "/remote", "03_23_10_00")
    p2 = _build_worktree_paths(commits, "/remote", "03_23_10_00")
    # Random suffix makes them unique even with same timestamp
    assert p1["IsaacLabTactile"] != p2["IsaacLabTactile"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_run_exp_v2.py::test_validate_submodule_commits_resolves_sha tests/test_run_exp_v2.py::test_build_worktree_paths_format -v 2>&1 | head -20
```

Expected: `ImportError` — functions not defined.

- [ ] **Step 3: Implement `_validate_submodule_commits()` and `_build_worktree_paths()` in `run_exp.py`**

Add near the top of `run_exp.py` (after existing imports, before class definitions):

```python
import secrets as _secrets


def _validate_submodule_commits(
    submodule_commits: dict,
    project_path: str,
) -> dict:
    """Validate submodule refs locally and resolve to full 40-char SHAs.

    Args:
        submodule_commits: {submodule_path: ref} — user-provided refs (may be
            short SHA, branch name, or tag).
        project_path: Absolute local path of the project root.

    Returns:
        {submodule_path: full_sha} with resolved 40-char SHAs.

    Raises:
        ValueError: If a submodule path does not exist or a ref cannot be
            resolved.
    """
    resolved = {}
    for sub_path, ref in submodule_commits.items():
        abs_sub = os.path.join(project_path, sub_path)
        if not os.path.isdir(abs_sub):
            raise ValueError(
                f"[chester] submodule_commits: path not found: '{abs_sub}'\n"
                f"  Key '{sub_path}' must be a directory under project_path."
            )
        try:
            full_sha = subprocess.check_output(
                ["git", "-C", abs_sub, "rev-parse", "--verify", f"{ref}^{{commit}}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            raise ValueError(
                f"[chester] submodule_commits: cannot resolve ref '{ref}' "
                f"in submodule '{sub_path}'.\n"
                f"  Run: git -C {abs_sub} rev-parse {ref}"
            )
        resolved[sub_path] = full_sha
    return resolved


def _build_worktree_paths(
    resolved_commits: dict,
    remote_dir: str,
    timestamp: str,
) -> dict:
    """Compute unique remote worktree paths for each pinned submodule.

    Names follow the pattern:
        {submodule_path}/.worktrees/{timestamp}_{random6hex}_{short_sha}/

    The timestamp reflects submission time (not execution time). The random
    6-hex suffix ensures uniqueness across concurrent same-minute launches.

    Args:
        resolved_commits: {submodule_path: full_40char_sha}
        remote_dir: Absolute remote project root path.
        timestamp: Submission timestamp string (e.g. "03_23_10_00").

    Returns:
        {submodule_path: abs_remote_worktree_path}
    """
    result = {}
    for sub_path, full_sha in resolved_commits.items():
        rand = _secrets.token_hex(3)       # 6 hex chars
        short_sha = full_sha[:12]
        wt_name = f"{timestamp}_{rand}_{short_sha}"
        wt_path = os.path.join(remote_dir, sub_path, ".worktrees", wt_name)
        result[sub_path] = wt_path
    return result
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_run_exp_v2.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_run_exp_v2.py src/chester/run_exp.py
git commit -m "feat(worktrees): add _validate_submodule_commits() and _build_worktree_paths()"
```

---

## Task 7: Wire into `run_experiment_lite()` and `_register_job_for_pull()`

**Files:**
- Modify: `tests/test_run_exp_v2.py`
- Modify: `src/chester/run_exp.py`

- [ ] **Step 1: Add failing tests**

```python
def test_run_experiment_lite_submodule_commits_requires_singularity(tmp_path, monkeypatch):
    """submodule_commits with no singularity raises ValueError before submission."""
    from chester.run_exp import run_experiment_lite
    from chester.backends.base import BackendConfig

    # Patch load_config to return a minimal local backend with no singularity
    fake_cfg = {
        "project_path": str(tmp_path),
        "log_dir": str(tmp_path / "data"),
        "package_manager": "python",
        "backends": {
            "local": {"type": "local"},
        },
    }
    fake_backend_cfg = BackendConfig(name="local", type="local", singularity=None)

    monkeypatch.setattr("chester.run_exp.load_config", lambda: fake_cfg)
    monkeypatch.setattr("chester.run_exp.get_backend", lambda mode, cfg: fake_backend_cfg)

    with pytest.raises(ValueError, match="singularity"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="local",
            exp_prefix="test",
            submodule_commits={"MySub": "abc1234"},
        )


def test_register_job_for_pull_stores_submodule_commits(tmp_path, monkeypatch):
    """_register_job_for_pull writes submodule_commits and submodule_worktrees to job file."""
    from chester.run_exp import _register_job_for_pull
    import json

    monkeypatch.setattr(
        "chester.run_exp.get_default_job_store_dir",
        lambda: str(tmp_path / "jobs"),
    )
    monkeypatch.setattr(
        "chester.run_exp.write_job_file",
        lambda job_store_dir, job: "job_001",
    )

    captured = {}

    def fake_write(job_store_dir, job):
        captured.update(job)
        return "job_001"

    monkeypatch.setattr("chester.run_exp.write_job_file", fake_write)

    _register_job_for_pull(
        host="gl",
        remote_log_dir="/remote/logs/exp1",
        local_log_dir="/local/logs/exp1",
        exp_name="exp1",
        exp_prefix="myexp",
        submodule_commits={"IsaacLabTactile": "a" * 40},
        submodule_worktrees={"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"},
    )

    assert captured["submodule_commits"] == {"IsaacLabTactile": "a" * 40}
    assert captured["submodule_worktrees"] == {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_run_exp_v2.py::test_run_experiment_lite_submodule_commits_requires_singularity tests/test_run_exp_v2.py::test_register_job_for_pull_stores_submodule_commits -v 2>&1 | head -30
```

Expected: `TypeError` (no `submodule_commits` param) or assertion failures.

- [ ] **Step 3: Add `submodule_commits` param and validation to `run_experiment_lite()`**

In the function signature, add:
```python
submodule_commits=None,
```

After step 3 (singularity override, around line 1077), add a new section:

```python
    # ----------------------------------------------------------------
    # 3.5. Validate submodule commit pinning
    # ----------------------------------------------------------------
    resolved_commits = {}
    submodule_worktrees = {}
    if submodule_commits:
        if backend.config.singularity is None:
            raise ValueError(
                f"[chester] submodule_commits requires singularity to be active for "
                f"backend '{mode}'. The current backend has no singularity config, or "
                f"singularity was disabled via use_singularity=False."
            )
        resolved_commits = _validate_submodule_commits(submodule_commits, project_path)
        remote_dir = backend.config.remote_dir or project_path
        submodule_worktrees = _build_worktree_paths(resolved_commits, remote_dir, timestamp)
        print(f"[chester] Submodule commit pinning:")
        for sub, sha in resolved_commits.items():
            wt = submodule_worktrees[sub]
            wt_rel = os.path.relpath(wt, remote_dir)
            print(f"  {sub}: {submodule_commits[sub]} -> {sha}")
            print(f"      worktree: {wt_rel}")
```

Then in the remote backend dispatch section (around line 1336–1344), add the worktree kwargs to `gen_kwargs`:

```python
        # Pass worktree info to backend for script generation (singularity only)
        if submodule_worktrees and backend.config.singularity:
            gen_kwargs["submodule_worktrees"] = submodule_worktrees
            gen_kwargs["submodule_resolved_commits"] = resolved_commits
```

- [ ] **Step 4: Extend `_register_job_for_pull()` signature**

Find the function definition and add two new optional params:

```python
def _register_job_for_pull(
    host: str,
    remote_log_dir: str,
    local_log_dir: str,
    exp_name: str,
    exp_prefix: str,
    extra_pull_dirs: list = None,
    slurm_job_id: int = None,
    submodule_commits: dict = None,
    submodule_worktrees: dict = None,
):
```

Inside the function, after building the `job` dict, add:

```python
    if submodule_commits:
        job['submodule_commits'] = submodule_commits
    if submodule_worktrees:
        job['submodule_worktrees'] = submodule_worktrees
```

Update the call site in `run_experiment_lite()` (around line 1397):

```python
                _register_job_for_pull(
                    host=backend_config.host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task.get('exp_name', ''),
                    exp_prefix=exp_prefix,
                    extra_pull_dirs=resolved_extra_pull_dirs,
                    slurm_job_id=slurm_job_id,
                    submodule_commits=resolved_commits or None,
                    submodule_worktrees=submodule_worktrees or None,
                )
```

- [ ] **Step 5: Run all tests**

```bash
uv run pytest tests/test_run_exp_v2.py tests/test_backend_slurm.py tests/test_backend_ssh.py tests/test_submodule_worktrees.py -v
```

Expected: all PASS.

- [ ] **Step 6: Run full test suite to check for regressions**

```bash
uv run pytest tests/ -v --ignore=tests/live_slurm_validation.py --ignore=tests/launch_mnist.py --ignore=tests/launch_test_extra_pull.py -x
```

Expected: all PASS, no regressions.

- [ ] **Step 7: Commit**

```bash
git add src/chester/run_exp.py tests/test_run_exp_v2.py
git commit -m "feat(worktrees): wire submodule_commits into run_experiment_lite() and job store"
```

---

## Completion Check

After all tasks are done, verify end-to-end with a dry run:

```bash
# From cotrain_dynamics/ project root — dry run prints the SLURM script
# You can inspect it for CHESTER_WT variables, trap, rewritten mounts, etc.
# (requires chester.yaml or .chester/config.yaml to be configured for a SLURM backend)
uv run python -c "
from chester.run_exp import run_experiment_lite, VariantGenerator
vg = VariantGenerator()
vg.add('task', ['training'])
for v in vg.variants():
    run_experiment_lite(
        stub_method_call=lambda variant, log_dir, exp_name: None,
        variant=v,
        mode='gl',
        exp_prefix='test_worktree',
        dry=True,
        submodule_commits={'third_party/rl_games_cotrain': 'HEAD'},
    )
"
```

Inspect the output for:
- `CHESTER_WT_0=...` variable assignment
- `_chester_wt_cleanup()` function with `|| true`
- `trap '_chester_wt_cleanup' EXIT`
- Rewritten `-B` mount pointing into `.worktrees/`
- Original non-submodule mounts (`configs`, `.chester`, etc.) unchanged
