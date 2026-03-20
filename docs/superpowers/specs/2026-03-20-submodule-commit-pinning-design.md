# Submodule Commit Pinning for Singularity Backends

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Chester launcher (`chester-ml`) — SLURM and SSH backends with singularity enabled

---

## Problem

When running experiments on remote clusters via singularity, project submodules
(`IsaacLabTactile`, `third_party/rl_games_cotrain`, etc.) are bind-mounted into
the container from the current working tree. There is no way to pin a specific
commit of a submodule for a given experiment without manually checking out that
commit — which would clobber the working tree and break other concurrent jobs.

This feature adds a `submodule_commits` parameter to `run_experiment_lite()` that
lets users specify an exact commit for one or more submodules. Chester creates a
git worktree at that commit on the remote host, redirects the singularity mounts
to that worktree, and removes the worktree when the job finishes.

---

## Non-Goals

- Local runs (no singularity): not supported. Submodule commit pinning only applies
  when singularity is active for the backend.
- Per-variant commits: `submodule_commits` is per-launch, applied identically to
  all variants submitted in one `run_experiment_lite()` call.
- Config-file-level submodule pinning: no changes to `.chester/config.yaml` schema.

---

## API

```python
run_experiment_lite(
    ...,
    submodule_commits={
        'IsaacLabTactile': 'abc1234',               # short SHA, branch, or tag
        'third_party/rl_games_cotrain': 'def5678',
    },
)
```

Keys are submodule paths relative to `project_path`, matching entries in
`.gitmodules`. Values are any git-resolvable ref. If `submodule_commits` is
`None` (default), behaviour is unchanged.

---

## Local Validation (before any script is generated)

Validation runs once per launch, **after the backend config is resolved but
before the variant loop begins**. "Singularity active" means
`backend_config.singularity is not None` (i.e. the backend has a singularity
config block regardless of the `enabled` flag — `use_singularity` overrides
are applied before this check, so by the time validation runs the backend's
singularity field reflects the final active state).

1. **Singularity guard**: if `submodule_commits` is non-empty and
   `backend.config.singularity is None`, raise `ValueError` with a clear message
   explaining that submodule commit pinning requires singularity to be active.

2. **For each `(submodule_path, ref)` pair**:
   - Verify `{project_path}/{submodule_path}` exists and is a git repository.
   - Run `git -C {project_path}/{submodule_path} rev-parse --verify {ref}^{{commit}}`
     to resolve the ref to a full 40-character SHA.
   - On failure, raise `ValueError` naming the submodule and ref.

3. Store resolved full SHAs internally; use them (not the user-provided refs) in
   all subsequent steps.

**Validation helper:**
```python
def _validate_submodule_commits(
    submodule_commits: Dict[str, str],
    project_path: str,
) -> Dict[str, str]:
    """Validate submodule refs and resolve to full SHAs.

    Returns {submodule_path: full_40char_sha}.
    Raises ValueError on any bad input.
    """
```

**Call site in `run_experiment_lite()`** (pseudocode):
```python
# After use_singularity override is applied to backend.config:
if submodule_commits:
    if backend.config.singularity is None:
        raise ValueError("submodule_commits requires singularity ...")
    resolved_commits = _validate_submodule_commits(submodule_commits, project_path)
    submodule_worktrees = _build_worktree_paths(resolved_commits, remote_dir, timestamp)
else:
    resolved_commits = {}
    submodule_worktrees = {}
```

---

## Worktree Naming

One worktree path is generated per submodule per launch (shared across all
variants in the launch):

```
{submodule_path}/.worktrees/{mm_dd_HH_MM}_{random6hex}_{short_sha}/
```

Example:
```
IsaacLabTactile/.worktrees/03_20_14_32_a3f9c1_abc1234def5/
```

- **Timestamp** (`mm_dd_HH_MM`): reflects submission time (not execution time,
  since SLURM jobs may queue for hours). Uses chester's existing `timestamp`
  global. This is intentional — it identifies when the launch was initiated.
- **Random 6-hex suffix**: ensures uniqueness across concurrent jobs launched
  in the same minute. Generated via `secrets.token_hex(3)`.
- **Short SHA** (first 12 chars of full SHA): human-identifiable in filesystem
  listings.

**Helper** (pure function called at launch time, before the variant loop):
```python
def _build_worktree_paths(
    resolved_commits: Dict[str, str],   # {submodule_path: full_sha}
    remote_dir: str,
    timestamp: str,                     # e.g. "03_20_14_32"
) -> Dict[str, str]:
    """Return {submodule_path: absolute_remote_worktree_path}."""
```

The full remote path is: `{remote_dir}/{submodule_path}/.worktrees/{name}`.

---

## Generated Script Structure

The following is injected into the host-side bash script (outside singularity),
**after module loads and prepare commands, before the singularity call**.

### Variable assignments and cleanup trap

```bash
# --- chester: submodule worktree setup ---
CHESTER_WT_0=/home/houhd/code/cotrain_dynamics/IsaacLabTactile/.worktrees/03_20_14_32_a3f9c1_abc1234def5
CHESTER_WT_1=/home/houhd/code/cotrain_dynamics/third_party/rl_games_cotrain/.worktrees/03_20_14_32_a3f9c1_def5678abc1

_chester_wt_cleanup() {
    git -C /home/houhd/code/cotrain_dynamics/IsaacLabTactile \
        worktree remove --force "$CHESTER_WT_0" 2>/dev/null || true
    git -C /home/houhd/code/cotrain_dynamics/third_party/rl_games_cotrain \
        worktree remove --force "$CHESTER_WT_1" 2>/dev/null || true
}
trap '_chester_wt_cleanup' EXIT
trap 'trap - EXIT; _chester_wt_cleanup; exit 130' INT
trap 'trap - EXIT; _chester_wt_cleanup; exit 143' TERM
```

The `|| true` on each `worktree remove` is **required**: if `git worktree add`
for `$CHESTER_WT_1` fails (e.g. bad commit), `set -e` fires the EXIT trap.
The cleanup function must not itself fail when cleaning up a worktree that was
never created. Without `|| true`, the cleanup would exit non-zero and prevent
cleanup of any already-created worktrees earlier in the list.

### Worktree creation

```bash
git -C /home/houhd/code/cotrain_dynamics/IsaacLabTactile \
    worktree add "$CHESTER_WT_0" abc1234def5678abc1234def5678abc1234def5678
git -C /home/houhd/code/cotrain_dynamics/third_party/rl_games_cotrain \
    worktree add "$CHESTER_WT_1" def5678abc1234def5678abc1234def5678abc1234
```

### Singularity call (mounts rewritten)

Mounts are rewritten in Python before being passed to `wrap_with_singularity()`.
The rewritten mount list is passed as a `mounts_override` parameter (see below).

Before rewriting:
```
IsaacLabTactile/source:/workspace/IsaacLabTactile/source
IsaacLabTactile/apps:/workspace/IsaacLabTactile/apps
third_party/rl_games_cotrain:/workspace/third_party/rl_games_cotrain
```

After rewriting:
```
$CHESTER_WT_0/source:/workspace/IsaacLabTactile/source
$CHESTER_WT_0/apps:/workspace/IsaacLabTactile/apps
$CHESTER_WT_1:/workspace/third_party/rl_games_cotrain
```

The bash variable references (`$CHESTER_WT_N`) are emitted unquoted so the
shell expands them at runtime. The existing `$`-prefix guard in
`wrap_with_singularity()` (the `not src.startswith(("~", "$"))` check) already
skips resolution of these paths, so they pass through correctly.

### Serial steps

For jobs with `order='serial'`, worktree setup/teardown wraps all serial steps.
The same worktree is reused across all steps within the script.

---

## Mount Rewriting Logic

### `wrap_with_singularity()` extension

Add a `mounts_override` parameter:

```python
def wrap_with_singularity(
    self,
    commands: List[str],
    mounts_override: Optional[List[str]] = None,
) -> str:
```

When `mounts_override` is not `None`, it is used in place of `sing.mounts` for
building the `-B` flags. All other singularity flags (gpu, fakeroot, overlay,
workdir, image) are unchanged. This keeps the original config immutable.

### `_rewrite_mounts_for_worktrees()` helper

Implemented as a standalone function (not a method) in `base.py`:

```python
def _rewrite_mounts_for_worktrees(
    mounts: List[str],
    submodule_worktrees: Dict[str, str],   # {submodule_path: abs_worktree_path}
    remote_dir: str,
) -> List[str]:
    """Rewrite mount sources that fall under a pinned submodule to point
    at the corresponding worktree.

    Returns a new mount list; the input is not mutated.
    """
```

**Algorithm** for each mount string `src:dst` (or bare `src`):

1. Parse `src` and `dst` by splitting on the first `:`.
2. Skip rewriting if `src` starts with `~` or `$` (already absolute/expanded).
3. Resolve `src` to an absolute remote path by prepending `remote_dir` if not
   already absolute.
4. For each `(submodule_path, wt_path)` in `submodule_worktrees`:
   - Compute `abs_submodule = os.path.normpath(os.path.join(remote_dir, submodule_path))`
   - If `resolved_src == abs_submodule`: replace `src` with `$CHESTER_WT_N`
   - If `resolved_src` starts with `abs_submodule + "/"`: replace the
     `abs_submodule` prefix with `$CHESTER_WT_N`
     (the explicit `/` suffix guard prevents matching `IsaacLabTactile_v2` when
     the submodule is `IsaacLabTactile` — this is intentional and must not be
     simplified to a bare `startswith(abs_submodule)` check)
   - Otherwise: leave `src` unchanged
5. Reconstruct the mount string as `new_src:dst` (or bare `new_src`).

`$CHESTER_WT_N` is the bash variable reference string (e.g. `"$CHESTER_WT_0"`),
derived from the enumerated index of the submodule in `submodule_worktrees`.
The index ordering must be consistent with the variable assignments emitted in
the setup commands.

---

## Worktree Setup/Cleanup Command Builders

Both are standalone functions in `base.py`:

```python
def _build_worktree_setup_commands(
    submodule_worktrees: Dict[str, str],   # {submodule_path: abs_worktree_path}
    resolved_commits: Dict[str, str],      # {submodule_path: full_sha}
    remote_dir: str,
) -> List[str]:
    """Return bash lines for variable assignments, trap, and git worktree add."""

def _build_worktree_cleanup_commands(
    submodule_worktrees: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Return the body of _chester_wt_cleanup() as bash lines."""
```

`submodule_worktrees` dict ordering must be stable (Python 3.7+ dict insertion
order) so that `$CHESTER_WT_0`, `$CHESTER_WT_1`, ... match across the variable
assignment, cleanup, and mount-rewriting steps.

---

## `generate_script()` Signature Extension

Both `SlurmBackend.generate_script()` and `SSHBackend.generate_script()` gain
two new keyword parameters:

```python
def generate_script(
    self,
    task: Dict[str, Any],
    script: str,
    ...,                                    # existing params unchanged
    submodule_worktrees: Optional[Dict[str, str]] = None,
    # {submodule_path: abs_remote_worktree_path}; None means feature inactive
    submodule_resolved_commits: Optional[Dict[str, str]] = None,
    # {submodule_path: full_40char_sha}; required when submodule_worktrees is set
) -> str:
```

**Updated call site in `run_exp.py`** (within the backend dispatch):
```python
gen_kwargs["submodule_worktrees"] = submodule_worktrees
gen_kwargs["submodule_resolved_commits"] = resolved_commits
```

**Script injection in `generate_script()`** — insert after prepare commands and
before the singularity call. The relative ordering with respect to overlay setup
does not matter (worktree creation has no dependency on the overlay image), so
placing it immediately after prepare commands and before overlay setup is fine:

```python
if submodule_worktrees:
    lines.extend(_build_worktree_setup_commands(
        submodule_worktrees, submodule_resolved_commits, remote_dir
    ))
    rewritten_mounts = _rewrite_mounts_for_worktrees(
        sing.mounts, submodule_worktrees, remote_dir
    )
else:
    rewritten_mounts = None

# overlay setup (unchanged position)
lines.extend(self.get_overlay_setup_commands())

# ... then when calling wrap_with_singularity():
lines.append(self.wrap_with_singularity(inner, mounts_override=rewritten_mounts))
```

---

## `_register_job_for_pull()` Signature Extension

Add two explicit keyword parameters:

```python
def _register_job_for_pull(
    host: str,
    remote_log_dir: str,
    local_log_dir: str,
    exp_name: str,
    exp_prefix: str,
    extra_pull_dirs: list = None,
    slurm_job_id: int = None,
    submodule_commits: Optional[Dict[str, str]] = None,    # NEW: full SHAs
    submodule_worktrees: Optional[Dict[str, str]] = None,  # NEW: abs remote paths
):
```

These are written into the job dict only when non-None:
```python
if submodule_commits:
    job['submodule_commits'] = submodule_commits
if submodule_worktrees:
    job['submodule_worktrees'] = submodule_worktrees
```

---

## Job Metadata

Fields added to the job dict in the chester job store:

```python
{
    # existing fields ...
    'submodule_commits': {
        'IsaacLabTactile': 'abc1234def5678abc1234def5678abc1234def5678',
        'third_party/rl_games_cotrain': 'def5678abc1234def5678abc1234def5678abc1234',
    },
    'submodule_worktrees': {
        'IsaacLabTactile': '/home/.../IsaacLabTactile/.worktrees/03_20_14_32_a3f9c1_abc1234def5',
        'third_party/rl_games_cotrain': '/home/.../third_party/rl_games_cotrain/.worktrees/...',
    },
}
```

`submodule_commits` uses full 40-char SHAs for reproducibility.
`submodule_worktrees` stores the full remote path for debugging (in case cleanup
failed, the user knows exactly where to look on the remote host).

### Submission-time log output

```
[chester] Submodule commit pinning:
  IsaacLabTactile: abc1234 -> abc1234def5678abc1234def5678abc1234def5678
      worktree: IsaacLabTactile/.worktrees/03_20_14_32_a3f9c1_abc1234def5
  third_party/rl_games_cotrain: def5678 -> def5678abc1234def5678abc1234def5678abc1234
      worktree: third_party/rl_games_cotrain/.worktrees/03_20_14_32_a3f9c1_def5678abc1
```

---

## Code Changes Summary

| File | Change |
|------|--------|
| `src/chester/run_exp.py` | Add `submodule_commits` param; add `_validate_submodule_commits()` and `_build_worktree_paths()` helpers; call validation after singularity override, before variant loop; pass `submodule_worktrees` and `submodule_resolved_commits` into `generate_script()` via `gen_kwargs`; pass to `_register_job_for_pull()` |
| `src/chester/backends/base.py` | Add `mounts_override` param to `wrap_with_singularity()`; add standalone functions `_rewrite_mounts_for_worktrees()`, `_build_worktree_setup_commands()`, `_build_worktree_cleanup_commands()` |
| `src/chester/backends/slurm.py` | Add `submodule_worktrees` and `submodule_resolved_commits` to `generate_script()`; inject setup commands and pass `mounts_override` to `wrap_with_singularity()` |
| `src/chester/backends/ssh.py` | Same changes as `slurm.py` for `generate_script()` |
| `src/chester/run_exp.py` (`_register_job_for_pull`) | Add `submodule_commits` and `submodule_worktrees` keyword params; write to job dict when non-None |

No new files. No `.chester/config.yaml` schema changes.

---

## Error Cases

| Condition | Behaviour |
|-----------|-----------|
| `submodule_commits` specified but `backend.config.singularity is None` | `ValueError` at launch time (after singularity override applied, before variant loop) |
| Submodule path not found locally | `ValueError` naming the path |
| Ref not resolvable in submodule | `ValueError` naming submodule and ref |
| Remote `git worktree add` fails (e.g. commit not on remote) | Script exits non-zero under `set -e`; EXIT trap fires; `|| true` on each `worktree remove` in cleanup ensures partial-creation is safe |
| Job force-cancelled (SIGTERM) | `trap TERM` fires cleanup before exit |
| Cleanup itself fails | `|| true` swallows error; worktree path is recorded in job store for manual cleanup on remote |
| Mount source prefix collision (e.g. `IsaacLabTactile_v2`) | The `abs_submodule + "/"` suffix guard prevents false matches — `IsaacLabTactile_v2/foo` does not match submodule `IsaacLabTactile` |
