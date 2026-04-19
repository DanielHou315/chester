# Submodule Commit Pinning

Chester can pin specific submodule commits per job using git worktrees on the remote host. This allows running different experiments against different versions of a submodule (e.g., comparing algorithm versions without full resyncs) in the same launcher run.

**Requires Singularity to be active** — the worktree paths are injected as bind-mount overrides.

## Usage

```python
run_experiment_lite(
    ...,
    submodule_commits={
        'IsaacLabTactile': 'abc1234',        # short SHA, branch, or tag
        'third_party/mylib': 'main',          # branch name resolved to SHA
    },
)
```

Each entry is `{submodule_path: git_ref}` where:
- `submodule_path`: path relative to project root
- `git_ref`: any ref resolvable by `git rev-parse` (SHA, branch, tag, HEAD~N, etc.)

Chester:
1. Resolves each ref locally to a full 40-char SHA via `git rev-parse`
2. Prints the resolved SHA and worktree path for each submodule
3. On the remote: creates a git worktree at `.worktrees/<submodule>/<sha>/` inside `remote_dir`
4. Rewrites the matching Singularity bind mounts to point to the worktree path
5. On job exit: cleanup trap removes the worktree

## Requirements

- Singularity must be active for the backend (`enabled: true` or `use_singularity=True`)
- The submodule must exist at the given path in the local repo
- The remote must have the commit available (it's rsynced as part of `.git/`)
- The submodule path must appear in the backend's `singularity.mounts` for the rewrite to work

## What happens on the remote

```bash
# Created by Chester in the generated script:
CHESTER_WT_0=/home/user/myproject/.worktrees/IsaacLabTactile/abc1234.../
git -C /home/user/myproject/IsaacLabTactile worktree add "$CHESTER_WT_0" abc1234...

_chester_wt_cleanup() {
    git -C /home/user/myproject/IsaacLabTactile worktree remove --force "$CHESTER_WT_0"
}
trap '_chester_wt_cleanup' EXIT

# Singularity mount is rewritten from:
#   IsaacLabTactile/source:/workspace/IsaacLabTactile/source
# to:
#   /home/user/myproject/.worktrees/IsaacLabTactile/abc1234.../source:/workspace/IsaacLabTactile/source
```

## Console output at launch time

Chester prints a summary for each pinned submodule:
```
[chester] Submodule commit pinning:
  IsaacLabTactile: main -> abc1234def5678...
      worktree: .worktrees/IsaacLabTactile/abc1234def5678.../
```

## Example: A/B comparison sweep

```python
vg = VariantGenerator()
vg.add('seed', [1, 2, 3])
vg.add('lib_version', ['v1.0', 'v2.0-beta'])   # conceptual label only

# Map version labels to git refs
version_to_ref = {'v1.0': 'v1.0.0', 'v2.0-beta': 'feature/new-algo'}

for v in vg.variants():
    run_experiment_lite(
        ...,
        submodule_commits={'third_party/mylib': version_to_ref[v['lib_version']]},
    )
```
