# Git Snapshot

Chester automatically saves the git state to the experiment log directory before launching. This records everything needed to reproduce the exact code state of each run.

**Enabled by default.** Disable with:
```python
run_experiment_lite(..., git_snapshot=False)
```

## Files written to `{log_dir}/`

| File | Contents |
|------|----------|
| `git_info.json` | Commit hash, branch, dirty flag, timestamp, submodule status, untracked symlinks |
| `git_diff.patch` | Full unified diff of all staged/unstaged changes; untracked file names as comments; dirty submodule diffs in labelled sections |

`git_diff.patch` is only written when there are uncommitted changes (in the parent repo or any submodule).

## `git_info.json` schema

```json
{
  "commit": "abc1234...",
  "branch": "main",
  "dirty": true,
  "timestamp": "2026-04-19T12:34:56",
  "submodules": [
    {
      "path": "third_party/mylib",
      "hash": "def5678...",
      "dirty": false,
      "untracked_files": []
    }
  ],
  "untracked_symlinks": ["data -> /mnt/nfs/data"]
}
```

## Submodule tracking

For each submodule:
- `dirty: bool` — whether the submodule has uncommitted changes
- `untracked_files: list` — list of untracked files in the submodule
- If dirty, the submodule's diff is appended to `git_diff.patch` under a `# === Submodule: <path> ===` header

## Recovery

Restore exact code state for a past run:

```bash
# From git_info.json
git checkout <commit>          # restore main repo
git apply git_diff.patch       # restore uncommitted changes

# For each submodule (from git_info.json submodules[*]):
cd third_party/mylib
git checkout <submodule_hash>
# apply the submodule section from git_diff.patch if dirty
```
