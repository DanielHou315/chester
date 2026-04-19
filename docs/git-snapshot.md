# Git Snapshot

Chester automatically saves the git state to the experiment log directory before launching. This records everything needed to reproduce the exact code state of each run.

**Enabled by default.** Disable with:
```python
run_experiment_lite(..., git_snapshot=False)
```

## Files written to `{log_dir}/`

Files are written into the **first variant's local log directory** (the same directory where `output.log` lands for the first job in a batch).

| File | Contents |
|------|----------|
| `git_info.json` | Commit hash, branch, dirty flag, timestamp, submodule status, untracked symlinks |
| `git_diff.patch` | `git diff HEAD` output (staged + unstaged tracked changes); untracked file names listed as comments; dirty submodule diffs appended in labelled sections |

`git_diff.patch` is only written when there are uncommitted changes in the parent repo or any submodule. Untracked file **names** (not content) are recorded as comments.

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
      "status": "up_to_date",
      "dirty": false,
      "untracked_files": []
    }
  ],
  "untracked_symlinks": [
    {
      "path": "data",
      "target": "/mnt/nfs/data",
      "target_exists": true
    }
  ]
}
```

### Field notes

- `dirty` (top-level): `true` if `git status --porcelain` reports any tracked-file changes, ignoring submodule state.
- `submodules[*].status`: one of `"up_to_date"`, `"modified"`, `"uninitialized"`, `"merge_conflict"` (from the leading character of `git submodule status`).
- `submodules[*].description`: optional — present only when `git submodule status` includes a tag/branch annotation in parentheses.
- `untracked_symlinks`: symlinks present in the working tree that are **not** tracked by git (mode `120000`). Common directories (`data/`, `wandb/`, `.venv/`, `.worktrees/`, etc.) are excluded from the scan.

## Submodule tracking

For each submodule:
- `status: str` — `"up_to_date"` | `"modified"` | `"uninitialized"` | `"merge_conflict"`
- `dirty: bool` — whether the submodule working tree has uncommitted changes
- `untracked_files: list[str]` — untracked file paths inside the submodule
- If dirty, the submodule's diff (`git diff HEAD` inside the submodule) is appended to `git_diff.patch` under a `# === Submodule: <path> ===` header, followed by untracked file names as comments

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
