# Release Skill

Automates the chester-ml release workflow using twine.

## Usage

```
/release patch   # Bug fixes (0.3.0 -> 0.3.1)
/release minor   # New features (0.3.0 -> 0.4.0)
/release major   # Breaking changes (0.3.0 -> 1.0.0)
```

## Instructions

When the user invokes `/release [patch|minor|major]`, perform these steps in order:

### 1. Pre-flight Checks

```bash
# Ensure on main branch
git checkout main

# Pull latest
git pull origin main

# Check for uncommitted changes
git status --porcelain
```

If there are uncommitted changes, STOP and ask the user to commit or stash them first.

### 2. Verify Critical Files Are Tracked

Check that essential package files are not gitignored:

```bash
# These should return nothing (not ignored)
git check-ignore src/chester/config.py
git check-ignore src/chester/run_exp.py
```

If any critical files are gitignored, fix `.gitignore` and add them with `git add -f`.

### 3. Run Tests Locally

**IMPORTANT:** Tests must be run locally before releasing. They cannot run on GitHub CI because they require SSH access to remote hosts.

```bash
cd tests
uv sync
uv run python -m pytest test_local.py -v
```

All tests must pass before proceeding. If tests fail:
1. Debug and fix the issues
2. Commit the fixes
3. Re-run tests until they pass

### 4. Bump Version

Run the bump script with the specified part (default: patch):

```bash
python scripts/bump_version.py <part>
```

Note the new version number from the output.

### 5. Commit Version Bump

```bash
git add pyproject.toml
git commit -m "Bump version to <new_version>"
git push origin main
```

### 6. Sync to Release Branch

```bash
git checkout release
git merge main --ff-only
git push origin release
```

### 7. Build Package

```bash
rm -rf dist/
uv build
```

Verify the dist/ directory contains the wheel and tarball.

### 8. Publish to PyPI

```bash
twine upload dist/*
```

This uses credentials from `~/.pypirc`. If it fails, inform the user to check their PyPI credentials.

### 9. Tag Release

```bash
git tag -a v<new_version> -m "Release v<new_version>"
git push origin v<new_version>
```

### 10. Return to Main

```bash
git checkout main
```

### 11. Summary

Report to the user:
- New version number
- PyPI package URL: https://pypi.org/project/chester-ml/<new_version>/
- Git tag created

## Error Handling

- If any git command fails, stop and report the error
- If twine upload fails, the user may need to configure `~/.pypirc`
- If merge fails (not fast-forward), the release branch may have diverged - ask user how to proceed
