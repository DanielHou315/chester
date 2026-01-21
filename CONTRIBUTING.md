# Contributing to Chester

## Development Setup

```bash
git clone https://github.com/DanielHou315/chester.git
cd chester
uv sync
```

## Branch Structure

- `main` - Development branch, all changes merge here first
- `release` - Production branch, synced from main for releases

## Release Workflow

Releases are done locally using twine. The workflow ensures code is tested on `main` before being published.

### Prerequisites

1. Ensure twine is available: `uv add --dev twine` or `pip install twine`
2. Configure `~/.pypirc` with PyPI credentials:
   ```ini
   [distutils]
   index-servers = pypi

   [pypi]
   username = __token__
   password = pypi-xxxx...
   ```

### Release Steps

1. **Ensure you're on main with clean state:**
   ```bash
   git checkout main
   git pull origin main
   git status  # Should be clean
   ```

2. **Bump version number:**
   ```bash
   # Use the bump script (patch/minor/major)
   python scripts/bump_version.py patch   # 0.3.0 -> 0.3.1
   python scripts/bump_version.py minor   # 0.3.0 -> 0.4.0
   python scripts/bump_version.py major   # 0.3.0 -> 1.0.0
   ```

3. **Commit and push to main:**
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

4. **Sync to release branch:**
   ```bash
   git checkout release
   git merge main --ff-only
   git push origin release
   ```

5. **Build and publish:**
   ```bash
   rm -rf dist/
   uv build
   twine upload dist/*
   ```

6. **Tag the release:**
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

7. **Return to main:**
   ```bash
   git checkout main
   ```

### Quick Release (Using Claude Skill)

If you have Claude Code configured, use the `/release` skill:

```
/release patch   # For bug fixes
/release minor   # For new features
/release major   # For breaking changes
```

The skill automates all the steps above.

## Code Style

- Use type hints where practical
- Follow PEP 8
- Keep functions focused and documented

## Testing

```bash
# Run tests directory
cd tests && uv run python launch_mnist.py --mode local --quick

# Test config loading
uv run python -m chester.config
```
