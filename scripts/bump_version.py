#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import sys

def bump_version(part):
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("Error: pyproject.toml not found")
        sys.exit(1)
        
    content = pyproject_path.read_text()
    
    # Regex to find version with loose spacing
    pattern = r'version\s*=\s*"(\d+)\.(\d+)\.(\d+)"'
    match = re.search(pattern, content)
    
    if not match:
        print("Error: Could not find version pattern 'version = \"x.y.z\"' in pyproject.toml")
        sys.exit(1)
    
    major, minor, patch = map(int, match.groups())
    old_version = f"{major}.{minor}.{patch}"
    
    if part == "major":
        major += 1
        minor = 0
        patch = 0
    elif part == "minor":
        minor += 1
        patch = 0
    elif part == "patch":
        patch += 1
    
    new_version = f"{major}.{minor}.{patch}"
    new_content = re.sub(pattern, f'version = "{new_version}"', content, count=1)
    
    pyproject_path.write_text(new_content)
    print(f"Bumped version: {old_version} -> {new_version}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bump version in pyproject.toml")
    parser.add_argument("part", choices=["major", "minor", "patch"], default="patch", nargs="?", 
                        help="Part of version to bump (default: patch)")
    args = parser.parse_args()
    bump_version(args.part)
