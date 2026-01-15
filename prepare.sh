#!/usr/bin/env bash
# prepare.sh - Chester environment setup script
#
# This script is sourced by chester before running experiments.
# Package manager setup (uv/conda) is handled by chester based on chester.yaml config.
#
# This file is for custom project-specific setup that runs AFTER package manager setup.
# Common uses:
#   - Setting custom environment variables
#   - Running project-specific initialization
#   - Compiling custom CUDA extensions
#
# Note: For backwards compatibility, this file is optional.
# If you don't need custom setup, you can delete this file.

echo "[chester] prepare.sh loaded"

# Add any custom project setup below:
# export MY_CUSTOM_VAR=value
# ./my_custom_script.sh
