#!/bin/bash
# prepare.sh - Environment setup for chester tests
# This script is sourced by chester before running experiments

# Note: The chester repo is synced to remote by launch_mnist.py before this runs,
# so pyproject.toml's `chester = { path = ".." }` works correctly.

# Set CUDA_HOME if nvcc is available
if command -v nvcc &> /dev/null; then
    export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    echo "[prepare.sh] CUDA_HOME=$CUDA_HOME"
fi
