#!/bin/bash
# Great Lakes SLURM backend - environment setup
#
# Modules (singularity, cuda) are loaded by chester automatically
# based on the 'modules' and 'cuda_module' fields in config.yaml.
# Singularity wrapping is also handled automatically.
#
# Use this script for any additional environment setup your project needs:
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export CUDA_HOME=/usr/local/cuda
