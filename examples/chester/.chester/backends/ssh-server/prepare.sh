#!/bin/bash
# SSH backend - load project environment via direnv
direnv allow .
eval "$(direnv export bash)"
