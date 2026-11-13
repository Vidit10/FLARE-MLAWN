#!/usr/bin/env bash
# Week-4 pipeline (Linux, WSL or the Docker image): federation runs (E8, E9),
# observer runs (Steps F-G), then the tables and figures.
# Usage: bash scripts/week4_all.sh [parallel runs]   (default 4)
# About 65 core-hours in total; finished runs are skipped on a restart.
set -euo pipefail
J=${1:-4}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.week4_launch --grid all --seeds 0 1 2 --jobs "$J"
python -m experiments.week4_eval --part all
