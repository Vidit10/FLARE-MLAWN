#!/usr/bin/env bash
# Week-5 pipeline (Linux, macOS, WSL or the Docker image): federation size x users
# (fl_scale, fl_learn) and the single-cell comparison of every controller under every
# attack (compare), then the tables and figures in results/week5/summary/.
# Usage: bash scripts/week5_all.sh [parallel runs] [seeds...]   (defaults: 4, seeds 0 1 2)
# About 200 core-hours for three seeds (roughly a third per seed); runs are ordered by seed,
# and finished runs are skipped when the script is started again.
set -euo pipefail
J=${1:-4}
shift || true
SEEDS=${*:-0 1 2}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/week5
python -m experiments.week5_launch --grid all --seeds $SEEDS --jobs "$J" 2>&1 | tee -a results/week5/launch.log
python -m experiments.week5_eval 2>&1 | tee results/week5/eval_summary.log
