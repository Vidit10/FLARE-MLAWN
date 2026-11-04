#!/usr/bin/env bash
# Whole week-3 pipeline (Linux, WSL or the Docker image), in the order used for the results.
# Usage: bash scripts/week3_all.sh [workers]     (default: 4)
# Steps overwrite their outputs; move results/week3 away first to keep an old run.
# Time on 2 cores: about 13 h; with 12 cores about 3 h.
set -euo pipefail
W=${1:-4}
export RRA_WORKERS=$W
python -m experiments.week3_sanitizer                                   # detector / sanitizer
python -m experiments.week3_train --runs all clean_none --seeds 0 1 2 --jobs "$W"
python -m experiments.week3_eval --part all --episodes 8 --seed 0 --workers "$W"   # detect, T2, E6, zeta
for s in 1 2; do
  python -m experiments.week3_eval --part e6 --episodes 8 --seed $s --workers "$W" --save-dir results/week3/seed$s
  python -m experiments.week3_eval --part zeta --episodes 8 --seed $s --workers "$W" --save-dir results/week3/seed$s
  python -m experiments.week3_eval --part detect --episodes 8 --seed $s --workers "$W" --save-dir results/week3/seed$s
done
python -m experiments.week3_e7 --part all --episodes 8 --seed 0 --workers "$W"     # suite, eps, users
python -m experiments.week3_seeds                                       # mean and spread over seeds
