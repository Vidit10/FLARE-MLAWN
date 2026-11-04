"""Week 3, Stages 1-2: collect clean reports, train the detector and sanitizer.

Usage: python -m experiments.week3_sanitizer [--episodes 60] [--out results/week3]

Data: clean per-user reports from the Python simulator with every clean
controller of weeks 1-2 (EqualShare, PF, the 3 trained PPO seeds), loads
U = 5..40, seeds 50 000+ (disjoint from training and evaluation seeds).
Split 70/15/15 for autoencoder training, conformal calibration and test.
Output: <out>/sanitizer.pt and sanitizer.json (threshold, test false-positive
rate, denoising errors, data sizes).
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from rra.agents.ppo import PPOController
from rra.baselines import EqualShare, ProportionalFair
from rra.config import EnvConfig
from rra.defence.sanitizer import SanitizerConfig, collect_reports, train_sanitizer
from rra.env.pysim import PySimEnv

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", default="results/week3")
    ap.add_argument("--no-cond", action="store_true", help="do not condition on the slice (report's original AE)")
    ap.add_argument("--name", default="sanitizer")
    a = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    env = PySimEnv(EnvConfig(feature_scale="log"))
    ctrls = [EqualShare(env), ProportionalFair(env)] + [
        PPOController.load(p) for p in sorted(glob.glob("results/week2/ppo_s*/actor.pt"))]
    S, C = collect_reports(env, ctrls, a.episodes)
    print(f"collected {len(S)} clean reports in {time.time() - t0:.0f} s", flush=True)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(os.path.join(a.out, "clean_reports.npz"), S=S, C=C)
    cfg = SanitizerConfig(cond_slice=not a.no_cond)
    san, stats = train_sanitizer(S, C, cfg)
    stats["runtime_s"] = round(time.time() - t0, 1)
    stats["simulator"] = "rra.env.pysim (Python)"
    san.save(os.path.join(a.out, f"{a.name}.pt"), stats)
    print(f"saved {a.out}/{a.name}.pt  ({time.time() - t0:.0f} s)")
