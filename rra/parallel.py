"""Run independent jobs on several CPU cores (fork-based, Linux/WSL/Docker).

Used for evaluation episodes. Each job is a zero-argument function; workers
inherit the parent's objects through fork, so controllers, attackers and
simulators need not be pickled. Torch is limited to one thread per worker so
that workers do not compete with each other. Results do not depend on the
number of workers: every episode is seeded by its own seed.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, List, Optional

_JOBS: List[Callable] = []


def default_workers() -> int:
    """Number of workers: $RRA_WORKERS if set, else 1 (serial)."""
    return max(1, int(os.environ.get("RRA_WORKERS", "1")))


def available_cores() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _run(i: int):
    return _JOBS[i]()


def pmap(jobs: List[Callable], workers: Optional[int] = None) -> list:
    """Call every job and return the results in order."""
    workers = default_workers() if workers is None else workers
    workers = min(workers, len(jobs), available_cores())
    if workers <= 1 or "fork" not in mp.get_all_start_methods():
        return [j() for j in jobs]
    global _JOBS
    _JOBS = list(jobs)
    try:
        with mp.get_context("fork").Pool(workers, initializer=_init_worker) as pool:
            return pool.map(_run, range(len(jobs)), chunksize=1)
    finally:
        _JOBS = []
