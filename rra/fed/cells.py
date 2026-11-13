"""The eight cells of the federation (FLARE report: C = 8, non-identical traffic).

Every cell uses the same radio set-up as the single-cell experiments (report
Table "Parameters"), but differs in slice mix, load range and neighbour load
(interference), so the cells' data are not identically distributed. The
numbers are scenario choices, not measurements; they span hot spots, light
suburban cells, URLLC-heavy factories and IoT-heavy districts.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Tuple

from ..config import EnvConfig, RadioConfig


@dataclass(frozen=True)
class CellSpec:
    name: str
    slice_mix: Tuple[float, float, float]      # eMBB, URLLC, mMTC
    users: Tuple[int, int]                     # mean load drawn per episode
    neighbour_load: Tuple[float, float]        # interference from the 6 neighbours


CELLS = [
    CellSpec("eMBB hot spot", (0.70, 0.15, 0.15), (15, 30), (0.5, 0.9)),
    CellSpec("URLLC factory", (0.15, 0.70, 0.15), (5, 20), (0.3, 0.6)),
    CellSpec("IoT district", (0.15, 0.15, 0.70), (15, 35), (0.3, 0.7)),
    CellSpec("mixed urban", (0.40, 0.30, 0.30), (5, 30), (0.3, 0.8)),
    CellSpec("light suburban", (0.50, 0.25, 0.25), (3, 12), (0.1, 0.4)),
    CellSpec("stadium", (0.60, 0.30, 0.10), (25, 40), (0.6, 0.9)),
    CellSpec("campus", (0.35, 0.45, 0.20), (8, 25), (0.3, 0.7)),
    CellSpec("industrial IoT", (0.10, 0.40, 0.50), (10, 30), (0.4, 0.8)),
]

# The cell the aggregator's own reference model trains on (FLTrust's "root
# dataset"): a clean digital-twin copy of the default single-cell scenario.
ROOT_CELL = CellSpec("root (digital twin)", (0.40, 0.30, 0.30), (5, 30), (0.3, 0.8))


def cell_env_config(spec: CellSpec, base: EnvConfig = None) -> EnvConfig:
    base = base or EnvConfig(feature_scale="log")
    radio = dataclasses.replace(base.radio, neighbour_load=spec.neighbour_load)
    return dataclasses.replace(base, radio=radio, slice_mix=spec.slice_mix, target_users=spec.users)


# Order in which cells join when the federation is smaller than eight (scaling runs):
# each prefix mixes different traffic, e.g. 2 cells = mixed urban + URLLC factory,
# 4 cells = + eMBB hot spot + IoT district.
SCALE_ORDER = (3, 1, 0, 2, 5, 6, 7, 4)


def cell_subset(n: int, users: int = None):
    """The first n cells of SCALE_ORDER, with a fixed number of users per cell if given.

    Returns (ids into CELLS, specs)."""
    if not 1 <= n <= len(CELLS):
        raise ValueError(f"n must be 1..{len(CELLS)}")
    ids = list(SCALE_ORDER[:n])
    specs = [CELLS[i] if users is None else dataclasses.replace(CELLS[i], users=(users, users)) for i in ids]
    return ids, specs
