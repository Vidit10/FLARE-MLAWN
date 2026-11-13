"""Steps F and G of the FLARE report: telling traffic shifts from attacks, and the observer rApp.

Step F. Every cell reports a short summary of its situation each window:
    kappa = [load U / u_max, slice mix (3), median of its trusted reports (5)]
and the per-user trust scores of the window. The observer asks the report's
two questions:
    (1) how many users look suspicious?   f_flag = share of users with trust < 0.5
    (2) has any other cell seen this?      d_lib  = distance from kappa to the nearest
                                                    library context of another cell
and also how far the cell has moved from its own stored context, d_own. Rule:
    attack     f_flag >= f_attack (many users distrusted)
    shift      the context moved (d_own > d_change), few users look odd
               (f_flag <= f_odd) and the new context is closer to another
               cell's than to the cell's own (d_lib < d_own): reuse that policy
    attack     more users look odd than clean windows ever show (f_flag > f_odd)
    unclear    the context moved but matches no other cell (keep the policy,
               set the data aside)
    no change  otherwise
f_odd and d_change are calibrated on clean windows (experiments/week4_observer.py).
The report-level drift (MMD) stays in the data trust of Step D; within one cell
it varies too much with the load to detect a change on its own.

Step G. The observer watches the cell's objective J. A sharp drop (below
``drop`` times the running level) that Step F calls an attack triggers
retraining in the digital twin, a shadow test (the candidate runs on replayed
conditions next to the live model without taking control) and deployment
only if the candidate is better; after deployment, a renewed drop rolls the
cell back to the previous model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..env.interface import N_SLICES


def cell_summary(reports: np.ndarray, slices: np.ndarray, trust: np.ndarray, n_users: float,
                 u_max: int = 40) -> np.ndarray:
    """kappa: load, slice mix, median trusted report (users with trust >= 0.5)."""
    mix = np.bincount(slices, minlength=N_SLICES) / max(len(slices), 1)
    ok = trust >= 0.5
    med = np.median(reports[ok], 0) if ok.any() else np.median(reports, 0)
    return np.concatenate([[n_users / u_max], mix, med]).astype(float)


@dataclass
class ShiftRule:
    f_attack: float = 0.20       # this share of users distrusted: attack
    f_odd: float = 0.06          # more distrusted users than clean windows show (calibrated)
    d_change: float = 0.15       # the context moved this far from the cell's own (calibrated)

    def decide(self, f_flag: float, d_lib: float, d_own: float) -> str:
        if f_flag >= self.f_attack:
            return "attack"
        changed = d_own > self.d_change
        if changed and f_flag <= self.f_odd:
            return "shift" if d_lib < d_own else "unclear"
        if f_flag > self.f_odd:
            return "attack"
        return "no change"


def nearest(kappa: np.ndarray, library: Dict[str, np.ndarray], exclude: Sequence[str] = ()) -> tuple:
    best, bd = None, np.inf
    for k, v in library.items():
        if k in exclude:
            continue
        d = float(np.linalg.norm(kappa - v))
        if d < bd:
            best, bd = k, d
    return best, bd


@dataclass
class ObserverState:
    level: Optional[float] = None    # running level of J
    alpha: float = 0.2               # smoothing of the running level
    drop: float = 0.9                # a window below drop * level is a sharp drop

    def update(self, J: float) -> bool:
        """Feed one window's J; return True on a sharp drop."""
        if self.level is None:
            self.level = J
            return False
        dropped = J < self.drop * self.level
        if not dropped:
            self.level = (1 - self.alpha) * self.level + self.alpha * J
        return dropped
