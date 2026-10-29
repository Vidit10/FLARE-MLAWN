"""Shared interface between controllers (xApps) and simulators.

Every simulator (the fast Python simulator now, ns-3 / 5G-LENA later) implements
``RRAEnv``.  Controllers, attackers and the federation code only ever see the
objects defined here, so they never need to know which simulator is underneath.

Design rules
------------
* Arrays have a fixed length ``u_max`` (one slot per possible user) plus an
  ``active`` mask.  Models must only use rows where ``active`` is True; this keeps
  the code vectorised while the controller itself stays defined for any number
  of active users.
* ``reported`` features travel over E2 / the shared database and can be attacked.
  ``trusted`` features (the xApp's own last decision, the user's slice) cannot.
* Rewards and KPIs are always computed from what the network actually delivered
  (``StepInfo``), never from the reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# Order of the five reported per-user features.
REPORTED_FEATURES: Tuple[str, ...] = (
    "sinr_db",           # wideband SINR from the UE's CQI/CSI report
    "buffer_bits",       # gNB downlink (RLC) buffer for the user
    "demand_bps",        # recent arrival rate measured at the gNB
    "interference_dbm",  # interference per PRB measured by the UE
    "avg_rate_bps",      # long-term delivered rate (gNB counter)
)
N_REPORTED = len(REPORTED_FEATURES)

SLICES: Tuple[str, ...] = ("eMBB", "URLLC", "mMTC")
N_SLICES = len(SLICES)


@dataclass
class NormBounds:
    """Fixed bounds used to scale each reported feature to [0, 1].

    The attack budget epsilon is defined on the scaled values, so it has the
    same meaning for every feature *in scaled units*.

    Two scalings are supported per feature:
    * linear: s = (x - low) / (high - low)
    * log:    s = (g(x) - g(low)) / (g(high) - g(low)),  g(x) = log10(1 + x / x0)
      Used for buffer, demand and average rate, whose values span several
      orders of magnitude across slices.  An epsilon change is then a
      *relative* change of the raw value (a multiplicative factor), the same
      for a 10 kbit URLLC buffer as for a 10 Mbit eMBB buffer.
    """
    low: np.ndarray   # shape (N_REPORTED,)
    high: np.ndarray  # shape (N_REPORTED,)
    log_feat: Optional[np.ndarray] = None   # (N_REPORTED,) bool; None = all linear
    x0: Optional[np.ndarray] = None         # (N_REPORTED,) reference values for log features

    def __post_init__(self):
        n = len(self.low)
        self.log_feat = np.zeros(n, bool) if self.log_feat is None else np.asarray(self.log_feat, bool)
        self.x0 = np.ones(n) if self.x0 is None else np.asarray(self.x0, float)

    def _g(self, x):
        return np.where(self.log_feat, np.log10(1.0 + np.maximum(x, 0.0) / self.x0), x)

    def _g_inv(self, y):
        yl = np.where(self.log_feat, y, 0.0)          # avoid overflow on linear entries
        return np.where(self.log_feat, self.x0 * (10.0 ** yl - 1.0), y)

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        lo, hi = self._g(self.low), self._g(self.high)
        return np.clip((self._g(raw) - lo) / (hi - lo), 0.0, 1.0)

    def denormalize(self, norm: np.ndarray) -> np.ndarray:
        lo, hi = self._g(self.low), self._g(self.high)
        return self._g_inv(lo + np.clip(norm, 0.0, 1.0) * (hi - lo))

    def scaled_range(self) -> np.ndarray:
        """Width of each feature's range after the transform (dB or decades)."""
        return self._g(self.high) - self._g(self.low)


@dataclass
class Observation:
    """What the xApp sees at the start of a control interval."""
    t: int
    active: np.ndarray        # (u_max,) bool
    reported: np.ndarray      # (u_max, N_REPORTED) raw units; may be manipulated
    reported_norm: np.ndarray # (u_max, N_REPORTED) in [0, 1]
    last_prb_share: np.ndarray    # (u_max,) trusted: n_{t-1} / N
    last_power_share: np.ndarray  # (u_max,) trusted: p_{t-1} / P_max
    slice_id: np.ndarray      # (u_max,) int in {0,1,2}; trusted (core network)

    @property
    def n_active(self) -> int:
        return int(self.active.sum())

    def copy(self) -> "Observation":
        return Observation(
            t=self.t,
            active=self.active.copy(),
            reported=self.reported.copy(),
            reported_norm=self.reported_norm.copy(),
            last_prb_share=self.last_prb_share.copy(),
            last_power_share=self.last_power_share.copy(),
            slice_id=self.slice_id.copy(),
        )


@dataclass
class Action:
    """Shares chosen by a controller.

    ``prb_share`` and ``power_share`` are non-negative and each sums to at most 1
    over active users; whatever is left is "unused".  The environment turns the
    shares into integer PRB counts and powers with ``to_budgets``.
    """
    prb_share: np.ndarray    # (u_max,)
    power_share: np.ndarray  # (u_max,)


@dataclass
class StepInfo:
    """Ground truth after one interval (computed from delivered traffic)."""
    n_prb: np.ndarray           # (u_max,) PRBs actually given
    power_mw: np.ndarray        # (u_max,) power actually given
    delivered_bits: np.ndarray  # (u_max,)
    offered_bits: np.ndarray    # (u_max,) arrivals in this interval
    dropped_bits: np.ndarray    # (u_max,) buffer overflow
    rate_bps: np.ndarray        # (u_max,) delivered rate this interval
    avg_rate_bps: np.ndarray    # (u_max,) long-term rate after update
    true_sinr_db: np.ndarray    # (u_max,) true SINR with the chosen power
    delay_s: np.ndarray         # (u_max,) queueing delay estimate after service
    violated: np.ndarray        # (u_max,) bool SLA violation per user
    active: np.ndarray          # (u_max,) bool users active in this interval
    slice_id: np.ndarray        # (u_max,)
    # cell-level terms of the objective (Sec. III-D of the report)
    utility: float              # Phi_t
    power_frac: float           # P_t
    prb_frac: float             # B_t
    slice_violation: np.ndarray # (N_SLICES,) V_{l,t}; nan if slice has no users
    extra: Dict[str, float] = field(default_factory=dict)


class RRAEnv:
    """Base class every simulator implements."""

    n_prb: int
    p_max_mw: float
    u_max: int
    t_c: float
    bounds: NormBounds

    def reset(self, seed: int | None = None) -> Observation:
        raise NotImplementedError

    def step(self, action: Action) -> Tuple[Observation, StepInfo, bool]:
        """Apply one interval of allocation; return (next_obs, info, done)."""
        raise NotImplementedError


def to_budgets(prb_share: np.ndarray, power_share: np.ndarray, active: np.ndarray,
               n_prb: int, p_max_mw: float) -> Tuple[np.ndarray, np.ndarray]:
    """Turn shares into feasible budgets.

    * PRBs: largest-remainder rounding, so sum(n) <= n_prb exactly and n is integer.
    * Power: p = share * P_max, so sum(p) <= P_max.
    * Inactive users get nothing; a user with 0 PRBs gets 0 power.
    """
    x = np.where(active, np.clip(prb_share, 0.0, None), 0.0)
    y = np.where(active, np.clip(power_share, 0.0, None), 0.0)
    if x.sum() > 1.0:
        x = x / x.sum()
    if y.sum() > 1.0:
        y = y / y.sum()
    target = n_prb * x
    n = np.floor(target).astype(int)
    used_target = int(np.floor(target.sum() + 1e-9))
    remaining = used_target - int(n.sum())
    if remaining > 0:
        frac = target - n
        order = np.argsort(-frac, kind="stable")
        n[order[:remaining]] += 1
    p = y * p_max_mw
    p = np.where(n > 0, p, 0.0)
    return n, p
