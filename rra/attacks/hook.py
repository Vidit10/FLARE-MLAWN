"""The attacker: a function that sits between the simulator and the xApp.

    simulator --reports--> [ Attacker.perturb ] --reports'--> xApp --action--> simulator

It changes only the *reported* features, only for the users it controls, and
never by more than ``eps`` per feature in the normalised [0, 1] scale (an
L-infinity budget).  The network's ground truth (true channel, real buffer,
delivered bits) is never touched, so the damage shows up in the KPIs exactly
as it would in a real cell.

Attack points (which features an attacker in that position can change):
* A1 - compromised UE: its own channel reports only (SINR and interference).
* A2 - E2 interface (between the gNB and the near-RT RIC): every reported
       feature of every user it chooses.
* A3 - malicious xApp writing to the shared database (SDL): same reach as A2.

Strategies:
* ``noise``      - uniform random noise in [-eps, eps] (non-adversarial control).
* ``direction``  - push each feature in a fixed direction (e.g. inflate SINR).
* ``search``     - query-based: try ``n_candidates`` random sign patterns, ask
                   the controller what it would do for each, keep the one whose
                   allocation is worst for the true network (a black-box attack:
                   it needs only to query the controller, not its gradients).
Gradient attacks (FGSM / PGD) on the DRL policy plug into the same class in
Phase 2 through ``strategy="gradient"``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from ..env import link
from ..env.interface import (N_REPORTED, REPORTED_FEATURES, Action, NormBounds,
                             Observation, RRAEnv, to_budgets)

FEATURE_INDEX = {f: i for i, f in enumerate(REPORTED_FEATURES)}

ATTACK_POINTS: Dict[str, Sequence[str]] = {
    "A1": ("sinr_db", "interference_dbm"),
    "A2": REPORTED_FEATURES,
    "A3": REPORTED_FEATURES,
}


def eps_in_physical_units(bounds: NormBounds, eps: float) -> Dict[str, float]:
    """What a normalised budget eps means for each feature, in its own unit."""
    return {f: float(eps * (bounds.high[i] - bounds.low[i]))
            for i, f in enumerate(REPORTED_FEATURES)}


@dataclass
class Attacker:
    eps: float = 0.0
    point: str = "A2"
    k: Optional[int] = None             # number of users controlled (None = all)
    strategy: str = "noise"             # noise | direction | search
    features: Optional[Sequence[str]] = None  # subset of the point's features
    direction: Optional[Dict[str, float]] = None  # sign per feature for "direction"
    n_candidates: int = 16
    name: str = ""
    _rng: np.random.Generator = field(default_factory=np.random.default_rng, repr=False)
    _order: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        allowed = ATTACK_POINTS[self.point]
        feats = tuple(self.features) if self.features is not None else tuple(allowed)
        bad = [f for f in feats if f not in allowed]
        if bad:
            raise ValueError(f"attack point {self.point} cannot change {bad}")
        self.feature_mask = np.zeros(N_REPORTED, bool)
        for f in feats:
            self.feature_mask[FEATURE_INDEX[f]] = True
        if not self.name:
            k = "all" if self.k is None else self.k
            self.name = f"{self.point}-{self.strategy}-eps{self.eps:g}-k{k}"

    # ----------------------------------------------------------------- setup
    def reset(self, env: RRAEnv, controller=None, seed: Optional[int] = None) -> None:
        self.env = env
        self.controller = controller
        self._rng = np.random.default_rng(seed)
        # fixed random priority over user slots: the attacker controls the k
        # active users that come first in this order (stable while they stay)
        self._order = self._rng.permutation(env.u_max)

    def controlled_users(self, obs: Observation) -> np.ndarray:
        mask = np.zeros(obs.active.size, bool)
        act_in_order = [u for u in self._order if obs.active[u]]
        chosen = act_in_order if self.k is None else act_in_order[: self.k]
        mask[chosen] = True
        return mask

    def entry_mask(self, obs: Observation) -> np.ndarray:
        """(u_max, N_REPORTED) True where the attacker may change a value."""
        return np.outer(self.controlled_users(obs), self.feature_mask)

    # ----------------------------------------------------------- perturbation
    def apply_delta(self, obs: Observation, delta: np.ndarray) -> Observation:
        """Project delta onto the budget and the [0, 1] box, return new obs."""
        m = self.entry_mask(obs)
        d = np.clip(delta, -self.eps, self.eps) * m
        out = obs.copy()
        out.reported_norm = np.clip(obs.reported_norm + d, 0.0, 1.0)
        changed = m & (d != 0)
        raw_new = self.env.bounds.denormalize(out.reported_norm)
        out.reported = np.where(changed, raw_new, obs.reported)
        return out

    def perturb(self, obs: Observation) -> Observation:
        if self.eps <= 0:
            return obs
        if self.strategy == "noise":
            delta = self._rng.uniform(-self.eps, self.eps, size=obs.reported.shape)
        elif self.strategy == "direction":
            sign = np.zeros(N_REPORTED)
            for f, s in (self.direction or {"sinr_db": +1.0}).items():
                sign[FEATURE_INDEX[f]] = np.sign(s)
            delta = np.tile(sign * self.eps, (obs.reported.shape[0], 1))
        elif self.strategy == "search":
            delta = self._search(obs)
        else:
            raise ValueError(self.strategy)
        return self.apply_delta(obs, delta)

    # ------------------------------------------------------- query-based search
    def _search(self, obs: Observation) -> np.ndarray:
        if self.controller is None:
            raise ValueError("search attack needs reset(env, controller)")
        best_delta, best_score = np.zeros_like(obs.reported), predicted_utility(self.env, obs, self.controller.act(obs))
        for _ in range(self.n_candidates):
            delta = self.eps * self._rng.choice([-1.0, 1.0], size=obs.reported.shape)
            score = predicted_utility(self.env, obs, self.controller.act(self.apply_delta(obs, delta)))
            if score < best_score:
                best_delta, best_score = delta, score
        return best_delta


def predicted_utility(env: RRAEnv, true_obs: Observation, action: Action) -> float:
    """Attacker's one-step estimate of the cell utility Phi for an action.

    Uses the *true* reports (the attacker sees them before changing them):
    rate from true SINR scaled by the power actually given per PRB, capped by
    the data the user really has, then the moving-average update and the log
    utility of the report.  This is only the attacker's guess; the real damage
    is measured by the simulator.
    """
    r = env.cfg.radio
    n, p = to_budgets(action.prb_share, action.power_share, true_obs.active, env.n_prb, env.p_max_mw)
    ppp = np.where(n > 0, p / np.maximum(n, 1), 0.0)
    gain = np.where(n > 0, ppp / (env.p_max_mw / env.n_prb), 0.0)
    sinr_lin = 10 ** (true_obs.reported[:, 0] / 10) * gain
    se = link.spectral_efficiency(sinr_lin, r.se_alpha, r.sinr_min_db, r.se_max)
    bits = np.minimum(n * r.prb_bw_hz * se * env.t_c,
                      true_obs.reported[:, 1] + true_obs.reported[:, 2] * env.t_c)
    tw = env.cfg.avg_rate_window
    rbar = (1 - 1 / tw) * true_obs.reported[:, 4] + (bits / env.t_c) / tw
    a = true_obs.active
    return float(np.mean(np.log1p(rbar[a] / env.cfg.r_ref_bps))) if a.any() else 0.0
