"""Non-learning controllers used as reference points.

Both read only the (possibly attacked) observation, exactly like the DRL xApp
will, so the same attacker can be pointed at them.
"""
from __future__ import annotations

import numpy as np

from ..env import link
from ..env.interface import Action, Observation, RRAEnv


class Controller:
    name = "controller"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError


class EqualShare(Controller):
    """Every active user gets the same PRB and power share."""
    name = "EqualShare"

    def __init__(self, env: RRAEnv):
        self.env = env

    def act(self, obs: Observation) -> Action:
        a = obs.active.astype(float)
        share = a / max(a.sum(), 1.0)
        return Action(prb_share=share, power_share=share.copy())


class ProportionalFair(Controller):
    """Proportional-fair scheduler with a demand cap and equal power per PRB.

    Over one control interval PRBs are handed out one at a time to the user with
    the largest PF metric  se_u / Rbar_u,  where Rbar_u is the moving-average rate
    *including* what the user has already been given in this interval (so the
    allocation spreads out, as time-domain PF does over the many slots inside
    one interval).  A user stops receiving PRBs once the PRBs given are enough to
    send its whole reported buffer plus expected arrivals (demand cap).
    Power is equal per PRB: power share = PRB share.

    Inputs used (all reported, therefore attackable): SINR, buffer, demand,
    average rate.
    """
    name = "PF"

    def __init__(self, env: RRAEnv, use_demand_cap: bool = True):
        self.env = env
        r = env.cfg.radio
        self.alpha, self.sinr_min, self.se_max = r.se_alpha, r.sinr_min_db, r.se_max
        self.prb_bw = r.prb_bw_hz
        self.tw = env.cfg.avg_rate_window
        self.use_demand_cap = use_demand_cap

    def act(self, obs: Observation) -> Action:
        env = self.env
        act = obs.active
        sinr_db = obs.reported[:, 0]
        se = link.spectral_efficiency(10 ** (sinr_db / 10), self.alpha, self.sinr_min, self.se_max)
        bits_per_prb = self.prb_bw * se * env.t_c
        need_bits = obs.reported[:, 1] + obs.reported[:, 2] * env.t_c   # buffer + arrivals
        rbar = np.maximum(obs.reported[:, 4], 1e3)
        eligible = act & (se > 0)
        if self.use_demand_cap:
            cap = np.where(bits_per_prb > 0, np.ceil(need_bits / np.maximum(bits_per_prb, 1e-9)), 0)
        else:
            cap = np.full(env.u_max, env.n_prb)
        n = np.zeros(env.u_max, int)
        for _ in range(env.n_prb):
            ok = eligible & (n < cap)
            if not ok.any():
                break
            r_eff = (1 - 1 / self.tw) * rbar + (n * bits_per_prb / env.t_c) / self.tw
            metric = np.where(ok, se / r_eff, -np.inf)
            n[int(np.argmax(metric))] += 1
        share = n / env.n_prb
        return Action(prb_share=share, power_share=share.copy())
