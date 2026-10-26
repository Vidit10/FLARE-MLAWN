"""Fast Python simulator of one downlink cell with three slices.

This is the "analytical" environment of the plan: it implements the radio,
traffic, queue and SLA model of the report (Sec. III-B/C) at the granularity of
one control interval T_c.  It is meant for development and for the large
experiments (many cells, seeds and attacks); key results are later re-checked
in ns-3 / 5G-LENA through the same ``RRAEnv`` interface.

Modelling choices (documented in docs/week1_design.md):
* Geometry: target gNB at the origin, 6 interfering gNBs on a hexagon with
  inter-site distance ISD; users are dropped uniformly and kept only if the
  target cell is their strongest cell (best-server association).
* Channel: 3GPP TR 38.901 UMa path loss + LOS probability + log-normal
  shadowing (fixed per user), and fast fading averaged over the interval.
* Interference: each neighbour transmits on a fraction rho_j(t) of its PRBs at
  nominal power; rho_j drifts slowly, so measured interference changes over time.
* Link model: attenuated, truncated Shannon; delivered bits are limited by both
  capacity and the data waiting in the buffer.
* Randomness never depends on the controller's actions, so every method sees the
  same users, traffic and channels for a given seed (fair comparisons).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..config import EnvConfig
from . import link
from .interface import (N_REPORTED, N_SLICES, Action, NormBounds, Observation,
                        RRAEnv, StepInfo, to_budgets)

EMBB, URLLC, MMTC = 0, 1, 2


class PySimEnv(RRAEnv):
    def __init__(self, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        r = self.cfg.radio
        self.n_prb = r.n_prb
        self.p_max_mw = float(link.dbm_to_mw(r.p_max_dbm))
        self.u_max = self.cfg.u_max
        self.t_c = self.cfg.t_c
        self.noise_mw = float(link.dbm_to_mw(
            r.noise_psd_dbm_hz + 10 * np.log10(r.prb_bw_hz) + r.ue_noise_figure_db))
        self.p_prb_nominal = self.p_max_mw / self.n_prb
        ang = np.deg2rad(30 + 60 * np.arange(6))
        self.neigh_xy = r.isd_m * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        tc = self.cfg.traffic
        self.bounds = NormBounds(
            low=np.array([-10.0, 0.0, 0.0, -130.0, 0.0]),
            high=np.array([40.0, tc.buffer_cap_bits, 20e6, -80.0, 50e6]),
        )
        self._rng = np.random.default_rng(0)

    # ------------------------------------------------------------------ users
    def _alloc_arrays(self):
        U = self.u_max
        self.active = np.zeros(U, bool)
        self.slice_id = np.zeros(U, int)
        self.age = np.zeros(U, int)
        self.gain_serv = np.zeros(U)            # linear, incl. shadowing
        self.gain_neigh = np.zeros((U, 6))
        self.rate_mean = np.zeros(U)            # mean arrival rate (bit/s)
        self.queue = np.zeros(U)
        self.avg_rate = np.zeros(U)
        self.demand_est = np.zeros(U)
        self.fading = np.ones(U)
        self.last_n = np.zeros(U, int)
        self.last_p = np.zeros(U)
        W = self.cfg.sla.mmtc_window
        self.hist_off = np.zeros((U, W))
        self.hist_del = np.zeros((U, W))

    def _spawn(self, slot: int):
        cfg, r, rng = self.cfg, self.cfg.radio, self._rng
        # Best-server association: drop a user at a random point around the
        # target gNB and keep it only if the target cell is its strongest cell
        # (including shadowing), as it would be after cell selection.
        radius = 0.8 * r.isd_m
        while True:
            rr = radius * np.sqrt(rng.uniform())
            if rr < r.min_dist_m:
                continue
            th = rng.uniform(0, 2 * np.pi)
            xy = np.array([rr * np.cos(th), rr * np.sin(th)])
            d_all = np.concatenate([[rr], np.linalg.norm(self.neigh_xy - xy, axis=1)])
            d_all = np.maximum(d_all, r.min_dist_m)
            los = rng.uniform(size=7) < link.uma_los_probability(d_all)
            pl = link.uma_pathloss_db(d_all, los, r.carrier_ghz, r.bs_height_m, r.ue_height_m)
            sf = rng.normal(0.0, link.shadowing_std_db(los))
            g = link.dbm_to_mw(r.antenna_gain_db - pl - sf)
            if g[0] >= g[1:].max():
                break
        self.gain_serv[slot] = g[0]
        self.gain_neigh[slot] = g[1:]
        s = rng.choice(3, p=np.array(cfg.slice_mix) / np.sum(cfg.slice_mix))
        self.slice_id[slot] = s
        tc = cfg.traffic
        if s == EMBB:
            lam = rng.uniform(*tc.embb_rate_bps)
        elif s == URLLC:
            lam = rng.uniform(*tc.urllc_rate_bps)
        else:
            lam = tc.mmtc_burst_prob * np.mean(tc.mmtc_burst_bits) / self.t_c
        self.rate_mean[slot] = lam
        self.queue[slot] = 0.0
        self.avg_rate[slot] = lam          # optimistic start avoids PF division by zero
        self.demand_est[slot] = lam
        self.age[slot] = 0
        self.last_n[slot] = 0
        self.last_p[slot] = 0.0
        self.hist_off[slot] = 0.0
        self.hist_del[slot] = 0.0
        self.active[slot] = True

    def _churn(self):
        """Users leave and arrive (birth-death process around the target load)."""
        rng, hold = self._rng, self.cfg.traffic.mean_holding_intervals
        if not self.churn:
            return
        leave = self.active & (rng.uniform(size=self.u_max) < 1.0 / hold)
        self.active[leave] = False
        n_new = rng.poisson(self.u_target / hold)
        n_new = max(n_new, self.cfg.min_users - int(self.active.sum()))
        free = np.flatnonzero(~self.active)
        for slot in free[:max(n_new, 0)]:
            self._spawn(slot)

    # -------------------------------------------------------------- dynamics
    def _draw_interval_randomness(self):
        """Fading, neighbour load and arrivals for the coming interval.

        Drawn before the controller acts and independently of its action.
        """
        rng, r, tc = self._rng, self.cfg.radio, self.cfg.traffic
        k = r.fading_avg_samples
        self.fading = rng.gamma(k, 1.0 / k, size=self.u_max)
        lo, hi = r.neighbour_load
        self.neigh_load = np.clip(self.neigh_load + rng.normal(0, 0.03, 6), lo, hi)
        arr = np.zeros(self.u_max)
        for s, cv in ((EMBB, tc.embb_cv), (URLLC, tc.urllc_cv)):
            idx = self.active & (self.slice_id == s)
            mean = self.rate_mean[idx] * self.t_c
            shape = 1.0 / cv ** 2
            arr[idx] = rng.gamma(shape, mean / shape)
        idx = np.flatnonzero(self.active & (self.slice_id == MMTC))
        burst = rng.uniform(size=idx.size) < tc.mmtc_burst_prob
        arr[idx] = np.where(burst, rng.uniform(*tc.mmtc_burst_bits, size=idx.size), 0.0)
        self.arrivals = arr

    def _interference_mw(self) -> np.ndarray:
        return (self.gain_neigh * self.p_prb_nominal * self.neigh_load).sum(axis=1)

    def _observe(self) -> Observation:
        I = self._interference_mw()
        sinr_rep = (self.p_prb_nominal * self.gain_serv * self.fading) / (I + self.noise_mw)
        raw = np.zeros((self.u_max, N_REPORTED))
        raw[:, 0] = link.mw_to_dbm(sinr_rep)          # dB (ratio)
        raw[:, 1] = self.queue
        raw[:, 2] = self.demand_est
        raw[:, 3] = link.mw_to_dbm(I)
        raw[:, 4] = self.avg_rate
        raw[~self.active] = 0.0
        norm = self.bounds.normalize(raw)
        norm[~self.active] = 0.0
        return Observation(
            t=self.t, active=self.active.copy(), reported=raw, reported_norm=norm,
            last_prb_share=self.last_n / self.n_prb,
            last_power_share=self.last_p / self.p_max_mw,
            slice_id=self.slice_id.copy(),
        )

    # ------------------------------------------------------------------ API
    def reset(self, seed: Optional[int] = None, n_users: Optional[int] = None,
              churn: bool = True) -> Observation:
        """Start an episode.

        ``n_users`` fixes the mean load (and the starting number of users);
        ``churn=False`` keeps exactly that many users for the whole episode.
        """
        self._rng = np.random.default_rng(seed)
        self._alloc_arrays()
        lo, hi = self.cfg.radio.neighbour_load
        self.neigh_load = self._rng.uniform(lo, hi, size=6)
        if n_users is None:
            n_users = int(self._rng.integers(self.cfg.target_users[0], self.cfg.target_users[1] + 1))
        self.u_target = min(n_users, self.u_max)
        self.churn = churn
        for slot in range(self.u_target):
            self._spawn(slot)
        self.t = 0
        self._draw_interval_randomness()
        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, StepInfo, bool]:
        cfg, r, sla = self.cfg, self.cfg.radio, self.cfg.sla
        n, p = to_budgets(action.prb_share, action.power_share, self.active,
                          self.n_prb, self.p_max_mw)
        I = self._interference_mw()
        with np.errstate(divide="ignore", invalid="ignore"):
            p_per_prb = np.where(n > 0, p / np.maximum(n, 1), 0.0)
        sinr = p_per_prb * self.gain_serv * self.fading / (I + self.noise_mw)
        se = link.spectral_efficiency(sinr, r.se_alpha, r.sinr_min_db, r.se_max)
        cap_bits = n * r.prb_bw_hz * se * self.t_c

        offered = np.where(self.active, self.arrivals, 0.0)
        backlog = self.queue + offered
        delivered = np.minimum(cap_bits, backlog)
        q_new = backlog - delivered
        dropped = np.maximum(q_new - cfg.traffic.buffer_cap_bits, 0.0)
        q_new = q_new - dropped
        rate = delivered / self.t_c
        tw = cfg.avg_rate_window
        self.avg_rate = np.where(self.active, (1 - 1 / tw) * self.avg_rate + rate / tw, 0.0)
        self.demand_est = np.where(self.active, 0.9 * self.demand_est + 0.1 * offered / self.t_c, 0.0)
        self.queue = np.where(self.active, q_new, 0.0)
        self.hist_off = np.roll(self.hist_off, -1, axis=1)
        self.hist_del = np.roll(self.hist_del, -1, axis=1)
        self.hist_off[:, -1] = offered
        self.hist_del[:, -1] = delivered

        # SLA indicators (Sec. III-B of the report)
        # queueing-delay estimate q/R (report Sec. III-B); capped at 1 s, which
        # simply means "starved" (no service while data is waiting)
        delay = np.where(self.queue > 0,
                         np.minimum(self.queue / np.maximum(rate, sla.rate_floor_bps), 1.0), 0.0)
        warm = self.age >= 5
        v_embb = (self.slice_id == EMBB) & (self.avg_rate < sla.embb_min_rate_bps)
        v_urllc = (self.slice_id == URLLC) & (delay > sla.urllc_max_delay_s)
        off_w, del_w = self.hist_off.sum(1), self.hist_del.sum(1)
        ratio = np.where(off_w > 0, del_w / np.maximum(off_w, 1e-9), 1.0)
        v_mmtc = (self.slice_id == MMTC) & (ratio < sla.mmtc_min_delivery)
        violated = self.active & warm & (v_embb | v_urllc | v_mmtc)
        slice_v = np.full(N_SLICES, np.nan)
        for s in range(N_SLICES):
            m = self.active & warm & (self.slice_id == s)
            if m.any():
                slice_v[s] = violated[m].mean()

        act = self.active
        utility = float(np.mean(np.log1p(self.avg_rate[act] / cfg.r_ref_bps))) if act.any() else 0.0
        info = StepInfo(
            n_prb=n, power_mw=p, delivered_bits=delivered, offered_bits=offered,
            dropped_bits=dropped, rate_bps=rate, avg_rate_bps=self.avg_rate.copy(),
            true_sinr_db=link.mw_to_dbm(np.maximum(sinr, 1e-30)), delay_s=delay,
            violated=violated, active=act.copy(), slice_id=self.slice_id.copy(),
            utility=utility, power_frac=float(p.sum() / self.p_max_mw),
            prb_frac=float(n.sum() / self.n_prb), slice_violation=slice_v,
            extra={"n_active": float(act.sum()), "neigh_load": float(self.neigh_load.mean())},
        )
        self.last_n, self.last_p = n, p
        self.age = np.where(act, self.age + 1, 0)
        self.t += 1
        self._churn()
        self._draw_interval_randomness()
        done = self.t >= cfg.episode_len
        return self._observe(), info, done


def reward(info: StepInfo, lambdas: np.ndarray, zeta_p: float = 0.0,
           zeta_n: float = 0.0) -> float:
    """Per-step reward of Eq. (6) in the report, from ground-truth KPIs."""
    v = np.nan_to_num(info.slice_violation, nan=0.0)
    return info.utility - zeta_p * info.power_frac - zeta_n * info.prb_frac - float(np.dot(lambdas, v))
