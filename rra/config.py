"""Scenario parameters (Table III of the mid-semester report), in one place.

All values are planned defaults and can be overridden per experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RadioConfig:
    carrier_ghz: float = 3.5
    bandwidth_mhz: float = 40.0
    n_prb: int = 106                  # 3GPP: 40 MHz at 30 kHz SCS
    prb_bw_hz: float = 12 * 30e3      # 360 kHz
    p_max_dbm: float = 43.0
    bs_height_m: float = 25.0
    ue_height_m: float = 1.5
    isd_m: float = 500.0              # inter-site distance (hexagonal layout)
    min_dist_m: float = 35.0          # UMa minimum 2D distance
    noise_psd_dbm_hz: float = -174.0
    ue_noise_figure_db: float = 7.0
    antenna_gain_db: float = 0.0      # omni, no sectorisation (kept simple)
    # link model: attenuated, truncated Shannon (3GPP TR 36.942 Annex A style)
    se_alpha: float = 0.6
    sinr_min_db: float = -10.0
    se_max: float = 7.4               # ~256-QAM peak spectral efficiency (bit/s/Hz)
    # fast fading averaged over one control interval: Gamma(k, 1/k) power gain
    fading_avg_samples: int = 10
    # neighbour cells: fraction of their PRBs busy, drifting in this range
    neighbour_load: Tuple[float, float] = (0.3, 0.8)


@dataclass
class TrafficConfig:
    # eMBB: high-rate flows (bit/s), coefficient of variation of per-interval volume
    embb_rate_bps: Tuple[float, float] = (6e6, 15e6)
    embb_cv: float = 0.5
    # URLLC: small periodic packets, low rate, strict delay
    urllc_rate_bps: Tuple[float, float] = (0.2e6, 1.0e6)
    urllc_cv: float = 0.3
    # mMTC: sporadic bursts
    mmtc_burst_prob: float = 0.2
    mmtc_burst_bits: Tuple[float, float] = (8e3, 80e3)
    buffer_cap_bits: float = 16e6     # per-user buffer (2 MB); overflow is dropped
    # user arrivals/departures (on/off) -> varying number of active users
    mean_holding_intervals: float = 150.0


@dataclass
class SLAConfig:
    embb_min_rate_bps: float = 5e6
    urllc_max_delay_s: float = 10e-3
    mmtc_min_delivery: float = 0.99
    mmtc_window: int = 50             # intervals used for the delivery ratio
    rate_floor_bps: float = 1e3       # avoids division by zero in delay estimate
    tolerated: Tuple[float, float, float] = (0.05, 0.01, 0.05)  # nu_l per slice


@dataclass
class EnvConfig:
    radio: RadioConfig = field(default_factory=RadioConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    sla: SLAConfig = field(default_factory=SLAConfig)
    t_c: float = 0.1                  # control interval (s)
    u_max: int = 40                   # slots for users (max active)
    target_users: Tuple[int, int] = (5, 30)   # mean load drawn per episode
    min_users: int = 3
    slice_mix: Tuple[float, float, float] = (0.4, 0.3, 0.3)  # eMBB, URLLC, mMTC
    episode_len: int = 300
    avg_rate_window: float = 20.0     # t_w for the moving-average rate
    r_ref_bps: float = 1e6            # reference rate in the utility
    # Scaling of reported features to [0, 1] (Sec. III-C of the report).
    # "log": buffer, demand and average rate on a log scale (proposed in week 1,
    # so that epsilon is a relative change for every slice); "linear": the
    # min-max scaling written in the mid-semester report.
    feature_scale: str = "log"
