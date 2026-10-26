"""Radio helpers: 3GPP TR 38.901 urban-macro path loss and the link model."""
from __future__ import annotations

import numpy as np

C_LIGHT = 3e8


def uma_los_probability(d2d: np.ndarray) -> np.ndarray:
    """TR 38.901 Table 7.4.2-1, UMa, for UE height <= 13 m (C'(h_UT) = 0)."""
    d = np.maximum(d2d, 1e-3)
    p = 18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d)
    return np.where(d <= 18.0, 1.0, np.clip(p, 0.0, 1.0))


def uma_pathloss_db(d2d: np.ndarray, los: np.ndarray, fc_ghz: float,
                    h_bs: float, h_ut: float) -> np.ndarray:
    """TR 38.901 Table 7.4.1-1, UMa path loss in dB (no shadowing).

    LOS uses the two-slope model with breakpoint d'_BP = 4 h'_BS h'_UT f_c / c
    (effective heights h' = h - 1 m).  NLOS is max(PL_LOS, PL'_NLOS).
    """
    d3d = np.sqrt(d2d ** 2 + (h_bs - h_ut) ** 2)
    fc_hz = fc_ghz * 1e9
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0) * fc_hz / C_LIGHT
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.0 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos_prime = (13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
                     - 0.6 * (h_ut - 1.5))
    pl_nlos = np.maximum(pl_los, pl_nlos_prime)
    return np.where(los, pl_los, pl_nlos)


def shadowing_std_db(los: np.ndarray) -> np.ndarray:
    """UMa shadow-fading standard deviation: 4 dB LOS, 6 dB NLOS."""
    return np.where(los, 4.0, 6.0)


def spectral_efficiency(sinr_lin: np.ndarray, alpha: float, sinr_min_db: float,
                        se_max: float) -> np.ndarray:
    """Attenuated, truncated Shannon bound (bit/s/Hz)."""
    sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-12))
    se = np.minimum(se_max, alpha * np.log2(1.0 + sinr_lin))
    return np.where(sinr_db < sinr_min_db, 0.0, se)


def dbm_to_mw(x_dbm):
    return 10.0 ** (np.asarray(x_dbm) / 10.0)


def mw_to_dbm(x_mw):
    return 10.0 * np.log10(np.maximum(np.asarray(x_mw), 1e-30))
