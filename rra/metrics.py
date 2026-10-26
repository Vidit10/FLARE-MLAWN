"""Episode metrics, computed only from ground truth (StepInfo).

The names follow the metric tiers of the report:
* objective terms: utility Phi, power use P, PRB use B, slice violation rates;
* guards: Jain fairness and 5th-percentile rate (eMBB slice), URLLC p99 delay;
* diagnostics: cell throughput, delivered/offered ratio, dropped traffic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .env.interface import N_SLICES, SLICES, StepInfo


def jain(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.size == 0 or np.all(x == 0):
        return float("nan")
    return float(x.sum() ** 2 / (x.size * (x ** 2).sum()))


@dataclass
class EpisodeLog:
    """Collects StepInfo objects and summarises them; skips a warm-up period."""
    warmup: int = 20
    infos: List[StepInfo] = field(default_factory=list)

    def add(self, info: StepInfo) -> None:
        self.infos.append(info)

    def summary(self) -> Dict[str, float]:
        I = self.infos[self.warmup:]
        if not I:
            raise ValueError("episode shorter than warm-up")
        out: Dict[str, float] = {}
        out["utility"] = float(np.mean([i.utility for i in I]))
        out["power_frac"] = float(np.mean([i.power_frac for i in I]))
        out["prb_frac"] = float(np.mean([i.prb_frac for i in I]))
        sv = np.array([i.slice_violation for i in I])          # (T, L)
        for l, name in enumerate(SLICES):
            col = sv[:, l]
            out[f"viol_{name}"] = float(np.nanmean(col)) if np.isfinite(col).any() else float("nan")
        # Users: rates averaged over the time each user is active
        thr = np.array([i.rate_bps[i.active].sum() for i in I])
        out["cell_thr_mbps"] = float(thr.mean() / 1e6)
        off = sum(i.offered_bits[i.active].sum() for i in I)
        dlv = sum(i.delivered_bits[i.active].sum() for i in I)
        drp = sum(i.dropped_bits[i.active].sum() for i in I)
        out["delivered_ratio"] = float(dlv / off) if off > 0 else float("nan")
        out["dropped_frac"] = float(drp / off) if off > 0 else 0.0
        # Fairness and cell-edge rate are computed inside the eMBB slice: mixing
        # slices with very different demands (Mbit/s vs kbit/s) would make them
        # meaningless.
        emb = [i.avg_rate_bps[i.active & (i.slice_id == 0)] for i in I]
        emb = [x for x in emb if x.size >= 2]
        out["jain_eMBB"] = float(np.mean([jain(x) for x in emb])) if emb else float("nan")
        out["p5_rate_eMBB_mbps"] = float(np.mean([np.percentile(x, 5) for x in emb]) / 1e6) if emb else float("nan")
        urllc_d = np.concatenate([i.delay_s[i.active & (i.slice_id == 1)] for i in I])
        out["urllc_p99_delay_ms"] = float(np.percentile(urllc_d, 99) * 1e3) if urllc_d.size else float("nan")
        out["any_violation"] = float(np.mean([i.violated[i.active].mean() for i in I if i.active.any()]))
        out["n_active"] = float(np.mean([i.active.sum() for i in I]))
        return out


def recovery_ratio(attacked: float, clean: float, floor: float) -> float:
    """Fraction of the clean-to-floor gap that survives under attack.

    1 = no loss, 0 = as bad as the reference floor (e.g. a simple heuristic).
    Defined in the report as the worst-case recovery ratio when taken as the
    minimum over attacks.
    """
    gap = clean - floor
    return float((attacked - floor) / gap) if abs(gap) > 1e-12 else float("nan")
