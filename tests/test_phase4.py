"""Phase 4 (week 4): federation with two-tier trust, observer rules."""
import numpy as np
import torch

from rra.agents.policy import Actor, Critic
from rra.fed.aggregate import aggregate, data_trust, mmd2
from rra.fed.cells import CELLS, cell_env_config
from rra.fed.learner import CellLearner, LocalConfig, set_shared, shared_vector
from rra.fed.observer import ObserverState, ShiftRule, cell_summary, nearest


def _deltas(seed=0, C=8, P=50):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(P, generator=g)
    return [base + 0.3 * torch.randn(P, generator=g) for _ in range(C)]


def test_fedavg_and_median():
    d = _deltas()
    u, w, _ = aggregate("fedavg", d)
    assert torch.allclose(u, torch.stack(d).mean(0)) and np.allclose(w, 1 / 8)
    u, _, _ = aggregate("median", d)
    assert torch.allclose(u, torch.stack(d).median(0).values)


def test_two_tier_downweights_opposed_and_distrusted_cells():
    d = _deltas()
    d[0] = -5 * d[0]                                   # malicious: opposed and boosted
    trust = np.ones(8); trust[1] = 0.1                 # deceived: low data trust
    u, w, info = aggregate("two_tier", d, trust, np.ones(8))
    assert w[0] == 0 and w[1] < 0.5 * np.median(w[2:]) and abs(w.sum() - 1) < 1e-9
    ref = torch.stack(d).median(0).values
    assert u.norm() <= ref.norm() * (1 + 1e-5)          # every update was scaled to at most the reference norm
    _, w2, _ = aggregate("two_tier_no_data", d, trust, np.ones(8))
    assert w2[1] > w[1]                                 # without data trust the deceived cell counts fully


def test_fltrust_uses_root_direction_and_norm():
    d = _deltas()
    root = torch.stack(d).mean(0)
    d[3] = -d[3]
    u, w, info = aggregate("fltrust", d, root=root)
    assert w[3] == 0 and abs(w.sum() - 1) < 1e-9 and u.norm() <= root.norm() * (1 + 1e-5)


def test_mmd_and_data_trust():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(size=(300, 5)), rng.uniform(size=(300, 5))
    z = np.clip(y + 0.15, 0, 1)
    assert mmd2(x, y) < mmd2(x, z)
    assert data_trust(0.9, 0.0, 2.0) == 0.9 and data_trust(0.9, 1.0, 2.0) < 0.9


def test_shared_vector_roundtrip_touches_only_shared_layers():
    a = Actor(prior="need")
    v = shared_vector(a)
    head_before = [p.clone() for p in a.head.parameters()]
    set_shared(a, v + 1.0)
    assert torch.allclose(shared_vector(a), v + 1.0)
    assert all(torch.equal(p, q) for p, q in zip(head_before, a.head.parameters()))
    assert v.numel() * 4 < 80 * 1024                    # report guard: under 80 KB per cell per round


def test_cell_learner_round_and_malicious_cell():
    cfg = LocalConfig(n_envs=2, n_steps=16, epochs=1, minibatch=16)
    a, c = Actor(prior="need"), Critic()
    L = CellLearner(0, CELLS[1], cfg, a, c, np.full(3, 0.5), None, seed=1)
    r = L.local_round(1)
    assert r["delta"].shape == shared_vector(a).shape and r["reports"].shape[1] == 5 and r["steps"] == 32
    assert abs(r["rho_bar"] - 1.0) < 1e-6               # no sanitizer: every user fully trusted
    Lm = CellLearner(1, CELLS[1], cfg, a, c, np.full(3, 0.5), None, seed=1, malicious=True)
    before = shared_vector(Lm.actor)
    rm = Lm.local_round(1)
    assert torch.equal(shared_vector(Lm.actor), before)  # its own deployed model is untouched
    assert rm["delta"].norm() > 0
    assert cell_env_config(CELLS[2]).slice_mix == CELLS[2].slice_mix


def test_shift_rule_and_observer():
    rule = ShiftRule(f_attack=0.2, f_odd=0.06, d_change=0.15)
    assert rule.decide(0.01, 0.05, 0.5) == "shift"          # moved, few odd users, matches another cell
    assert rule.decide(0.01, 0.60, 0.5) == "unclear"        # moved, but like no other cell
    assert rule.decide(0.5, 0.05, 0.5) == "attack"          # many users distrusted
    assert rule.decide(0.10, 0.50, 0.05) == "attack"        # more odd users than clean windows show
    assert rule.decide(0.01, 0.40, 0.05) == "no change"
    k = cell_summary(np.full((10, 5), 0.5), np.array([0] * 5 + [1] * 5), np.ones(10), 10)
    assert k.shape == (9,) and np.isclose(k[1:4].sum(), 1)
    assert nearest(k, {"a": k + 1, "b": k + 0.1})[0] == "b"
    o = ObserverState()
    assert not o.update(1.0) and not o.update(0.95) and o.update(0.5)
