"""Phase-0 checks: the plumbing must be right before any learning is added."""
import numpy as np
import pytest

from rra.attacks import Attacker, eps_in_physical_units
from rra.baselines import EqualShare, ProportionalFair
from rra.env import link
from rra.env.interface import Action, to_budgets
from rra.env.pysim import PySimEnv
from rra.runner import run_episode


@pytest.fixture
def env():
    return PySimEnv()


def test_to_budgets_feasible():
    rng = np.random.default_rng(0)
    for _ in range(200):
        act = rng.uniform(size=40) < 0.5
        x, y = rng.uniform(size=40) * 3, rng.uniform(size=40)
        n, p = to_budgets(x, y, act, 106, 20e3)
        assert n.dtype.kind == "i" and n.min() >= 0
        assert n.sum() <= 106 and p.sum() <= 20e3 + 1e-6
        assert np.all(n[~act] == 0) and np.all(p[n == 0] == 0)


def test_largest_remainder_uses_full_share():
    act = np.ones(3, bool)
    n, _ = to_budgets(np.array([1 / 3] * 3), np.zeros(3), act, 106, 1.0)
    assert n.sum() == 106 and n.max() - n.min() <= 1


def test_pathloss_sane():
    pl = link.uma_pathloss_db(np.array([100.0, 500.0]), np.array([True, False]), 3.5, 25, 1.5)
    # TR 38.901 UMa at 3.5 GHz: roughly 80-90 dB LOS at 100 m, ~130 dB NLOS at 500 m
    assert 75 < pl[0] < 95 and 120 < pl[1] < 140


def test_normalisation_roundtrip(env):
    obs = env.reset(seed=3, n_users=10)
    a = obs.active
    back = env.bounds.denormalize(obs.reported_norm)
    inside = (obs.reported >= env.bounds.low) & (obs.reported <= env.bounds.high)
    assert np.allclose(back[a][inside[a]], obs.reported[a][inside[a]])


def test_randomness_independent_of_actions(env):
    """Same seed => same users, channels and traffic whatever the controller does."""
    offered = []
    for ctrl in (EqualShare(env), ProportionalFair(env)):
        obs = env.reset(seed=7, n_users=15)
        seq = []
        for _ in range(40):
            obs, info, _ = env.step(ctrl.act(obs))
            seq.append(info.offered_bits.copy())
        offered.append(np.array(seq))
    assert np.allclose(offered[0], offered[1])


def test_conservation(env):
    obs = env.reset(seed=1, n_users=20)
    ctrl = ProportionalFair(env)
    for _ in range(50):
        q0 = env.queue.copy()
        obs, info, _ = env.step(ctrl.act(obs))
        a = info.active & (env.age > 1)            # users present before and after
        lhs = q0[a] + info.offered_bits[a]
        rhs = env.queue[a] + info.delivered_bits[a] + info.dropped_bits[a]
        assert np.allclose(lhs, rhs)


def test_pf_respects_budgets(env):
    obs = env.reset(seed=2, n_users=30)
    act = ProportionalFair(env).act(obs)
    assert act.prb_share.sum() <= 1 + 1e-9 and act.power_share.sum() <= 1 + 1e-9


@pytest.mark.parametrize("point,strategy", [("A1", "direction"), ("A2", "noise"), ("A2", "search")])
def test_attack_budget_and_mask(env, point, strategy):
    obs = env.reset(seed=4, n_users=20)
    att = Attacker(eps=0.05, point=point, k=3, strategy=strategy, n_candidates=4)
    att.reset(env, ProportionalFair(env), seed=0)
    adv = att.perturb(obs)
    d = adv.reported_norm - obs.reported_norm
    assert np.abs(d).max() <= 0.05 + 1e-9
    assert np.all(d[~att.entry_mask(obs)] == 0)              # untouched elsewhere
    assert att.controlled_users(obs).sum() == 3
    if point == "A1":
        assert np.all(d[:, [1, 2, 4]] == 0)                   # UE cannot touch gNB counters
    # trusted fields never change
    assert np.array_equal(adv.last_prb_share, obs.last_prb_share)
    assert np.array_equal(adv.slice_id, obs.slice_id)


def test_a1_cannot_touch_buffer():
    with pytest.raises(ValueError):
        Attacker(eps=0.05, point="A1", features=["buffer_bits"])


def test_eps_physical():
    phys = eps_in_physical_units(PySimEnv().bounds, 0.05)
    assert phys["sinr_db"] == "+/-2.5 dB"
    assert phys["buffer_bits"].startswith("x1.6")      # log scale: ~x1.62


@pytest.mark.parametrize("scale", ["log", "linear"])
def test_scaling_roundtrip_both_modes(scale):
    from rra.config import EnvConfig
    env = PySimEnv(EnvConfig(feature_scale=scale))
    x = np.array([[3.0, 5e4, 2e6, -95.0, 8e6], [-9.0, 0.0, 0.0, -129.0, 1e3]])
    assert np.allclose(env.bounds.denormalize(env.bounds.normalize(x)), x, rtol=1e-9, atol=1e-6)


def test_episode_runs(env):
    s = run_episode(env, ProportionalFair(env), seed=0, n_users=10, steps=60)
    assert 0 <= s["viol_eMBB"] <= 1 and s["cell_thr_mbps"] > 0
