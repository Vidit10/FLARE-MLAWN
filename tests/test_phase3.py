"""Phase 3 (week 3): detector, sanitizer, defended controller, adaptive attacks."""
import numpy as np
import torch

from rra.agents.policy import Actor, obs_features
from rra.attacks import GradientAttacker
from rra.defence.controller import DefendedController
from rra.defence.robust import batch_attack, mean_kl
from rra.defence.sanitizer import Sanitizer, SanitizerConfig, conformal_threshold
from rra.env.pysim import PySimEnv


def _san(tau=1e-3):
    torch.manual_seed(0)
    s = Sanitizer(SanitizerConfig())
    s.tau.fill_(tau)
    s.eval()
    for p in s.parameters():
        p.requires_grad_(False)
    return s


def test_conformal_threshold_controls_false_positives():
    rng = np.random.default_rng(0)
    cal, test = rng.exponential(size=2000), rng.exponential(size=200_000)
    tau = conformal_threshold(cal, 0.01)
    assert np.mean(test > tau) < 0.015


def test_sanitizer_shapes_masks_and_modes():
    san = _san()
    s = torch.rand(2, 7, 5); oh = torch.eye(3)[torch.randint(0, 3, (2, 7))]
    m = torch.ones(2, 7, dtype=torch.bool); m[1, 4:] = False
    st, rho, e = san(s, oh, m, mode="rand")
    assert st.shape == s.shape and rho.shape == (2, 7)
    assert torch.all(st[1, 4:] == 0) and torch.all(rho[1, 4:] == 0)
    assert torch.all((rho >= 0) & (rho <= 1))
    st0, rho0, _ = san(s, oh, m, mode="none")
    assert torch.equal(st0, s) and torch.equal(rho0, m.float())
    _, rho1, _ = san(s, oh, m, mode="det", gate="always")
    assert torch.all(rho1 == 0)


def test_sanitizer_is_per_user_equivariant_for_any_U():
    san = _san()
    for U in (3, 11, 40):
        s = torch.rand(1, U, 5); oh = torch.eye(3)[torch.randint(0, 3, (1, U))]
        m = torch.ones(1, U, dtype=torch.bool)
        perm = torch.randperm(U)
        xi = torch.randn(san.cfg.n_heads, 1, U, san.cfg.latent) * san.cfg.sigma
        a, ra, _ = san(s, oh, m, "rand", xi=xi)
        b, rb, _ = san(s[:, perm], oh[:, perm], m, "rand", xi=xi[:, :, perm])
        assert torch.allclose(a[:, perm], b, atol=1e-6) and torch.allclose(ra[:, perm], rb, atol=1e-6)


def test_randomized_sanitizer_is_random_and_deterministic_one_is_not():
    san = _san(tau=0.0)               # everything flagged -> output = denoiser
    s = torch.rand(1, 5, 5); oh = torch.eye(3)[torch.zeros(1, 5, dtype=torch.long)]
    m = torch.ones(1, 5, dtype=torch.bool)
    r1, r2 = san(s, oh, m, "rand")[0], san(s, oh, m, "rand")[0]
    d1, d2 = san(s, oh, m, "det")[0], san(s, oh, m, "det")[0]
    assert not torch.allclose(r1, r2) and torch.equal(d1, d2)


def test_defended_controller_feasible_and_attack_respects_budget():
    env = PySimEnv()
    ctrl = DefendedController(Actor(prior="need"), _san(), mode="rand")
    obs = env.reset(seed=3, n_users=12)
    a = ctrl.act(obs)
    assert a.prb_share.sum() <= 1 + 1e-6 and a.power_share.sum() <= 1 + 1e-6
    from rra.env.interface import to_budgets
    n, pw = to_budgets(a.prb_share, a.power_share, obs.active, env.n_prb, env.p_max_mw)
    assert n.sum() <= env.n_prb and np.all(n[~obs.active] == 0) and np.all(pw[~obs.active] == 0)
    for kw in ({"objective": "kl", "eot": 4}, {"objective": "util", "eot": 2, "stealth": True},
               {"objective": "util", "adaptive": False}):
        att = GradientAttacker(eps=0.05, steps=3, **kw)
        att.reset(env, ctrl, seed=1)
        o2 = att.perturb(obs)
        d = o2.reported_norm - obs.reported_norm
        assert np.abs(d).max() <= 0.05 + 1e-6
        assert np.all(d[~obs.active] == 0)


def test_stealth_reduces_flagged_users():
    env = PySimEnv()
    san = _san()
    obs = env.reset(seed=4, n_users=15)
    x, _ = obs_features(obs)
    oh = torch.from_numpy(x[:, 8:11])
    san.tau.fill_(float(np.quantile(san.error(torch.from_numpy(x[:, :5]), oh)[obs.active].numpy(), 0.99)))
    ctrl = DefendedController(Actor(prior="need"), san, mode="rand")
    flagged = []
    for st in (False, True):
        att = GradientAttacker(eps=0.1, steps=3, objective="util", stealth=st)
        att.reset(env, ctrl, seed=2)
        o2 = att.perturb(obs)
        e = san.error(torch.from_numpy(o2.reported_norm.astype(np.float32)), oh)
        flagged.append(int((e > san.tau)[torch.from_numpy(obs.active)].sum()))
    assert flagged[1] <= flagged[0]


def test_batch_attack_budget_and_mean_kl():
    env = PySimEnv()
    actor, san = Actor(prior="need"), _san()
    obs = [env.reset(seed=s, n_users=8 + s) for s in range(3)]
    feats = [obs_features(o) for o in obs]
    x0 = torch.from_numpy(np.stack([f[0] for f in feats])); m = torch.from_numpy(np.stack([f[1] for f in feats]))
    d = batch_attack(actor, san, "rand", "soft", x0, m, 0.1, 3, 2, ["kl", "util", "kl"], env, obs)
    assert d.abs().max() <= 0.1 + 1e-6 and torch.all(d[~m] == 0)
    assert all(p.requires_grad for p in actor.parameters())
    out = actor(x0, m)
    assert torch.allclose(mean_kl(out, out), torch.zeros(3), atol=1e-6)
    x1 = x0.clone(); x1[..., :5] = (x1[..., :5] + d).clamp(0, 1)
    assert torch.all(mean_kl(out, actor(x1, m)) >= -1e-6)


def test_learned_attacks_respect_budget_and_masks():
    from rra.attacks import GeneratorAttacker, TransferAttacker, UAPAttacker
    from rra.attacks.learned import Generator
    env = PySimEnv()
    ctrl = DefendedController(Actor(prior="need"), _san(), mode="rand")
    sur = DefendedController(Actor(prior="need"), None, mode="none")
    obs = env.reset(seed=5, n_users=9)
    torch.manual_seed(0)
    table = np.random.default_rng(0).uniform(-0.3, 0.3, size=(3, 5))
    for att in (UAPAttacker(eps=0.05, table=table), GeneratorAttacker(eps=0.05, gen=Generator(0.05)),
                TransferAttacker(eps=0.05, steps=2, objective="util", surrogate=sur)):
        att.reset(env, ctrl, seed=1)
        d = att.perturb(obs).reported_norm - obs.reported_norm
        assert np.abs(d).max() <= 0.05 + 1e-6 and np.all(d[~obs.active] == 0)
    # UAP: same perturbation for every user of a slice (where the [0, 1] box does not bind)
    att = UAPAttacker(eps=0.05, table=np.clip(table, -0.05, 0.05)); att.reset(env, ctrl, seed=1)
    d = att.perturb(obs).reported_norm - obs.reported_norm
    inside = (obs.reported_norm > 0.05) & (obs.reported_norm < 0.95)
    for k in range(3):
        rows = obs.active & (obs.slice_id == k)
        for f in range(5):
            v = d[rows & inside[:, f], f]
            assert v.size == 0 or np.allclose(v, v[0], atol=1e-6)


def test_generator_is_per_user_equivariant():
    from rra.attacks.learned import Generator
    torch.manual_seed(0)
    g = Generator(0.1)
    s = torch.rand(1, 12, 5); oh = torch.eye(3)[torch.randint(0, 3, (1, 12))]
    m = torch.ones(1, 12, dtype=torch.bool); m[0, 9:] = False
    perm = torch.randperm(12)
    a, b = g(s, oh, m), g(s[:, perm], oh[:, perm], m[:, perm])
    assert torch.allclose(a[:, perm], b, atol=1e-6) and torch.all(a[0, 9:] == 0) and a.abs().max() <= 0.1


def test_fitted_attacks_lower_predicted_utility():
    from rra.attacks import collect_states, fit_generator, fit_uap
    from rra.attacks.learned import _pred_util, _stack
    env = PySimEnv()
    ctrl = DefendedController(Actor(prior="need"), None, mode="none")
    states = collect_states(env, ctrl, n_episodes=1, users=(10,), every=10)
    x0, m = _stack(states)
    sid = torch.from_numpy(np.stack([o.slice_id for o in states])).long()
    mask = m.unsqueeze(-1).float()
    with torch.no_grad():
        u0 = _pred_util(env, ctrl, states, x0[..., :5], x0, m, 1)[0]
        tab = torch.from_numpy(fit_uap(env, ctrl, states, 0.1, iters=30, batch=8, eot=1)).float()
        u1 = _pred_util(env, ctrl, states, (x0[..., :5] + tab[sid] * mask).clamp(0, 1), x0, m, 1)[0]
        g = fit_generator(env, ctrl, states, 0.1, iters=60, batch=8, eot=1, lr=3e-3)
        u2 = _pred_util(env, ctrl, states, (x0[..., :5] + g(x0[..., :5], x0[..., 8:11], m)).clamp(0, 1), x0, m, 1)[0]
    assert u1 < u0 and u2 < u0
