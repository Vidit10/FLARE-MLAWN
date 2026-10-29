"""Phase-1 checks: Dirichlet maths, equivariance, feasibility of PPO actions."""
import numpy as np
import torch

from rra.agents.policy import (Actor, dir_entropy, dir_kl, dir_log_prob,
                               dir_sample, obs_features)
from rra.agents.ppo import PPOController
from rra.env.pysim import PySimEnv

torch.manual_seed(0)


def _masked_case():
    alpha = torch.rand(3, 7) * 3 + 0.2
    cmask = torch.tensor([[1, 1, 1, 0, 0, 1, 1], [1] * 7, [0, 1, 0, 1, 0, 0, 1]], dtype=torch.bool)
    return alpha, cmask


def test_dirichlet_matches_torch():
    alpha, cmask = _masked_case()
    x = dir_sample(alpha, cmask)
    assert torch.allclose(x.sum(-1), torch.ones(3)) and torch.all(x[~cmask] == 0)
    beta = torch.rand(3, 7) * 3 + 0.2
    for b in range(3):
        a, xb, bb = alpha[b][cmask[b]], x[b][cmask[b]], beta[b][cmask[b]]
        ref = torch.distributions.Dirichlet(a)
        assert torch.allclose(dir_log_prob(alpha[b:b+1], x[b:b+1], cmask[b:b+1])[0], ref.log_prob(xb), atol=1e-4)
        assert torch.allclose(dir_entropy(alpha[b:b+1], cmask[b:b+1])[0], ref.entropy(), atol=1e-4)
        kl_ref = torch.distributions.kl_divergence(ref, torch.distributions.Dirichlet(bb))
        assert torch.allclose(dir_kl(alpha[b:b+1], beta[b:b+1], cmask[b:b+1])[0], kl_ref, atol=1e-4)


def test_actor_is_permutation_equivariant():
    env = PySimEnv()
    obs = env.reset(seed=5, n_users=12)
    x, m = obs_features(obs)
    actor = Actor()
    for p in actor.parameters():           # random weights, not the zero init
        torch.nn.init.normal_(p, std=0.3)
    perm = np.random.default_rng(0).permutation(env.u_max)
    o1 = actor(torch.from_numpy(x)[None], torch.from_numpy(m)[None])
    o2 = actor(torch.from_numpy(x[perm])[None], torch.from_numpy(m[perm])[None])
    for a, b in ((o1.alpha_prb, o2.alpha_prb), (o1.alpha_pow, o2.alpha_pow)):
        assert torch.allclose(a[0, :-1][perm], b[0, :-1], rtol=1e-4, atol=1e-5)
        assert torch.allclose(a[0, -1], b[0, -1], rtol=1e-4)


def test_ppo_actions_feasible():
    env = PySimEnv()
    ctrl = PPOController(Actor(), stochastic=True)
    obs = env.reset(seed=1, n_users=25)
    for _ in range(20):
        a = ctrl.act(obs)
        assert a.prb_share.sum() <= 1 + 1e-5 and a.power_share.sum() <= 1 + 1e-5
        assert np.all(a.prb_share[~obs.active] == 0)
        obs, info, _ = env.step(a)
        assert info.n_prb.sum() <= env.n_prb


def _random_actor():
    torch.manual_seed(1)
    actor = Actor()
    for p in actor.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return actor


def test_gradient_attack_budget_and_mask():
    from rra.attacks import GradientAttacker
    env = PySimEnv()
    ctrl = PPOController(_random_actor())
    obs = env.reset(seed=3, n_users=15)
    for strat, objective, point in (("fgsm", "kl", "A2"), ("pgd", "kl", "A1"), ("pgd", "util", "A2")):
        att = GradientAttacker(eps=0.05, point=point, k=4, strategy=strat, objective=objective, steps=5)
        att.reset(env, ctrl, seed=0)
        adv = att.perturb(obs)
        d = adv.reported_norm - obs.reported_norm
        assert np.abs(d).max() <= 0.05 + 1e-6
        assert np.all(d[~att.entry_mask(obs)] == 0)
        assert np.all(adv.reported_norm >= 0) and np.all(adv.reported_norm <= 1)


def test_pgd_beats_random_noise_on_kl():
    from rra.attacks import Attacker, GradientAttacker
    env = PySimEnv()
    ctrl = PPOController(_random_actor())
    obs = env.reset(seed=4, n_users=20)
    x0, m = obs_features(obs)
    clean = ctrl.actor(torch.from_numpy(x0)[None], torch.from_numpy(m)[None])

    def kl_of(o):
        x, mm = obs_features(o)
        out = ctrl.actor(torch.from_numpy(x)[None], torch.from_numpy(mm)[None])
        return float(dir_kl(clean.alpha_prb, out.alpha_prb, out.cmask) + dir_kl(clean.alpha_pow, out.alpha_pow, out.cmask))

    pgd = GradientAttacker(eps=0.05, strategy="pgd", objective="kl"); pgd.reset(env, ctrl, seed=0)
    noise = Attacker(eps=0.05, strategy="noise"); noise.reset(env, ctrl, seed=0)
    with torch.no_grad():
        assert kl_of(pgd.perturb(obs)) > 2 * kl_of(noise.perturb(obs))


def test_need_feature_matches_numpy_denormalisation():
    from rra.agents.policy import NeedFeature
    env = PySimEnv()
    obs = env.reset(seed=6, n_users=15)
    nf = NeedFeature()
    raw_t = nf.raw(torch.tensor(obs.reported_norm, dtype=torch.float32)).numpy()
    raw_np = env.bounds.denormalize(obs.reported_norm)
    a = obs.active
    assert np.allclose(raw_t[a], raw_np[a], rtol=1e-4, atol=1e-2)


def test_untrained_prior_policy_gives_needed_prbs():
    env = PySimEnv()
    obs = env.reset(seed=2, n_users=8, churn=False)      # light load: total need < N
    actor = Actor(prior="need")
    x, m = obs_features(obs)
    out = actor(torch.from_numpy(x)[None], torch.from_numpy(m)[None])
    prb, _ = out.mean()
    need = actor.need.need_prbs(torch.from_numpy(x[:, :5])).numpy() * m
    assert need.sum() < env.n_prb
    # share_u = (need_u + 0.5) / (N + 0.5 (U + 1)) when total need < N
    expected = (need[m] + 0.5) / (env.n_prb + 0.5 * (m.sum() + 1))
    assert np.allclose(prb[0, :-1].detach().numpy()[m], expected, rtol=1e-3)


def test_prior_actor_is_equivariant():
    env = PySimEnv()
    obs = env.reset(seed=9, n_users=18)
    x, m = obs_features(obs)
    actor = Actor(prior="need")
    for p in actor.parameters():
        torch.nn.init.normal_(p, std=0.3)
    perm = np.random.default_rng(1).permutation(env.u_max)
    o1 = actor(torch.from_numpy(x)[None], torch.from_numpy(m)[None])
    o2 = actor(torch.from_numpy(x[perm])[None], torch.from_numpy(m[perm])[None])
    assert torch.allclose(o1.alpha_prb[0, :-1][perm], o2.alpha_prb[0, :-1], rtol=1e-4, atol=1e-5)
