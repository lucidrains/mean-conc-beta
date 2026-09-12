from __future__ import annotations
import math
import pytest
param = pytest.mark.parametrize
from pytest import raises

import torch
from torch import tensor
import gymnasium as gym

from mean_conc_beta import Beta

# e2e - forward, sample, rsample, log prob, entropy, mode, backward

@param('pos_fn', ['exp', 'softplus'])
@param('val_range', [(-1., 1.), (0., 1.)])
def test_end_to_end(pos_fn, val_range):
    beta = Beta(pos_fn = pos_fn, val_range = val_range)
    params = torch.randn(8, 4, 2, requires_grad = True)

    dist = beta(params)
    low, high = val_range

    actions = dist.sample()
    assert actions.shape == (8, 4)
    assert ((actions >= low) & (actions <= high)).all()

    assert beta.log_prob(dist, actions).shape == (8,)
    assert beta.entropy(dist).shape == (8,)
    assert beta.mode(params).shape == (8, 4)
    assert (dist.variance >= 0.).all()

    dist.rsample().sum().backward()
    assert torch.isfinite(params.grad).all()

# mean is preserved exactly, regardless of concentration

@param('val_range', [(-1., 1.), (0., 1.)])
def test_mean_preservation(val_range):
    beta = Beta(val_range = val_range)
    low, high = val_range

    unit_means = torch.linspace(0.1, 0.9, 5)
    params = torch.stack([torch.atanh(unit_means * 2. - 1.), torch.zeros(5)], dim = -1)

    expected = unit_means * (high - low) + low

    assert torch.allclose(beta.mean(params), expected, atol = 1e-5)
    assert torch.allclose(beta(params).mean, expected, atol = 1e-5)

# unbounded floor forces alpha, beta > 1, default damping avoids U shapes

@param('pos_fn', ['exp', 'softplus'])
@param('val_range', [(-1., 1.), (0., 1.)])
def test_unimodality(pos_fn, val_range):
    params = tensor([[[-100., 0.], [100., 0.]], [[0., -10.], [0., 10.]]])

    unbounded = Beta(pos_fn = pos_fn, val_range = val_range, max_unimodal_floor = None)(params).base_dist
    assert (unbounded.concentration1 > 1.).all()
    assert (unbounded.concentration0 > 1.).all()

    damped = Beta(pos_fn = pos_fn, val_range = val_range)(params).base_dist
    assert (torch.maximum(damped.concentration1, damped.concentration0) > 1.).all()

# detaching the unimodal floor keeps boundary gradients bounded

def test_detach_unimodal():
    def raw_mean_grad(detach):
        raw_mean = tensor([5.], requires_grad = True)
        beta = Beta(detach_unimodal = detach, max_unimodal_floor = None)
        params = torch.stack([raw_mean, tensor([0.])], dim = -1)
        beta.log_prob(beta(params), tensor([0.])).backward()
        return raw_mean.grad.abs().item()

    assert raw_mean_grad(True) < 50.
    assert raw_mean_grad(False) > 1000.

# bounds can be given as val_range / range / bounds or the first positional argument

@param('kwargs', [
    dict(val_range = (-2., 2.)),
    dict(range = (-2., 2.)),
    dict(bounds = (-2., 2.)),
])
def test_bounds(kwargs):
    beta = Beta(**kwargs)
    assert beta.bounds == (-2., 2.)

    actions = beta(torch.randn(8, 4, 2)).sample()
    assert ((actions >= -2.) & (actions <= 2.)).all()

def test_positional_bounds_and_pos_fn():
    beta = Beta((-0.4, 0.4))
    assert beta.bounds == (-0.4, 0.4)

    beta = Beta((-0.4, 0.4), 'softplus')
    assert beta.bounds == (-0.4, 0.4) and beta.pos_fn == 'softplus'

    beta = Beta('exp', (-0.4, 0.4))
    assert beta.bounds == (-0.4, 0.4) and beta.pos_fn == 'exp'

    beta = Beta('softplus')
    assert beta.bounds == (-1., 1.) and beta.pos_fn == 'softplus'

# log prob stays finite and differentiable at and beyond the support bounds

@param('val_range', [(-1., 1.), (0., 1.)])
def test_log_prob_at_bounds(val_range):
    beta = Beta(val_range = val_range)
    params = torch.randn(2, requires_grad = True)
    dist = beta(params)

    low, high = val_range
    actions = tensor([low, high, low - 1., high + 1.])

    log_probs = dist.log_prob(actions)
    assert torch.isfinite(log_probs).all()
    assert torch.allclose(log_probs[0], log_probs[2])
    assert torch.allclose(log_probs[1], log_probs[3])

    log_probs.sum().backward()
    assert torch.isfinite(params.grad).all()

# temperature sharpens / flattens the distribution

def test_temperature():
    beta = Beta()
    params = torch.randn(8, 4, 2) * 0.5
    action = torch.rand(8, 4)

    sharp = beta(params, temperature = 0.5)
    wide = beta(params, temperature = 2.)
    assert sharp.entropy().mean() < wide.entropy().mean()

    assert torch.allclose(beta.log_prob(params, action, temperature = 0.5), beta.log_prob(sharp, action))
    assert beta.sample(params, (2,), temperature = 0.5).shape == (2, 8, 4)
    assert beta.rsample(params, (2,), temperature = 0.5).shape == (2, 8, 4)

    with raises(AssertionError):
        beta(params, temperature = 0.)

# unit range is exactly the native beta, no jacobian shift

def test_unit_range_matches_native_beta():
    beta = Beta(range = (0., 1.))
    dist = beta(torch.randn(8, 4, 2))
    actions = dist.sample()

    assert torch.allclose(beta.log_prob(dist, actions, sum_action_dim = False), dist.log_prob(actions))
    assert torch.allclose(beta.entropy(dist, sum_action_dim = False), dist.entropy())

# arbitrary range rescales entropy by the log jacobian

def test_arbitrary_range():
    beta = Beta(range = (-2., 2.))
    dist = beta(torch.randn(8, 4, 2))

    assert ((dist.sample() >= -2.) & (dist.sample() <= 2.)).all()
    assert torch.allclose(beta.entropy(dist, sum_action_dim = False), dist.base_dist.entropy() + math.log(4.), atol = 1e-4)

# raw concentration is clamped to keep init_conc exact and extremes stable

def test_concentration_clamp():
    beta = Beta(init_conc = 10.)
    assert beta.clamp_exp == (-4., 4.)

    assert math.isclose(beta.concentration(tensor([0., 0.])).item(), 10., rel_tol = 1e-5)
    assert math.isclose(beta.concentration(tensor([0., 100.])).item(), 10. * math.exp(4.), rel_tol = 1e-4)
    assert math.isclose(beta.concentration(tensor([0., -100.])).item(), 10. * math.exp(-4.), rel_tol = 1e-4)
    assert math.isclose(Beta(init_conc = 1e6).concentration(tensor([0., 0.])).item(), 1e6, rel_tol = 1e-5)

    assert Beta(clamp_exp = 5.).clamp_exp == (-5., 5.)
    assert Beta(clamp_log_conc = 2.).clamp_exp == (-2., 2.)
    assert Beta(clamp_exp = None).clamp_log_conc is None

    with raises(AssertionError):
        Beta(min_conc = -1.)

# unimodal can be turned off, and the float shorthand sets the floor

def test_unimodal_flag():
    assert Beta().unimodal and Beta().max_unimodal_floor == 20.
    assert not Beta(unimodal = False).unimodal
    assert not Beta(unimodal = 0).unimodal
    assert Beta(unimodal = False).max_unimodal_floor is None
    assert Beta(unimodal = 15.).max_unimodal_floor == 15.

    params = tensor([[[100., 0.]]])

    def added_floor(beta):
        base = beta(params).base_dist
        return (base.concentration1 + base.concentration0 - beta.concentration(params)).item()

    assert math.isclose(added_floor(Beta(unimodal = False)), 0., abs_tol = 1e-5)
    assert math.isclose(added_floor(Beta(unimodal = True, max_unimodal_floor = 20.)), 20., rel_tol = 1e-4)

# entropy regularization should not move the mean

def test_detach_entropy_mean():
    beta = Beta()
    params = tensor([[0.5, 0.]], requires_grad = True)
    beta(params).entropy().backward()

    assert torch.allclose(params.grad[0, 0], tensor(0.), atol = 1e-6)
    assert not torch.allclose(params.grad[0, 1], tensor(0.))

# rescale env step - auto rescale and clip actions into the env action space

def test_rescale_env_step():
    beta = Beta()

    step = beta.rescale_env_step(lambda a: a, target_range = (-0.4, 0.4))
    assert torch.allclose(step(tensor([-1., 0., 1.])), tensor([-0.4, 0., 0.4]))

    step = beta.rescale_env_step(lambda a: a, target_range = (-0.4, 0.4), clip = (-0.5, 0.5))
    assert torch.allclose(step(tensor([-1., 0.2, 1.])), tensor([-0.4, 0.08, 0.4]))

    step = beta.rescale_env_step(lambda a: a, target_range = (-2., 2.), clip = True)
    assert torch.allclose(step(tensor([-1., 0.2, 1.])), tensor([-2., 0.4, 2.]))

    step = beta.rescale_env_step(lambda a: a, target_range = (-2., 2.), clip = False)
    assert torch.allclose(step(tensor([-3., 0., 3.])), tensor([-6., 0., 6.]))

    identity = lambda a: a
    assert beta.rescale_env_step(identity) is identity

# bounds must be a (low, high) pair or a (num_actions, 2) stack of pairs

def test_invalid_bounds():
    with raises(AssertionError):
        Beta(bounds = 0.)

    with raises(AssertionError):
        Beta(bounds = (-2., 2., 2.))

    with raises(AssertionError):
        Beta(bounds = ([-1., 0., -2.], [1., 1., 2.]))

def test_rescale_env_step_gym():
    env = gym.make('Pendulum-v1')
    env.reset(seed = 42)

    beta = Beta()
    step = beta.rescale_env_step(env.step, target_range = (-2., 2.))

    obs, reward, terminated, truncated, info = step(beta.sample(torch.randn(1, 2)).numpy())

    assert obs.shape == (3,)
    env.close()

# e2e - per-action bounds given as a list, tuple, or tensor of pairs

@param('bounds', [
    [[-1., 1.], [0., 1.], [-2., 2.]],
    ((-1., 1.), (0., 1.), (-2., 2.)),
    tensor([[-1., 1.], [0., 1.], [-2., 2.]])
])
def test_per_action_bounds(bounds):
    beta = Beta(bounds = bounds)
    assert torch.allclose(beta.low, tensor([-1., 0., -2.]))
    assert torch.allclose(beta.high, tensor([1., 1., 2.]))

    params = torch.randn(8, 3, 2, requires_grad = True)
    dist = beta(params)

    actions = dist.sample()
    assert ((actions >= beta.low) & (actions <= beta.high)).all()

    # per-action jacobian shifts entropy by the log scale

    assert torch.allclose(beta.entropy(dist, sum_action_dim = False), dist.base_dist.entropy() + (beta.high - beta.low).log(), atol = 1e-4)

    # ppo style loss and backward

    loss = -beta.log_prob(dist, actions).mean() - beta.entropy(dist).mean()
    loss.backward()
    assert torch.isfinite(params.grad).all()

    # kl divergence against a previous policy

    old_params = torch.randn(8, 3, 2)
    kl = beta.kl_divergence(old_params, params.detach())
    assert kl.shape == (8,)
    assert (kl >= 0.).all()

# two-action bounds as a list or tuple of pairs

@param('bounds', [
    [[-1., 1.], [0., 1.]],
    ((-1., 1.), (0., 1.))
])
def test_two_action_bounds(bounds):
    beta = Beta(bounds = bounds)
    assert torch.allclose(beta.low, tensor([-1., 0.]))
    assert torch.allclose(beta.high, tensor([1., 1.]))

# rescale env step - auto rescale and clip into per-action ranges

def test_per_action_rescale_env_step():
    beta = Beta()
    target_range = [[-1., 1.], [0., 1.], [-2., 2.]]

    step = beta.rescale_env_step(lambda a: a, target_range = target_range, clip = True)

    actions = tensor([[-3., -3., -3.], [-1., 0., 0.], [1., 1., 1.], [3., 3., 3.]])
    expected = tensor([[-1., 0., -2.], [-1., 0.5, 0.], [1., 1., 2.], [1., 1., 2.]])

    assert torch.allclose(step(actions), expected)
    assert torch.allclose(torch.from_numpy(step(actions.numpy())), expected)
