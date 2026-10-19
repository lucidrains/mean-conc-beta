from __future__ import annotations

import math

import gymnasium as gym
import pytest
import torch
from pytest import raises
from torch import tensor

from mean_conc_beta import Beta, LeakyTanh, leaky_tanh
from mean_conc_beta.mean_conc_beta import BOUNDS_LOSS_FNS, clamp

param = pytest.mark.parametrize

# e2e - forward, sample, rsample, log prob, entropy, mode, backward

@param('pos_fn', ['exp', 'softplus', 'elu'])
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
    beta = Beta(val_range = val_range, squash_fn = 'tanh')
    low, high = val_range

    unit_means = torch.linspace(0.1, 0.9, 5)
    params = torch.stack([torch.atanh(unit_means * 2. - 1.), torch.zeros(5)], dim = -1)

    expected = unit_means * (high - low) + low

    assert torch.allclose(beta.mean(params), expected, atol = 1e-5)
    assert torch.allclose(beta(params).mean, expected, atol = 1e-5)

# mean squashing functions and leaky tanh

@param('squash_fn', ['tanh', 'softsign', 'algebraic', 'leaky_tanh', leaky_tanh, LeakyTanh, LeakyTanh(leak = 0.2)])
def test_squash_fn(squash_fn):
    beta = Beta(squash_fn = squash_fn)
    params = torch.randn(8, 4, 2, requires_grad = True)
    dist = beta(params)

    assert ((dist.mean >= beta.low) & (dist.mean <= beta.high)).all()
    assert dist.rsample().shape == (8, 4)

    dist.rsample().sum().backward()
    assert torch.isfinite(params.grad).all()

def test_squash_fn_boundary_gradient():
    raw_mean = tensor([10.], requires_grad = True)
    params = torch.stack([raw_mean, tensor([0.])], dim = -1)

    # tanh vanishes to 0 in float32
    Beta(squash_fn = 'tanh').mean(params).backward()
    assert raw_mean.grad.item() == 0.

    # non-saturating alternatives keep gradient alive
    for fn in ('softsign', 'algebraic', 'leaky_tanh', leaky_tanh, LeakyTanh):
        raw_mean.grad.zero_()
        Beta(squash_fn = fn).mean(params).backward()
        assert raw_mean.grad.item() > 1e-4

def test_leaky_tanh():
    x = tensor([2.], requires_grad = True)

    # forward is exactly tanh, backward is floored at leak
    out = LeakyTanh(leak = 0.1)(x)
    assert torch.allclose(out, tensor([2.]).tanh(), atol = 1e-6)
    assert torch.allclose(leaky_tanh(tensor([2.]), leak = 0.1), tensor([2.]).tanh(), atol = 1e-6)

    out.backward()
    assert torch.allclose(x.grad, tensor([0.9 * (1. - math.tanh(2.) ** 2) + 0.1]))

    with raises(AssertionError):
        LeakyTanh(-0.1)

    with raises(AssertionError):
        leaky_tanh(x, -0.1)

    with raises(AssertionError):
        Beta(squash_fn = 'unknown')

# unbounded floor forces alpha, beta > 1, default damping avoids U shapes

@param('pos_fn', ['exp', 'softplus', 'elu'])
@param('val_range', [(-1., 1.), (0., 1.)])
def test_unimodality(pos_fn, val_range):
    params = tensor([[[-100., 0.], [100., 0.]], [[0., -10.], [0., 10.]]])

    unbounded = Beta(pos_fn = pos_fn, val_range = val_range, max_unimodal_floor = None)(params).base_dist
    assert (unbounded.concentration1 > 1.).all()
    assert (unbounded.concentration0 > 1.).all()

    damped = Beta(pos_fn = pos_fn, val_range = val_range)(params).base_dist
    assert (damped.concentration1 >= 1.).all()
    assert (damped.concentration0 >= 1.).all()

# detaching the unimodal floor keeps boundary gradients bounded

def test_detach_unimodal():
    def raw_mean_grad(detach):
        raw_mean = tensor([5.], requires_grad = True)
        beta = Beta(detach_unimodal = detach, max_unimodal_floor = None, squash_fn = 'tanh')
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

def test_integer_bounds_are_float():
    beta = Beta(bounds = (-1, 1))

    assert beta.low.is_floating_point() and beta.scale.is_floating_point()
    assert isinstance(beta.unit_eps, float)

    half = beta.half()
    assert half.low.dtype == torch.float16 and half.scale.dtype == torch.float16

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

@param('val_range', [(-1., 1.), (0., 1.), (-512., 512.)])
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
        beta(params, temperature = -1.)

# temperature = 0 is greedy / deterministic evaluation

def test_temperature_zero():
    beta = Beta()
    params = torch.randn(8, 4, 2, requires_grad = True)

    # the greedy limit is the raw unit mean, without the finite concentration unimodal nudge

    expected_mean = beta.affine(params)(beta.unit_mean_from_params(params))

    dist = beta(params, temperature = 0.)
    assert torch.allclose(dist.mean, expected_mean)
    assert torch.allclose(dist.mode, expected_mean)
    assert torch.allclose(dist.sample(), expected_mean)
    assert torch.allclose(dist.rsample(), expected_mean)
    assert torch.allclose(dist.variance, torch.zeros_like(expected_mean))
    assert torch.allclose(dist.entropy(), torch.zeros_like(expected_mean))
    assert torch.allclose(beta.entropy(params, temperature = 0.), torch.zeros(8))
    assert dist.sample((3,)).shape == (3, 8, 4)
    assert dist.sample(3).shape == (3, 8, 4)

    # deterministic kl divergence
    dist_clone = beta(params, temperature = 0.)
    assert torch.allclose(beta.kl_divergence(dist, dist_clone), torch.zeros(8))

    # deterministic log_prob is not defined
    with raises(NotImplementedError):
        dist.log_prob(expected_mean)

    with raises(NotImplementedError):
        beta.log_prob(dist, expected_mean)

    # module shortcuts
    assert torch.allclose(beta.sample(params, temperature = 0.), expected_mean)
    assert torch.allclose(beta.rsample(params, temperature = 0.), expected_mean)
    assert torch.allclose(beta.mode(params, temperature = 0.), expected_mean)
    assert beta.sample(params, 3, temperature = 0.).shape == (3, 8, 4)
    assert beta.rsample(params, 3, temperature = 0.).shape == (3, 8, 4)

    # backward works through rsample
    dist.rsample().sum().backward()
    assert torch.isfinite(params.grad).all()

    # per-action bounds
    bounds = [[-1., 1.], [0., 1.], [-2., 2.]]
    beta_per_action = Beta(bounds = bounds)
    params_per_action = torch.randn(5, 3, 2)
    dist_pa = beta_per_action(params_per_action, temperature = 0.)
    actions = dist_pa.sample()
    assert ((actions >= beta_per_action.low) & (actions <= beta_per_action.high)).all()

# mode handles non-unimodal distributions (alpha <= 1 or beta <= 1)

@param('bounds', [(-1., 1.), (0., 1.)])
def test_non_unimodal_mode(bounds):
    beta = Beta(bounds = bounds, unimodal = False, pos_fn = 'exp')
    low, high = bounds

    # (unit mean, conc) -> alpha = um * conc, beta = (1 - um) * conc
    # J shapes land on a bound by whichever concentration exceeds one, U shapes on the nearest

    cases = [
        (0.8, 1.5, 1.),
        (0.2, 1.5, 0.),
        (0.8, 0.5, 1.),
        (0.2, 0.5, 0.)
    ]

    for unit_mean, conc, expected in cases:
        params = tensor([[math.atanh(unit_mean * 2. - 1.), math.log(conc / beta.init_conc)]])
        expected_mode = low + expected * (high - low)
        assert torch.allclose(beta.mode(params), tensor([expected_mode]), atol = 1e-4)

    # uniform is finite, does not NaN

    params = tensor([[0., math.log(2. / beta.init_conc)]])
    assert torch.isfinite(beta.mode(params)).all()

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

# the uniform prior loss - kl against the uniform on the bounds reduces to the negative base beta entropy

@param('pos_fn', ['exp', 'softplus', 'elu'])
@param('val_range', [(-1., 1.), (0., 1.), (-3., 5.)])
def test_kl_uniform(pos_fn, val_range):
    torch.manual_seed(0)

    beta = Beta(pos_fn = pos_fn, val_range = val_range, unimodal = False)
    params = torch.stack((torch.randn(8, 4), torch.randn(8, 4)), dim = -1)

    kl = beta.kl_uniform(params)
    assert kl.shape == (8,)

    unit_mean, conc = beta.concentrations(params)
    alpha, beta_conc = unit_mean * conc, (1. - unit_mean) * conc

    # closed form -H(beta) = -[log B(alpha, beta) - (alpha - 1) psi(alpha) - (beta - 1) psi(beta) + (alpha + beta - 2) psi(alpha + beta)]

    expected = (
        -torch.lgamma(alpha) - torch.lgamma(beta_conc) + torch.lgamma(alpha + beta_conc)
        + (alpha - 1.) * torch.digamma(alpha)
        + (beta_conc - 1.) * torch.digamma(beta_conc)
        - (alpha + beta_conc - 2.) * torch.digamma(alpha + beta_conc)
    ).sum(dim = -1)

    assert torch.allclose(kl, expected, atol = 1e-5)
    assert (kl >= -1e-6).all()

    # invariant to the bounds, as everything but the shapes of the beta cancels

    other_bounds = Beta(pos_fn = pos_fn, val_range = (10., 11.), unimodal = False).kl_uniform(params)
    assert torch.allclose(kl, other_bounds, atol = 1e-6)

    # gradients flow through the concentration, with the mean detached by default

    params = params.detach().requires_grad_()
    beta.kl_uniform(params).sum().backward()

    assert torch.isfinite(params.grad).all()
    assert torch.allclose(params.grad[..., 0], torch.zeros_like(params.grad[..., 0]))
    assert (params.grad[..., 1].abs() > 0.).all()

    # opt out to let the mean move as well

    params = params.detach().requires_grad_()
    beta.kl_uniform(params, detach_mean = False).sum().backward()

    assert torch.isfinite(params.grad).all()
    assert (params.grad.abs() > 0.).all()

    assert torch.allclose(beta.prior_loss(params.detach()), beta.kl_uniform(params.detach()))

def test_kl_uniform_zero_at_uniform():
    # init_conc = 2 with a raw mean of 0 gives alpha = beta = 1, exactly the uniform, so the kl is 0

    kl = Beta(init_conc = 2., unimodal = False).kl_uniform(torch.zeros(8, 4, 2))
    assert kl.shape == (8,)
    assert torch.allclose(kl, torch.zeros(8), atol = 1e-5)

    # the deterministic distribution (temperature = 0) is asserted out

    with raises(AssertionError):
        Beta(unimodal = False).kl_uniform(torch.randn(8, 4, 2), temperature = 0.)

def test_kl_uniform_monte_carlo():
    torch.manual_seed(0)

    val_range = (-2., 3.)
    beta = Beta(val_range = val_range, unimodal = False)

    # moderate concentrations, where a plain monte carlo estimate of E_p[log p - log uniform] is tight

    params = torch.stack((torch.randn(8, 3) * 0.3, torch.randn(8, 3) * 0.2), dim = -1).double()
    dist = beta(params)

    samples = dist.rsample((200_000,))
    log_uniform = -math.log(val_range[1] - val_range[0])

    kl_mc = (dist.log_prob(samples) - log_uniform).mean(dim = 0)
    assert torch.allclose(kl_mc, beta.kl_uniform(params, sum_action_dim = False), atol = 2e-2)

# raw concentration is clamped to keep init_conc exact and extremes stable

def test_concentration_clamp():
    beta = Beta(init_conc = 10., pos_fn = 'exp')
    assert beta.clamp_exp == (-4., 4.)

    assert math.isclose(beta.concentration(tensor([0., 0.])).item(), 10., rel_tol = 1e-5)
    assert math.isclose(beta.concentration(tensor([0., 100.])).item(), 10. * math.exp(4.), rel_tol = 1e-4)
    assert math.isclose(beta.concentration(tensor([0., -100.])).item(), 10. * math.exp(-4.), rel_tol = 1e-4)
    assert math.isclose(Beta(init_conc = 1e6, pos_fn = 'exp').concentration(tensor([0., 0.])).item(), 1e6, rel_tol = 1e-5)
    assert math.isclose(Beta(pos_fn = 'softplus', init_conc = 1e6).concentration(tensor([0., 0.])).item(), 1e6, rel_tol = 1e-4)

    assert Beta(clamp_exp = 5.).clamp_exp == (-5., 5.)
    assert Beta(clamp_log_conc = 2.).clamp_exp == (-2., 2.)
    assert Beta(clamp_exp = None).clamp_log_conc is None

    with raises(AssertionError):
        Beta(min_conc = -1.)

# elu(x) + 1 as a positive function, with an identity preclamp

def test_elu_pos_fn():
    beta = Beta(pos_fn = 'elu', init_conc = 10.)
    assert math.isclose(beta.concentration(tensor([0.]), indexed = True).item(), 10., rel_tol = 1e-5)

    # the inverse switches to log below one

    assert math.isclose(Beta(pos_fn = 'elu', init_conc = 1.).concentration(tensor([0.]), indexed = True).item(), 1., rel_tol = 1e-5)
    assert math.isclose(Beta(pos_fn = 'elu', init_conc = 0.5).concentration(tensor([0.]), indexed = True).item(), 0.5, rel_tol = 1e-5)

    # clamp_exp does not apply

    beta = Beta(pos_fn = 'elu', init_conc = 10., clamp_exp = 2.)
    assert math.isclose(beta.concentration(tensor([4.]), indexed = True).item(), 14., rel_tol = 1e-4)
    assert math.isclose(beta.concentration(tensor([-5.]), indexed = True).item(), 5., rel_tol = 1e-4)

# unimodal can be turned off, and the float shorthand sets the floor

def test_unimodal_flag():
    assert Beta().unimodal and Beta().max_unimodal_floor == 50.
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

# the detach flag can be overridden on an already constructed distribution

def test_detach_entropy_mean_override():
    beta = Beta()

    def mean_grad(detach_mean):
        params = tensor([[0.5, 0.]], requires_grad = True)
        beta.entropy(beta(params), detach_mean = detach_mean).backward()
        return params.grad[0, 0]

    assert not torch.allclose(mean_grad(False), tensor(0.))
    assert torch.allclose(mean_grad(True), tensor(0.), atol = 1e-6)

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

    obs, *_ = step(beta.sample(torch.randn(1, 2)).numpy())

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

# kl across separate transforms / mixed precision casts of the same bounds

def test_kl_mixed_precision():
    beta = Beta()
    params_p = torch.randn(8, 3, 2).bfloat16()
    params_q = torch.randn(8, 3, 2).bfloat16()

    kl = beta.kl_divergence(params_p, params_q)
    assert kl.shape == (8,) and torch.isfinite(kl).all()

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

# clamp with mixed scalar and tensor bounds

def test_clamp_mixed():
    assert torch.allclose(clamp(0.5, tensor([0., -1.]), tensor([1., 1.])), tensor([0.5, 0.5]))
    assert torch.allclose(clamp(tensor([0.5, -2.]), -1., 1.), tensor([0.5, -1.]))

# sample and rsample with integer sample_shape

def test_sample_shape_int():
    beta = Beta()
    params = torch.randn(8, 4, 2)
    dist = beta(params)

    assert dist.sample(3).shape == (3, 8, 4)
    assert dist.rsample(3).shape == (3, 8, 4)
    assert beta.sample(params, 3).shape == (3, 8, 4)
    assert beta.rsample(params, 3).shape == (3, 8, 4)

# regular alpha / beta parameterization - raw params map directly to the two concentrations

@param('pos_fn', ['exp', 'softplus', 'elu'])
@param('val_range', [(-1., 1.), (0., 1.)])
def test_alpha_beta_parameterization(pos_fn, val_range):
    beta = Beta(pos_fn = pos_fn, val_range = val_range, param_with_alpha_beta = True, unimodal = False)
    params = torch.randn(8, 4, 2, requires_grad = True)

    alpha = beta.concentration(params[..., 0], indexed = True)
    beta_conc = beta.concentration(params[..., 1], indexed = True)

    dist = beta(params)
    assert torch.allclose(dist.base_dist.concentration1, alpha, atol = 1e-5)
    assert torch.allclose(dist.base_dist.concentration0, beta_conc, atol = 1e-5)

    low, high = val_range
    expected_mean = low + alpha / (alpha + beta_conc) * (high - low)

    assert torch.allclose(beta.mean(params), expected_mean, atol = 1e-5)
    assert torch.allclose(dist.mean, expected_mean, atol = 1e-5)

    # concentration is the sum of the two

    assert torch.allclose(beta.concentration(params), alpha + beta_conc, atol = 1e-5)

    # raw params of zero recover init_conc for both

    assert math.isclose(beta.concentration(tensor([0.]), indexed = True).item(), beta.init_conc, rel_tol = 1e-5)

    # temperature scales both concentrations

    sharp = beta(params, temperature = 0.5)
    assert torch.allclose(sharp.base_dist.concentration1, alpha * 2., atol = 1e-4)
    assert torch.allclose(sharp.base_dist.concentration0, beta_conc * 2., atol = 1e-4)

    # temperature 0 is deterministic at the mean

    dist_zero = beta(params, temperature = 0.)
    assert torch.allclose(dist_zero.mean, expected_mean, atol = 1e-5)

    # samples stay in bounds and gradients flow

    actions = dist.rsample()
    assert ((actions >= low) & (actions <= high)).all()
    assert beta.entropy(dist).shape == (8,)
    assert beta.log_prob(dist, actions).shape == (8,)

    dist.rsample().sum().backward()
    assert torch.isfinite(params.grad).all()

def test_alpha_beta_concentration_config():
    beta = Beta(param_with_alpha_beta = True, min_conc = 1., clamp_exp = 2., pos_fn = 'exp')

    # init_conc and min_conc set the raw offset, clamp_exp bounds the raw concentrations

    assert math.isclose(beta.concentration(tensor([0.]), indexed = True).item(), 10., rel_tol = 1e-5)
    assert math.isclose(beta.concentration(tensor([100.]), indexed = True).item(), 9. * math.exp(2.) + 1., rel_tol = 1e-4)
    assert math.isclose(beta.concentration(tensor([-100.]), indexed = True).item(), 9. * math.exp(-2.) + 1., rel_tol = 1e-4)

def test_alpha_beta_matches_native_beta():
    beta = Beta(range = (0., 1.), param_with_alpha_beta = True)
    params = torch.randn(8, 4, 2)
    dist = beta(params)

    actions = dist.sample()

    assert torch.allclose(beta.log_prob(dist, actions, sum_action_dim = False), dist.base_dist.log_prob(actions))
    assert torch.allclose(beta.entropy(dist, sum_action_dim = False), dist.base_dist.entropy())

# the unimodal floor applies to the alpha / beta parameterization as well

@param('pos_fn', ['exp', 'softplus', 'elu'])
def test_alpha_beta_unimodality(pos_fn):
    params = tensor([[[-100., 0.], [0., -100.], [100., 0.], [0., 100.]]])

    unbounded = Beta(pos_fn = pos_fn, param_with_alpha_beta = True, max_unimodal_floor = None)(params).base_dist
    assert (unbounded.concentration1 > 1.).all()
    assert (unbounded.concentration0 > 1.).all()

    damped = Beta(pos_fn = pos_fn, param_with_alpha_beta = True)(params).base_dist
    assert (damped.concentration1 >= 1.).all()
    assert (damped.concentration0 >= 1.).all()

# detaching the entropy mean also works for the alpha / beta parameterization -
# gradients flow only through the total concentration, in proportion to alpha and beta

def test_alpha_beta_detach_entropy_mean():
    params = tensor([[math.log(0.3), math.log(0.6)]], requires_grad = True)

    beta = Beta(param_with_alpha_beta = True, unimodal = False, pos_fn = 'exp')
    alpha = beta.concentration(params[..., 0], indexed = True)
    beta_conc = beta.concentration(params[..., 1], indexed = True)

    beta(params).entropy().backward()
    detached_grad = params.grad.clone()

    params.grad = None
    Beta(param_with_alpha_beta = True, unimodal = False, detach_entropy_mean = False, pos_fn = 'exp')(params).entropy().backward()

    assert torch.allclose(detached_grad[0, 0] / alpha, detached_grad[0, 1] / beta_conc)
    assert not torch.allclose(params.grad[0, 0] / alpha, params.grad[0, 1] / beta_conc)

def test_alpha_beta_mode_and_kl():
    beta = Beta(param_with_alpha_beta = True, unimodal = False, pos_fn = 'exp')

    # raw params log(0.3) and log(0.6) give alpha = 3 and beta = 6 at init_conc = 10
    # mode = (alpha - 1) / (alpha + beta - 2) = 2/7 on (-1., 1.)

    params = tensor([[[math.log(0.3), math.log(0.6)]]])
    expected_mode = -1. + (2. / 7.) * 2.

    assert torch.allclose(beta.mode(params), tensor([[expected_mode]]), atol = 1e-4)
    assert torch.allclose(beta.kl_divergence(params, params), torch.zeros(1), atol = 1e-5)

# raw params to transformed params, then onwards to sampling and other downstream operations

@param('pos_fn', ['exp', 'softplus', 'elu'])
@param('param_with_alpha_beta', [False, True])
def test_to_from_transformed(pos_fn, param_with_alpha_beta):
    beta = Beta(pos_fn = pos_fn, param_with_alpha_beta = param_with_alpha_beta)
    params = torch.randn(16, 2)

    unit_mean, conc = beta.concentrations(params)
    transformed = beta.to_transformed(params)

    assert torch.allclose(transformed.unit_mean, unit_mean)
    assert torch.allclose(transformed.concentration, conc)

    raw_dist = beta(params)
    transformed_dist = beta.from_transformed(transformed)

    assert torch.allclose(raw_dist.mean, transformed_dist.mean, atol = 1e-5)
    assert torch.allclose(raw_dist.variance, transformed_dist.variance, atol = 1e-5)

    actions = transformed_dist.sample()
    assert actions.shape == (16,)
    assert transformed_dist.log_prob(actions).shape == (16,)
    assert transformed_dist.entropy().shape == (16,)
    assert transformed_dist.rsample().shape == (16,)

@param('detach_mean', [True, False])
def test_from_transformed_inherits_detach_entropy_mean(detach_mean):
    beta = Beta(detach_entropy_mean = detach_mean)
    params = torch.randn(8, 2, requires_grad = True)

    dist = beta.from_transformed(beta.to_transformed(params))
    assert dist.detach_entropy_mean is detach_mean

    dist.entropy().sum().backward()

    mean_grad_zero = torch.allclose(params.grad[..., 0], torch.zeros_like(params.grad[..., 0]))
    assert mean_grad_zero is detach_mean

@param('pos_fn', ['exp', 'softplus', 'elu'])
def test_transformed_params_are_differentiable(pos_fn):
    beta = Beta(pos_fn = pos_fn)
    params = torch.randn(16, 2, requires_grad = True)

    transformed = beta.to_transformed(params)
    dist = beta.from_transformed(transformed)
    dist.rsample().sum().backward()

    assert params.grad is not None
    assert torch.isfinite(params.grad).all()

def test_deterministic_transformed():
    beta = Beta()
    action = torch.rand(16) * 2. - 1.

    transformed = beta.deterministic_transformed(action)
    dist = beta.from_transformed(transformed)

    assert (dist.mean - action).abs().max() < 1e-2
    assert dist.variance.max() < 1e-3

# regression - a saturated raw mean with wide bounds must not produce a zero beta concentration
# (1. - eps / scale can round back to 1. in float32, e.g. at action bounds of 512.)

@param('unimodal', [True, False])
@param('param_with_alpha_beta', [True, False])
@param('raw_mean', [20., 100., 1e4, -1e4])
@param('dtype', [torch.float32, torch.float64])
def test_saturated_mean_stays_valid(unimodal, param_with_alpha_beta, raw_mean, dtype):
    beta = Beta(
        bounds = [-512., 512.],
        init_conc = 2.,
        unimodal = unimodal,
        param_with_alpha_beta = param_with_alpha_beta,
        eps = 1e-5
    )

    raw_conc = 0. if not param_with_alpha_beta else 1e-3
    params = tensor([[raw_mean, raw_conc], [-raw_mean, raw_conc]], dtype = dtype, requires_grad = True)

    dist = beta(params)
    actions = dist.sample()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(beta.log_prob(dist, actions)).all()
    assert torch.isfinite(beta.entropy(dist)).all()

    dist.mean.sum().backward()
    assert torch.isfinite(params.grad).all()

# tiny eps must still keep the concentrations positive

@param('unimodal', [True, False])
def test_tiny_eps_stays_valid(unimodal):
    beta = Beta(
        bounds = [-512., 512.],
        init_conc = 2.,
        unimodal = unimodal,
        eps = 1e-12
    )

    params = tensor([[50., 0.], [-50., 0.]], requires_grad = True)

    dist = beta(params)
    actions = dist.sample()
    assert torch.isfinite(actions).all()
    assert torch.isfinite(beta.log_prob(dist, actions)).all()

# the out of bounds loss - the amount a mean action lands past the bounds, zero inside

@param('loss_fn', list(BOUNDS_LOSS_FNS.keys()))
def test_out_of_bounds_loss_fns(loss_fn):
    beta = Beta(bounds = (-1., 1.), overshoot = 0.1)
    loss = beta.out_of_bounds_loss(tensor([[[10., 0.], [0., 0.]]]), loss_fn = loss_fn, sum_action_dim = False)

    assert torch.isfinite(loss).all()
    assert (loss >= 0.).all()
    assert loss[0, 1] == 0.
    assert loss[0, 0] > 0.

def test_out_of_bounds_loss():
    params = tensor([[[10., 0.], [0., 0.], [-10., 0.]]])   # a batch of three actions

    # no overshoot, the mean is always inside the bounds

    assert torch.allclose(Beta(bounds = (-1., 1.)).out_of_bounds_loss(params), torch.zeros(1))

    # a saturated mean lands that much past the bound, squared

    beta = Beta(bounds = (-1., 1.), overshoot = 0.1)
    past = beta.mean(params)[0, 0].abs().item() * 1.1 - 1.

    assert torch.allclose(beta.out_of_bounds_loss(params, sum_action_dim = False), tensor([[past ** 2, 0., past ** 2]]), atol = 1e-5)

    # bounds overridable, backward works

    assert (beta.out_of_bounds_loss(params, bounds = (-0.5, 0.5)) > beta.out_of_bounds_loss(params)).all()

    differentiable = tensor([[[10., 0.], [0., 0.]]], requires_grad = True)
    beta.out_of_bounds_loss(differentiable).sum().backward()
    assert torch.isfinite(differentiable.grad).all()

# mse lets go of a mean that is far out, huber keeps pulling, or any callable

def test_out_of_bounds_loss_fns_far_out():
    params = tensor([[[10., 0.]]])
    beta = Beta(bounds = (-1., 1.), overshoot = 3.)
    far = beta.mean(params)[0, 0].abs() * 4. - 1.
    assert far > 1.

    assert torch.allclose(beta.out_of_bounds_loss(params, loss_fn = 'mse'), tensor([far ** 2]), atol = 1e-4)
    assert torch.allclose(beta.out_of_bounds_loss(params, loss_fn = 'huber'), tensor([far - 0.5]), atol = 1e-4)
    assert torch.allclose(beta.out_of_bounds_loss(params, loss_fn = lambda x: x), tensor([far]), atol = 1e-4)

    with raises(AssertionError):
        beta.out_of_bounds_loss(params, loss_fn = 'unknown')

# env wrapper kwargs - the action transform rescales the policy range onto the env bounds,
# with an optional overshoot that is clipped back onto them

def test_action_transform_kwargs():
    beta = Beta(bounds = (-2., 2.))
    assert beta.action_transform_kwargs() == dict(auto = True, from_range = (-2., 2.), overshoot = None)

    beta = Beta(bounds = (-2., 2.), overshoot = 0.05)
    assert beta.action_transform_kwargs() == dict(auto = True, from_range = (-2., 2.), overshoot = 0.05)

    beta = Beta(bounds = [(0., 1.), (-2., 2.)], overshoot = 0.05)
    assert beta.action_transform_kwargs() == dict(auto = True, from_range = ((0., -2.), (1., 2.)), overshoot = 0.05)

# e2e with env_ssl_wrapper if installed - a saturated mean action is clipped onto the env bound

def test_action_transform_kwargs_reach_bound():
    pytest.importorskip('env_ssl_wrapper')

    from env_ssl_wrapper.standardize import ActionTransformWrapper
    from env_ssl_wrapper.standardize.mocks import GymnasiumMockEnv

    # the mock env's bounds are the beta default (-1., 1.)

    beta = Beta(overshoot = 0.05)
    env = ActionTransformWrapper(GymnasiumMockEnv(), **beta.action_transform_kwargs())

    # a saturated mean still lands short of a bound, the wrapper clips the overshoot onto it

    saturated = beta.mean(tensor([[10., 0.], [-10., 0.]]))
    assert (saturated.abs() < 1. - 1e-3).all()

    env.step(saturated)
    assert torch.allclose(torch.from_numpy(env.last_action), tensor([1., -1.]), atol = 1e-5)
