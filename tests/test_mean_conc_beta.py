from __future__ import annotations
import math

import pytest
from pytest import raises

param = pytest.mark.parametrize

import torch
from torch import tensor
import gymnasium as gym

from mean_conc_beta import Beta, LeakyTanh, leaky_tanh
from mean_conc_beta.mean_conc_beta import clamp

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
        beta(params, temperature = -1.)

# temperature = 0 is greedy / deterministic evaluation

def test_temperature_zero():
    beta = Beta()
    params = torch.randn(8, 4, 2, requires_grad = True)
    expected_mean = beta.mean(params)

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

    # (unit mean, concentration) -> alpha = unit mean * conc, beta = (1 - unit mean) * conc
    # J shape: alpha > 1, beta <= 1 -> high bound; alpha <= 1, beta > 1 -> low bound
    # U shape: alpha, beta < 1 -> nearest bound

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
