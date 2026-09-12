from __future__ import annotations
import math
import pytest
from pytest import raises
param = pytest.mark.parametrize

import torch
from torch import tensor
from torch.distributions import AffineTransform, Beta as _Beta, SigmoidTransform

import gymnasium as gym

from mean_conc_beta import Beta
from mean_conc_beta.mean_conc_beta import (
    TransformedBeta,
    exists,
    rescale_from_to
)

# test mean preservation

@param('val_range', [(-1., 1.), (0., 1.)])
@param('pos_fn', ['exp', 'softplus'])
@param('detach_unimodal', [True, False])
def test_mean_preservation(val_range, pos_fn, detach_unimodal):
    beta = Beta(pos_fn = pos_fn, init_conc = 10.0, detach_unimodal = detach_unimodal, val_range = val_range)

    targets = [-0.8, -0.5, 0.0, 0.5, 0.8] if val_range == (-1., 1.) else [0.1, 0.25, 0.5, 0.75, 0.9]

    for target_m in targets:
        unit_target = (target_m + 1.) / 2. if val_range == (-1., 1.) else target_m
        raw_m = math.atanh(unit_target * 2. - 1.)
        params = tensor([[raw_m, 0.0]])
        dist = beta(params)

        true_mean = dist.mean.item()
        det_mean = beta.mean(params).item()

        assert abs(true_mean - target_m) < 1e-5
        assert abs(det_mean - target_m) < 1e-5

# test unimodality under extreme latents

@param('val_range', [(-1., 1.), (0., 1.)])
@param('pos_fn', ['exp', 'softplus'])
def test_unimodality_extreme_latents(val_range, pos_fn):
    beta = Beta(pos_fn = pos_fn, val_range = val_range)

    # extreme negative and positive raw means

    params = tensor([[[-100.0, 0.0], [100.0, 0.0]], [[0.0, -10.0], [0.0, 10.0]]])
    dist = beta(params)

    base_dist = getattr(dist, 'base_dist', dist)
    alpha = base_dist.concentration1
    beta_param = base_dist.concentration0

    assert (alpha > 1.0).all()
    assert (beta_param > 1.0).all()

# test unimodality under wide action ranges

@param('bounds', [(-2., 2.), (3., 7.)])
def test_unimodality_wide_bounds(bounds):
    beta = Beta(bounds = bounds)

    params = tensor([[[-100.0, 0.0], [100.0, 0.0]], [[0.0, -10.0], [0.0, 10.0]]])
    dist = beta(params)

    base_dist = getattr(dist, 'base_dist', dist)

    assert (base_dist.concentration1 > 1.0).all()
    assert (base_dist.concentration0 > 1.0).all()

# test detach unimodal floor

def test_detach_unimodal_default_is_true():
    beta = Beta()
    assert beta.detach_unimodal
    assert beta.has_rsample

def test_detach_unimodal_gradient():
    raw_mean_detach = tensor([5.0], requires_grad = True)
    beta_detach = Beta(detach_unimodal = True)
    params_detach = torch.stack([raw_mean_detach, tensor([0.0])], dim = -1)
    dist_detach = beta_detach(params_detach)
    lp_detach = beta_detach.log_prob(dist_detach, tensor([0.0]))
    lp_detach.backward()

    raw_mean_nodetach = tensor([5.0], requires_grad = True)
    beta_nodetach = Beta(detach_unimodal = False)
    params_nodetach = torch.stack([raw_mean_nodetach, tensor([0.0])], dim = -1)
    dist_nodetach = beta_nodetach(params_nodetach)
    lp_nodetach = beta_nodetach.log_prob(dist_nodetach, tensor([0.0]))
    lp_nodetach.backward()

    assert abs(raw_mean_detach.grad.item()) < 50.0
    assert abs(raw_mean_nodetach.grad.item()) > 1000.0

# test clamp exp

def test_clamp_exp():
    beta = Beta(init_conc = 10.0)

    assert beta.clamp_exp == (-4.0, 4.0)

    # raw_conc = 0 preserves exact init_conc

    assert abs(beta.concentration(tensor([0.0, 0.0])).item() - 10.0) < 1e-5
    assert abs(beta.concentration(tensor(0.0), indexed = True).item() - 10.0) < 1e-5

    # raw_conc = 100 is clamped to 4.0, yielding init_conc * exp(4.0)

    conc_huge = beta.concentration(tensor([0.0, 100.0])).item()
    expected_huge = 10.0 * math.exp(4.0)
    assert abs(conc_huge - expected_huge) / expected_huge < 1e-4

    # raw_conc = -100 is clamped to -4.0, yielding init_conc * exp(-4.0)

    conc_tiny = beta.concentration(tensor([0.0, -100.0])).item()
    expected_tiny = 10.0 * math.exp(-4.0)
    assert abs(conc_tiny - expected_tiny) / expected_tiny < 1e-4

    beta_custom = Beta(clamp_exp = 5.0)
    assert beta_custom.clamp_exp == (-5.0, 5.0)

def test_clamp_log_conc_alias():
    beta = Beta(clamp_log_conc = 2.0)

    assert beta.clamp_exp == (-2.0, 2.0)
    assert beta.clamp_log_conc == 2.0

    conc_huge = beta.concentration(tensor([0.0, 100.0])).item()
    expected_huge = 10.0 * math.exp(2.0)
    assert abs(conc_huge - expected_huge) / expected_huge < 1e-4

    assert Beta().clamp_log_conc == 4.0
    assert Beta(clamp_exp = None).clamp_log_conc is None

def test_init_conc_not_overridden():
    beta_large = Beta(init_conc = 1e6)
    conc_at_zero = beta_large.concentration(tensor([0.0, 0.0])).item()
    assert abs(conc_at_zero - 1e6) < 1.0

    conc_at_zero = beta_large.concentration(tensor(0.0), indexed = True).item()
    assert abs(conc_at_zero - 1e6) < 1.0

def test_negative_min_conc_raises():
    with raises(AssertionError):
        Beta(min_conc = -1.0)

# test entropy and log prob

def test_entropy_and_log_prob():
    beta = Beta()
    params = torch.randn(8, 4, 2)
    dist = beta(params)

    actions = dist.sample()
    assert actions.shape == (8, 4)
    assert ((actions >= -1.0) & (actions <= 1.0)).all()

    lp = beta.log_prob(dist, actions, sum_action_dim = True)
    assert lp.shape == (8,)

    ent = beta.entropy(dist, sum_action_dim = True)
    assert ent.shape == (8,)

@param('val_range', [(-1., 1.), (0., 1.)])
def test_log_prob_protected_at_bounds(val_range):
    beta = Beta(val_range = val_range)
    params = torch.randn(2, requires_grad = True)
    dist = beta(params)

    min_val, max_val = val_range

    # actions on and beyond the open support boundaries

    actions = tensor([min_val, max_val, min_val - 1., max_val + 1.])

    log_probs = dist.log_prob(actions)
    assert torch.isfinite(log_probs).all()

    # beyond the bounds is clamped to the same safe point as the bounds

    assert torch.allclose(log_probs[0], log_probs[2])
    assert torch.allclose(log_probs[1], log_probs[3])

    module_log_probs = beta.log_prob(dist, actions, sum_action_dim = False)
    assert torch.allclose(module_log_probs, log_probs)

    # ensure backward pass succeeds and produces finite gradients
    log_probs.sum().backward()
    assert exists(params.grad)
    assert torch.isfinite(params.grad).all()

    # test custom eps
    custom_eps = 1e-3
    custom_dist_lp = dist.log_prob(actions, eps = custom_eps)
    custom_mod_lp = beta.log_prob(dist, actions, eps = custom_eps, sum_action_dim = False)
    assert torch.allclose(custom_dist_lp, custom_mod_lp)

def test_transformed_beta_tensor_bounds():
    base = _Beta(tensor([2., 3.]), tensor([3., 2.]))
    transform = AffineTransform(loc = tensor([0., 1.]), scale = tensor([2., 3.]))
    dist = TransformedBeta(base, transform)

    actions = tensor([[0., 1.], [-1., 5.]])
    lp = dist.log_prob(actions)
    assert torch.isfinite(lp).all()


# test sample and rsample

def test_sample_and_rsample():
    beta = Beta()
    params = torch.randn(12, 3, 2, requires_grad = True)

    # dist.rsample()

    dist = beta(params)
    assert dist.has_rsample
    r_act = dist.rsample()
    assert r_act.shape == (12, 3)

    loss = r_act.sum()
    loss.backward()
    assert exists(params.grad)
    assert not torch.isnan(params.grad).any()

    # beta.rsample() shortcut

    params2 = torch.randn(6, 3, 2, requires_grad = True)
    r_act2 = beta.rsample(params2, (4,))
    assert r_act2.shape == (4, 6, 3)

    r_act2.sum().backward()
    assert exists(params2.grad)

# test mode

def test_mode():
    beta = Beta()
    params = torch.randn(5, 3, 2)
    mode = beta.mode(params)
    assert mode.shape == (5, 3)
    assert ((mode >= -1.0) & (mode <= 1.0)).all()

# test multi-dim batch

def test_multi_dim_batch():
    beta = Beta()
    params = torch.randn(4, 16, 6, 2)
    dist = beta(params)

    sample = dist.sample()
    assert sample.shape == (4, 16, 6)

    lp = beta.log_prob(dist, sample)
    assert lp.shape == (4, 16)

# test transformed beta

def test_transformed_beta_properties():
    beta = Beta()
    params = torch.randn(8, 4, 2)
    dist = beta(params)
    base_dist = dist.base_dist

    assert isinstance(dist, TransformedBeta)
    assert dist.mean.shape == (8, 4)
    assert dist.mode.shape == (8, 4)
    assert dist.variance.shape == (8, 4)
    assert dist.stddev.shape == (8, 4)

    assert torch.allclose(dist.mean, base_dist.mean * 2. - 1., atol = 1e-5)
    assert torch.allclose(dist.variance, base_dist.variance * 4., atol = 1e-5)
    assert torch.allclose(dist.entropy(), base_dist.entropy() + math.log(2.), atol = 1e-5)

def test_transformed_beta_arbitrary_affine():
    base_dist = _Beta(tensor([2., 3.]), tensor([3., 2.]))
    transform = AffineTransform(loc = 1., scale = 3.)
    dist = TransformedBeta(base_dist, transform)

    assert torch.allclose(dist.mean, transform(base_dist.mean))
    assert torch.allclose(dist.mode, transform(base_dist.mode))
    assert torch.allclose(dist.variance, base_dist.variance * 9.)
    assert torch.allclose(dist.entropy(), base_dist.entropy() + math.log(3.))

def test_transformed_beta_bounds_properties():
    base_dist = _Beta(tensor([2., 3.]), tensor([3., 2.]))
    transform = AffineTransform(loc = 1., scale = 3.)
    dist = TransformedBeta(base_dist, transform)

    assert dist.low == 1.0
    assert dist.high == 4.0
    assert dist.bounds == (1.0, 4.0)

def test_transformed_beta_rejects_non_affine():
    with raises(AssertionError):
        TransformedBeta(_Beta(tensor([2.]), tensor([3.])), SigmoidTransform())

# test temperature

def test_temperature():
    beta = Beta()
    params = torch.randn(8, 4, 2)
    action = torch.rand(8, 4)

    sharp = beta(params, temperature = 0.5)
    wide = beta(params, temperature = 2.)
    assert sharp.entropy().mean() < wide.entropy().mean()

    assert torch.allclose(beta.mode(params, temperature = 0.5), sharp.mode)
    assert torch.allclose(beta.entropy(params, temperature = 0.5), sharp.entropy().sum(dim = -1))
    assert torch.allclose(beta.log_prob(params, action, temperature = 0.5), beta.log_prob(sharp, action))

    assert beta.sample(params, (2,), temperature = 0.5).shape == (2, 8, 4)
    assert beta.rsample(params, (2,), temperature = 0.5).shape == (2, 8, 4)

    with raises(AssertionError):
        beta(params, temperature = 0.)

    with raises(AssertionError):
        beta(params, temperature = -1.)

# test 0 to 1 range

def test_unit_range_defaults_and_initialization():
    default_beta = Beta()
    assert default_beta.range == (-1.0, 1.0)
    assert default_beta.val_range == (-1.0, 1.0)
    assert default_beta.loc == -1.0
    assert default_beta.scale == 2.0
    assert exists(default_beta.transform)

    unit_beta = Beta(range = (0, 1))
    assert unit_beta.range == (0.0, 1.0)
    assert unit_beta.val_range == (0.0, 1.0)
    assert unit_beta.loc == 0.0
    assert unit_beta.scale == 1.0
    assert exists(unit_beta.transform)
    assert unit_beta.transform.loc == 0.0
    assert unit_beta.transform.scale == 1.0

    unit_beta_val = Beta(val_range = (0., 1.))
    assert unit_beta_val.range == (0.0, 1.0)
    assert unit_beta_val.val_range == (0.0, 1.0)
    assert exists(unit_beta_val.transform)

    with raises(AssertionError):
        Beta(range = (1, 0))

    with raises(AssertionError):
        Beta(range = (1, 1))

def test_unit_range_samples_and_bounds():
    beta = Beta(range = (0, 1))
    params = torch.randn(8, 4, 2, requires_grad = True)
    dist = beta(params)

    assert isinstance(dist, TransformedBeta)

    # samples in (0, 1)

    actions = dist.sample()
    assert actions.shape == (8, 4)
    assert ((actions >= 0.0) & (actions <= 1.0)).all()

    # mode in (0, 1)

    mode = beta.mode(params)
    assert mode.shape == (8, 4)
    assert ((mode >= 0.0) & (mode <= 1.0)).all()
    assert torch.allclose(mode, dist.mode)

    # mean in (0, 1)

    mean = beta.mean(params)
    assert mean.shape == (8, 4)
    assert ((mean >= 0.0) & (mean <= 1.0)).all()
    assert torch.allclose(mean, dist.mean)

    # rsample gradient backpropagation

    r_act = dist.rsample()
    assert ((r_act >= 0.0) & (r_act <= 1.0)).all()
    loss = r_act.sum()
    loss.backward()
    assert exists(params.grad)
    assert not torch.isnan(params.grad).any()

def test_unit_range_distribution_properties():
    beta = Beta(range = (0, 1))
    params = torch.randn(8, 4, 2)
    dist = beta(params)

    assert isinstance(dist, TransformedBeta)
    assert ((dist.mean >= 0.0) & (dist.mean <= 1.0)).all()
    assert ((dist.mode >= 0.0) & (dist.mode <= 1.0)).all()
    assert (dist.variance >= 0.0).all()
    assert (dist.stddev >= 0.0).all()
    assert not torch.isnan(dist.entropy()).any()

def test_unit_range_entropy_and_log_prob():
    beta = Beta(range = (0, 1))
    params = torch.randn(8, 4, 2)
    dist = beta(params)

    actions = dist.sample()
    assert ((actions >= 0.0) & (actions <= 1.0)).all()

    # log prob matches native Beta log_prob directly (no Jacobian shift)

    lp = beta.log_prob(dist, actions, sum_action_dim = False)
    assert torch.allclose(lp, dist.log_prob(actions), atol = 1e-5)

    lp_sum = beta.log_prob(dist, actions, sum_action_dim = True)
    assert lp_sum.shape == (8,)

    # entropy matches native Beta entropy directly

    ent = beta.entropy(dist, sum_action_dim = False)
    assert torch.allclose(ent, dist.entropy(), atol = 1e-5)

    ent_sum = beta.entropy(dist, sum_action_dim = True)
    assert ent_sum.shape == (8,)

def test_unit_range_temperature():
    beta = Beta(range = (0, 1))
    params = torch.randn(8, 4, 2)
    action = torch.rand(8, 4)

    sharp = beta(params, temperature = 0.5)
    wide = beta(params, temperature = 2.)
    assert sharp.entropy().mean() < wide.entropy().mean()

    assert torch.allclose(beta.mode(params, temperature = 0.5), sharp.mode)
    assert torch.allclose(beta.entropy(params, temperature = 0.5), sharp.entropy().sum(dim = -1))
    assert torch.allclose(beta.log_prob(params, action, temperature = 0.5), beta.log_prob(sharp, action))

    assert beta.sample(params, (2,), temperature = 0.5).shape == (2, 8, 4)
    assert beta.rsample(params, (2,), temperature = 0.5).shape == (2, 8, 4)

def test_arbitrary_range():
    beta = Beta(range = (-2., 2.))
    assert beta.range == (-2.0, 2.0)
    params = torch.randn(8, 4, 2)
    dist = beta(params)
    actions = dist.sample()
    assert ((actions >= -2.0) & (actions <= 2.0)).all()
    assert torch.allclose(beta.mean(params), dist.mean, atol = 1e-5)
    assert torch.allclose(beta.mode(params), dist.mode, atol = 1e-5)
    assert torch.allclose(beta.entropy(dist, sum_action_dim = False), dist.base_dist.entropy() + math.log(4.0), atol = 1e-4)

def test_bounds_alias_and_properties():
    beta = Beta(bounds = (-2., 2.))

    assert beta.bounds == (-2.0, 2.0)
    assert beta.low == -2.0
    assert beta.high == 2.0
    assert beta.val_range == (-2.0, 2.0)
    assert beta.range == (-2.0, 2.0)
    assert beta.loc == -2.0
    assert beta.scale == 4.0

    params = torch.randn(8, 4, 2)
    dist = beta(params)

    actions = dist.sample()
    assert ((actions >= -2.0) & (actions <= 2.0)).all()
    assert dist.bounds == (-2.0, 2.0)
    assert torch.allclose(beta.mean(params), dist.mean, atol = 1e-5)

# test rescale env step with gymnasium

@param('val_range', [(-1., 1.), (0., 1.)])
@param('target_range', [(-2.0, 2.0), 2.0])
def test_rescale_env_step_gym(val_range, target_range):
    env = gym.make('Pendulum-v1')
    env.reset(seed = 42)

    beta = Beta(range = val_range)
    step = beta.rescale_env_step(env.step, target_range = target_range)

    action = beta.sample(torch.randn(1, 2)).numpy()
    obs, reward, terminated, truncated, info = step(action)

    assert obs.shape == (3,)
    assert exists(reward)

    env.close()

def test_rescale_env_step_values():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, target_range = (-0.4, 0.4))
    assert torch.allclose(step(tensor([-1.0, 0.0, 1.0])), tensor([-0.4, 0.0, 0.4]))

def test_rescale_env_step_raw_scale():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, 0.4)
    assert torch.allclose(step(tensor([-1.0, 0.0, 1.0])), tensor([-0.4, 0.0, 0.4]))

def test_rescale_env_step_preconfigured():
    beta = Beta(target_range = (-0.4, 0.4))
    step = beta.rescale_env_step(lambda a: a)
    assert torch.allclose(step(tensor([-1.0, 1.0])), tensor([-0.4, 0.4]))

def test_rescale_env_step_noop_without_target_range():
    beta = Beta()
    identity = lambda a: a
    assert beta.rescale_env_step(identity) is identity

def test_rescale_from_to():
    x = tensor([-1.0, 0.0, 1.0])
    res = rescale_from_to(x, (-1.0, 1.0), (-0.4, 0.4))
    assert torch.allclose(res, tensor([-0.4, 0.0, 0.4]))

    # raw floats
    assert rescale_from_to(-1.0, (-1.0, 1.0), (-0.4, 0.4)) == -0.4
    assert rescale_from_to(1.0, (-1.0, 1.0), (-0.4, 0.4)) == 0.4

# test rescale env step with clipping

def test_rescale_env_step_clipping():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, scale = 1.5, clip = (-1.0, 1.0))
    assert torch.allclose(step(tensor([-1.0, 0.0, 1.0])), tensor([-1.0, 0.0, 1.0]))

def test_rescale_env_step_clip_true():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, scale = 2.0, clip = True)
    assert torch.allclose(step(tensor([-1.0, 0.2, 1.0])), tensor([-1.0, 0.4, 1.0]))

def test_rescale_env_step_clip_scalar():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, scale = 2.0, clip = 0.5)
    assert torch.allclose(step(tensor([-1.0, 0.2, 1.0])), tensor([-0.5, 0.4, 0.5]))

def test_rescale_env_step_clip_numpy():
    import numpy as np
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, scale = 2.0, clip = (-1.0, 1.0))
    res = step(np.array([-1.0, 0.2, 1.0]))
    assert np.allclose(res, np.array([-1.0, 0.4, 1.0]))

def test_rescale_env_step_clip_only():
    beta = Beta()
    step = beta.rescale_env_step(lambda a: a, clip = (-0.5, 0.5))
    assert torch.allclose(step(tensor([-1.0, 0.2, 1.0])), tensor([-0.5, 0.2, 0.5]))

# test detach entropy mean

def test_detach_entropy_mean_default():
    beta = Beta()
    assert beta.detach_entropy_mean

def test_detach_entropy_mean_gradient():
    beta = Beta()
    params = torch.tensor([[0.5, 0.0]], requires_grad = True)
    dist = beta(params)
    ent = dist.entropy()
    ent.backward()
    assert torch.allclose(params.grad[0, 0], torch.tensor(0.0), atol = 1e-6)
    assert not torch.allclose(params.grad[0, 1], torch.tensor(0.0))

    beta_false = Beta(detach_entropy_mean = False)
    params_false = torch.tensor([[0.5, 0.0]], requires_grad = True)
    dist_false = beta_false(params_false)
    ent_false = dist_false.entropy()
    ent_false.backward()
    assert not torch.allclose(params_false.grad[0, 0], torch.tensor(0.0))

def test_detach_entropy_mean_unit_range():
    beta = Beta(range = (0, 1))
    assert beta.detach_entropy_mean
    params = torch.tensor([[0.5, 0.0]], requires_grad = True)
    dist = beta(params)
    ent = dist.entropy()
    ent.backward()
    assert torch.allclose(params.grad[0, 0], torch.tensor(0.0), atol = 1e-6)
    assert not torch.allclose(params.grad[0, 1], torch.tensor(0.0))
