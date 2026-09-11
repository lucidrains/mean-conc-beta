from __future__ import annotations
import math
import pytest
from pytest import raises
param = pytest.mark.parametrize

import torch
from torch import tensor

from mean_conc_beta import Beta, exists

# test mean preservation

@param('pos_fn', ['exp', 'softplus'])
@param('detach_unimodal', [True, False])
def test_mean_preservation(pos_fn, detach_unimodal):
    beta = Beta(pos_fn = pos_fn, init_conc = 10.0, detach_unimodal = detach_unimodal)

    for target_m in [-0.8, -0.5, 0.0, 0.5, 0.8]:
        raw_m = math.atanh(target_m)
        params = tensor([[raw_m, 0.0]])
        dist = beta(params)

        true_mean = dist.base_dist.mean.item() * 2. - 1.
        det_mean = beta.mean(params).item()

        assert abs(true_mean - target_m) < 1e-5
        assert abs(det_mean - target_m) < 1e-5

# test unimodality under extreme latents

@param('pos_fn', ['exp', 'softplus'])
def test_unimodality_extreme_latents(pos_fn):
    beta = Beta(pos_fn = pos_fn)

    # extreme negative and positive raw means

    params = tensor([[[-100.0, 0.0], [100.0, 0.0]], [[0.0, -10.0], [0.0, 10.0]]])
    dist = beta(params)

    alpha = dist.base_dist.concentration1
    beta_param = dist.base_dist.concentration0

    assert (alpha > 1.0).all()
    assert (beta_param > 1.0).all()

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
    beta = Beta(init_conc = 10.0, clamp_exp = (-10.0, 10.0))

    # raw_conc = 0 preserves exact init_conc

    assert abs(beta.concentration(tensor(0.0)).item() - 10.0) < 1e-5

    # raw_conc = 100 is clamped to 10.0, yielding init_conc * exp(10.0)

    conc_huge = beta.concentration(tensor(100.0)).item()
    expected_huge = 10.0 * math.exp(10.0)
    assert abs(conc_huge - expected_huge) / expected_huge < 1e-4

    # raw_conc = -100 is clamped to -10.0, yielding init_conc * exp(-10.0)

    conc_tiny = beta.concentration(tensor(-100.0)).item()
    expected_tiny = 10.0 * math.exp(-10.0)
    assert abs(conc_tiny - expected_tiny) / expected_tiny < 1e-4

    beta_custom = Beta(clamp_exp = 5.0)
    assert beta_custom.clamp_exp == (-5.0, 5.0)

def test_init_conc_not_overridden():
    beta_large = Beta(init_conc = 1e6)
    conc_at_zero = beta_large.concentration(tensor(0.0)).item()
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
