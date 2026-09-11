from __future__ import annotations
import math

import torch
from torch import Tensor, Size, is_tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.distributions import (
    Distribution,
    Beta as _Beta,
    TransformedDistribution,
    AffineTransform
)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# transformed beta distribution on (-1, 1)

class TransformedBeta(TransformedDistribution):
    def __init__(
        self,
        base_dist: Distribution,
        transform: AffineTransform
    ):
        assert isinstance(transform, AffineTransform), 'TransformedBeta only supports affine transforms'
        assert transform.event_dim == 0, 'TransformedBeta only supports non-event affine transforms'

        super().__init__(base_dist, transform)

    @property
    def transform(self) -> AffineTransform:
        return self.transforms[0]

    @property
    def mean(self) -> Tensor:
        return self.transform(self.base_dist.mean)

    @property
    def mode(self) -> Tensor:
        return self.transform(self.base_dist.mode)

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance * self.transform.scale ** 2

    def entropy(self) -> Tensor:
        scale = self.transform.scale
        log_scale = scale.abs().log() if is_tensor(scale) else math.log(abs(scale))
        return self.base_dist.entropy() + log_scale

# beta distribution policy - unimodal mean-concentration reparameterization on (-1, 1),
# an affine shift of a unit-interval beta (y = 2x - 1)

class Beta(Module):
    def __init__(
        self,
        pos_fn = 'exp',
        init_conc = 10.,
        min_conc = 0.,
        eps = 1e-5,
        detach_unimodal = True,
        clamp_exp = (-10., 10.)
    ):
        super().__init__()
        assert pos_fn in ('exp', 'softplus'), f'pos_fn must be either exp or softplus, got {pos_fn}'
        assert min_conc >= 0., f'min_conc must be non-negative, got {min_conc}'
        assert init_conc > min_conc, f'init_conc ({init_conc}) must be greater than min_conc ({min_conc})'

        self.pos_fn = pos_fn
        self.init_conc = init_conc
        self.min_conc = min_conc
        self.eps = eps
        self.detach_unimodal = detach_unimodal

        if isinstance(clamp_exp, (int, float)):
            clamp_exp = (-float(clamp_exp), float(clamp_exp))

        self.clamp_exp = clamp_exp

        # raw offset so concentration at raw_conc = 0 is exactly init_conc

        self.raw_init_conc = math.log(init_conc - min_conc) if pos_fn == 'exp' else math.log(math.expm1(init_conc - min_conc))

    @property
    def has_rsample(self) -> bool:
        return True

    def to_dist(self, params_or_dist: Tensor | Distribution, temperature: float = 1.) -> Distribution:
        return params_or_dist if isinstance(params_or_dist, Distribution) else self(params_or_dist, temperature = temperature)

    def concentration(
        self,
        raw_conc: Tensor
    ) -> Tensor:
        if self.pos_fn == 'exp':
            if exists(self.clamp_exp):
                min_val, max_val = self.clamp_exp
                raw_conc = raw_conc.clamp(min = min_val, max = max_val)

            return (raw_conc + self.raw_init_conc).exp() + self.min_conc

        return F.softplus(raw_conc + self.raw_init_conc) + self.min_conc

    def mean(
        self,
        params: Tensor
    ) -> Tensor:
        raw_mean, _ = params.unbind(dim = -1)
        return raw_mean.tanh().clamp(min = -1. + self.eps, max = 1. - self.eps)

    def mode(
        self,
        params_or_dist: Tensor | Distribution,
        temperature: float = 1.
    ) -> Tensor:
        dist = self.to_dist(params_or_dist, temperature)
        base_dist = getattr(dist, 'base_dist', dist)
        return base_dist.mode * 2. - 1.

    def entropy(
        self,
        params_or_dist: Tensor | Distribution,
        sum_action_dim = True,
        temperature: float = 1.
    ) -> Tensor:
        # shifted beta entropy = base entropy + log(2), the affine jacobian

        dist = self.to_dist(params_or_dist, temperature)
        base_dist = getattr(dist, 'base_dist', dist)
        entropy = base_dist.entropy() + math.log(2.)
        return entropy.sum(dim = -1) if sum_action_dim else entropy

    def log_prob(
        self,
        params_or_dist: Tensor | Distribution,
        action: Tensor,
        sum_action_dim = True,
        eps = None,
        temperature: float = 1.
    ) -> Tensor:
        eps = default(eps, self.eps)
        action = action.clamp(min = -1. + eps, max = 1. - eps)
        dist = self.to_dist(params_or_dist, temperature)
        log_prob = dist.log_prob(action)
        return log_prob.sum(dim = -1) if sum_action_dim else log_prob

    def sample(
        self,
        params: Tensor,
        sample_shape = (),
        temperature: float = 1.
    ) -> Tensor:
        sample_shape = Size(sample_shape)
        return self(params, temperature = temperature).sample(sample_shape)

    def rsample(
        self,
        params: Tensor,
        sample_shape = (),
        temperature: float = 1.
    ) -> Tensor:
        sample_shape = Size(sample_shape)
        return self(params, temperature = temperature).rsample(sample_shape)

    def forward(
        self,
        params: Tensor,
        temperature: float = 1.
    ) -> TransformedBeta:
        assert temperature > 0., f'temperature must be positive, got {temperature}'

        raw_mean, raw_conc = params.unbind(dim = -1)

        # map (-1, 1) mean onto unit interval

        mean = self.mean(params)
        unit_mean = (mean + 1.) / 2.

        # temperature scales the concentration - lower temperature, sharper policy

        conc = self.concentration(raw_conc) / temperature

        # keep the beta unimodal without changing its mean

        min_unit_mean = torch.minimum(unit_mean, 1. - unit_mean).clamp(min = self.eps / 2.)

        if self.detach_unimodal:
            min_unit_mean = min_unit_mean.detach()

        conc = conc + 1. / min_unit_mean

        alpha = unit_mean * conc
        beta = (1. - unit_mean) * conc

        return TransformedBeta(_Beta(alpha, beta), AffineTransform(loc = -1., scale = 2.))
