from __future__ import annotations
from functools import wraps
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

from torch_einops_utils import clamp

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# linear rescale

def rescale_from_to(
    x,
    from_range = (-1., 1.),
    to_range = (-1., 1.)
):
    from_low, from_high = from_range
    to_low, to_high = to_range

    if is_tensor(x):
        from_low, from_high, to_low, to_high = [
            torch.as_tensor(t, device = x.device, dtype = x.dtype)
            for t in (from_low, from_high, to_low, to_high)
        ]

    norm = (x - from_low) / (from_high - from_low)
    return to_low + norm * (to_high - to_low)

# transformed beta distribution

class TransformedBeta(TransformedDistribution):
    def __init__(
        self,
        base_dist: Distribution,
        transform: AffineTransform,
        entropy_base_dist: Distribution | None = None
    ):
        assert isinstance(transform, AffineTransform), 'TransformedBeta only supports affine transforms'
        assert transform.event_dim == 0, 'TransformedBeta only supports non-event affine transforms'

        self.entropy_base_dist = default(entropy_base_dist, base_dist)
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
        return self.entropy_base_dist.entropy() + log_scale

# beta distribution policy - unimodal mean-concentration reparameterization on (-1, 1) or (0, 1),
# an affine shift of a unit-interval beta

class Beta(Module):
    def __init__(
        self,
        pos_fn = 'exp',
        init_conc = 10.,
        min_conc = 0.,
        eps = 1e-5,
        detach_unimodal = True,
        detach_entropy_mean = True,
        clamp_exp = (-4., 4.),
        val_range = (-1., 1.),
        range = None,
        target_range = None
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
        self.detach_entropy_mean = detach_entropy_mean

        if isinstance(clamp_exp, (int, float)):
            clamp_exp = (-float(clamp_exp), float(clamp_exp))

        self.clamp_exp = clamp_exp

        # range

        val_range = default(range, val_range)
        min_val, max_val = map(float, val_range)
        assert min_val < max_val, f'val_range min ({min_val}) must be less than max ({max_val})'

        self.val_range = (min_val, max_val)
        self.min_val = min_val
        self.max_val = max_val
        self.loc = min_val
        self.scale = max_val - min_val
        self.affine_transform = AffineTransform(loc = self.loc, scale = self.scale)
        self.target_range = target_range

        # raw offset so concentration at raw_conc = 0 is exactly init_conc

        self.raw_init_conc = math.log(init_conc - min_conc) if pos_fn == 'exp' else math.log(math.expm1(init_conc - min_conc))

    @property
    def range(self) -> tuple[float, float]:
        return self.val_range

    @property
    def transform(self) -> AffineTransform:
        return self.affine_transform

    @property
    def has_rsample(self) -> bool:
        return True

    # decorate env.step to auto-rescale and clip actions

    def rescale_env_step(
        self,
        step_fn,
        target_range = None,
        scale = None,
        clip = None
    ):
        target_range = default(target_range, default(scale, self.target_range))

        if clip is True:
            clip = self.val_range
        elif isinstance(clip, (int, float)):
            clip = (-clip, clip)

        if not exists(target_range) and not exists(clip):
            return step_fn

        @wraps(step_fn)
        def rescaled_step(action, *args, **kwargs):
            if isinstance(target_range, (int, float)):
                action = action * target_range
            elif exists(target_range):
                action = rescale_from_to(action, self.val_range, target_range)

            if exists(clip):
                action = clamp(action, *clip)

            return step_fn(action, *args, **kwargs)

        return rescaled_step

    rescale_step = rescale_env_step

    def to_dist(
        self,
        params_or_dist: Tensor | Distribution,
        temperature: float = 1.,
        detach_entropy_mean: bool | None = None
    ) -> Distribution:
        return params_or_dist if isinstance(params_or_dist, Distribution) else self(params_or_dist, temperature = temperature, detach_entropy_mean = detach_entropy_mean)

    def concentration(
        self,
        raw_conc: Tensor
    ) -> Tensor:
        if self.pos_fn == 'exp':
            if exists(self.clamp_exp):
                min_val, max_val = self.clamp_exp
                raw_conc = raw_conc.clamp(min = min_val, max = max_val)

            return (raw_conc + self.raw_init_conc).exp() + self.min_conc

        elif self.pos_fn == 'softplus':
            return F.softplus(raw_conc + self.raw_init_conc) + self.min_conc

    def mean(
        self,
        params: Tensor
    ) -> Tensor:
        raw_mean, _ = params.unbind(dim = -1)
        tanh_mean = raw_mean.tanh()

        if self.val_range == (-1., 1.):
            return tanh_mean.clamp(min = -1. + self.eps, max = 1. - self.eps)

        unit_mean = (tanh_mean + 1.) / 2.
        mean = unit_mean * self.scale + self.loc
        return mean.clamp(min = self.min_val + self.eps, max = self.max_val - self.eps)

    def mode(
        self,
        params_or_dist: Tensor | Distribution,
        temperature: float = 1.
    ) -> Tensor:
        dist = self.to_dist(params_or_dist, temperature)
        return dist.mode

    def entropy(
        self,
        params_or_dist: Tensor | Distribution,
        sum_action_dim = True,
        temperature: float = 1.,
        detach_mean: bool | None = None
    ) -> Tensor:
        dist = self.to_dist(params_or_dist, temperature = temperature, detach_entropy_mean = detach_mean)
        entropy = dist.entropy()
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
        action = action.clamp(min = self.min_val + eps, max = self.max_val - eps)
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
        temperature: float = 1.,
        detach_entropy_mean: bool | None = None
    ) -> Distribution:
        assert temperature > 0., f'temperature must be positive, got {temperature}'

        detach_entropy_mean = default(detach_entropy_mean, self.detach_entropy_mean)

        raw_mean, raw_conc = params.unbind(dim = -1)

        # map mean onto unit interval

        mean = self.mean(params)
        unit_mean = (mean - self.loc) / self.scale

        # temperature scales the concentration - lower temperature, sharper policy

        conc = self.concentration(raw_conc) / temperature

        # keep the beta unimodal without changing its mean

        min_unit_mean = torch.minimum(unit_mean, 1. - unit_mean).clamp(min = self.eps / 2.)

        if self.detach_unimodal:
            min_unit_mean = min_unit_mean.detach()

        conc = conc + 1. / min_unit_mean

        def to_beta(unit_mean, conc):
            return _Beta(unit_mean * conc, (1. - unit_mean) * conc)

        base_dist = to_beta(unit_mean, conc)

        if not detach_entropy_mean:
            return TransformedBeta(base_dist, self.transform)

        # detach mean so entropy only regularizes concentration without penalizing non-zero mean actions

        entropy_base_dist = to_beta(unit_mean.detach(), conc)

        return TransformedBeta(base_dist, self.transform, entropy_base_dist = entropy_base_dist)
