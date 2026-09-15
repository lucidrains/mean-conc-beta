from __future__ import annotations
import math
from functools import wraps

from numpy import ndarray

import torch
from torch import Tensor, Size, is_tensor, as_tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.distributions import (
    Distribution,
    Beta as _Beta,
    TransformedDistribution,
    AffineTransform
)
from torch.distributions.kl import kl_divergence, register_kl

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def inv_softplus(y: float) -> float:
    # exact inverse of softplus, switching to the asymptotic form for large y where expm1 overflows

    return math.log(math.expm1(y)) if y < 20. else y + math.log(-math.expm1(-y))

def parse_sample_shape(sample_shape) -> Size:
    return Size((sample_shape,) if isinstance(sample_shape, int) else sample_shape)

def clamp(x, low, high):
    if isinstance(x, ndarray):
        return clamp(as_tensor(x), low, high).numpy()

    if not any(map(is_tensor, (x, low, high))):
        return min(max(x, low), high)

    device = next((t.device for t in (x, low, high) if is_tensor(t)), None)
    dtype = next((t.dtype for t in (x, low, high) if is_tensor(t)), None)
    x, low, high = [as_tensor(t, device = device, dtype = dtype) for t in (x, low, high)]

    return x.clamp(min = low, max = high)

# bounds

def parse_bounds(bounds):
    # parses a (low, high) pair or a stack of per-action pairs into a tuple of lows and highs

    if isinstance(bounds, (tuple, list)) and all(map(is_tensor, bounds)):
        bounds = torch.stack(bounds, dim = -1)
    else:
        bounds = as_tensor(bounds)

    assert bounds.ndim in (1, 2) and bounds.shape[-1] == 2, f'bounds must have shape (2,) or (num_actions, 2), got {tuple(bounds.shape)}'

    low, high = bounds.unbind(dim = -1)
    assert (low < high).all(), 'lower bounds must be less than upper bounds'
    return low, high

# linear rescale

def rescale_from_to(
    x,
    from_range = (-1., 1.),
    to_range = (-1., 1.)
):
    if isinstance(x, ndarray):
        return rescale_from_to(as_tensor(x), from_range, to_range).numpy()

    from_low, from_high = parse_bounds(from_range)
    to_low, to_high = parse_bounds(to_range)

    if is_tensor(x):
        device, dtype = x.device, x.dtype

        from_low, from_high, to_low, to_high = [
            as_tensor(t, device = device, dtype = dtype)
            for t in (from_low, from_high, to_low, to_high)
        ]

    norm = (x - from_low) / (from_high - from_low)
    return to_low + norm * (to_high - to_low)

# deterministic distribution for temperature = 0 (greedy action)

class Deterministic(Distribution):
    has_rsample = True
    arg_constraints = {}

    def __init__(self, value: Tensor):
        self.value = value
        super().__init__(batch_shape = value.shape, validate_args = False)

    def rsample(self, sample_shape = ()):
        return self.value.expand(*parse_sample_shape(sample_shape), *self.batch_shape)

    sample = rsample

    def log_prob(self, value: Tensor) -> Tensor:
        raise NotImplementedError('cannot evaluate log_prob for deterministic distribution (temperature = 0)')

    @property
    def mean(self):
        return self.value

    @property
    def mode(self):
        return self.value

    @property
    def variance(self):
        return torch.zeros_like(self.value)

    def entropy(self):
        return torch.zeros_like(self.value)

@register_kl(Deterministic, Deterministic)
def _kl_deterministic_deterministic(p, q):
    return torch.where(p.value == q.value, torch.zeros_like(p.value), torch.full_like(p.value, float('inf')))

# transformed beta distribution

class TransformedBeta(TransformedDistribution):
    def __init__(
        self,
        base_dist: Distribution,
        transform: AffineTransform,
        entropy_base_dist: Distribution | None = None,
        eps: float = 1e-5
    ):
        assert isinstance(transform, AffineTransform), 'TransformedBeta only supports affine transforms'
        assert transform.event_dim == 0, 'TransformedBeta only supports non-event affine transforms'

        self.eps = eps
        self.entropy_base_dist = default(entropy_base_dist, base_dist)
        super().__init__(base_dist, transform)

    def sample(self, sample_shape = ()):
        return super().sample(parse_sample_shape(sample_shape))

    def rsample(self, sample_shape = ()):
        return super().rsample(parse_sample_shape(sample_shape))

    @property
    def transform(self) -> AffineTransform:
        return self.transforms[0]

    @property
    def low(self):
        return self.transform.loc

    @property
    def high(self):
        return self.transform.loc + self.transform.scale

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def scale(self):
        return self.transform.scale

    def log_prob(
        self,
        value: Tensor,
        eps: float | None = None
    ) -> Tensor:
        device, dtype = value.device, value.dtype
        eps = default(eps, self.eps)

        low, high = [
            as_tensor(t, device = device, dtype = dtype)
            for t in self.bounds
        ]

        value = value.clamp(min = low + eps, max = high - eps)
        return super().log_prob(value)

    @property
    def mean(self) -> Tensor:
        return self.transform(self.base_dist.mean)

    @property
    def mode(self) -> Tensor:
        if not isinstance(self.base_dist, _Beta):
            return self.transform(self.base_dist.mode)

        alpha, beta = self.base_dist.concentration1, self.base_dist.concentration0

        # beta mode (alpha - 1) / (alpha + beta - 2), snapping to the nearest bound for J / U shapes

        a, b = (alpha - 1.).clamp(min = 0.), (beta - 1.).clamp(min = 0.)
        total = a + b

        boundary = torch.where(alpha == beta, 0.5, (alpha > beta).type_as(alpha))
        mode = torch.where(total > 0., a / total.masked_fill(total == 0., 1.), boundary)

        return self.transform(mode)

    @property
    def variance(self) -> Tensor:
        return self.base_dist.variance * self.transform.scale ** 2

    def entropy(self) -> Tensor:
        if isinstance(self.base_dist, Deterministic):
            return torch.zeros_like(self.base_dist.value)

        log_scale = as_tensor(self.transform.scale).abs().log()
        return self.entropy_base_dist.entropy() + log_scale

# leaky tanh - exact tanh forward, straight through backward with the gradient floored at `leak`,
# so a mean saturated at a bound can still be pulled back

def leaky_tanh(x: Tensor, leak: float = 0.1) -> Tensor:
    assert 0. <= leak <= 1., f'leak factor must be between 0 and 1, got {leak}'
    tanh_x = x.tanh()
    surrogate = (1. - leak) * tanh_x + leak * x
    return surrogate + (tanh_x - surrogate).detach()

class LeakyTanh(Module):
    def __init__(
        self,
        leak = 0.1
    ):
        super().__init__()
        assert 0. <= leak <= 1., f'leak factor must be between 0 and 1, got {leak}'
        self.leak = leak

    def forward(self, x: Tensor) -> Tensor:
        return leaky_tanh(x, self.leak)

# mean squashing functions - map the raw mean from the real line onto (-1, 1)

SQUASH_FNS = dict(
    tanh = torch.tanh,
    softsign = F.softsign,
    algebraic = lambda x: x / (1. + x ** 2).sqrt(),
    leaky_tanh = LeakyTanh()
)

# beta distribution policy on (-1, 1) or (0, 1), an affine shift of a unit-interval beta,
# parameterized by a mean and concentration, or the two concentrations directly

class Beta(Module):
    def __init__(
        self,
        bounds = None,
        pos_fn = 'exp',
        init_conc = 10.,
        min_conc = 0.,
        eps = 1e-5,
        unimodal: bool | float = True,
        max_unimodal_floor: float | None = 20.,
        detach_unimodal = True,
        detach_entropy_mean = True,
        clamp_exp = (-4., 4.),
        val_range = (-1., 1.),
        range = None,
        target_range = None,
        clamp_log_conc = None,
        squash_fn = 'leaky_tanh',
        param_with_alpha_beta = False
    ):
        super().__init__()

        # bounds and pos_fn can be passed in either order

        if isinstance(bounds, str):
            if isinstance(pos_fn, str):
                pos_fn, bounds = bounds, None
            else:
                pos_fn, bounds = bounds, pos_fn

        assert pos_fn in ('exp', 'softplus'), f'pos_fn must be either exp or softplus, got {pos_fn}'
        assert min_conc >= 0., f'min_conc must be non-negative, got {min_conc}'
        assert init_conc > min_conc, f'init_conc ({init_conc}) must be greater than min_conc ({min_conc})'

        if isinstance(squash_fn, str):
            assert squash_fn in SQUASH_FNS, f'squash_fn must be one of {tuple(SQUASH_FNS.keys())}'
            squash_fn = SQUASH_FNS[squash_fn]

        if isinstance(squash_fn, type) and issubclass(squash_fn, Module):
            squash_fn = squash_fn()

        assert callable(squash_fn), 'squash_fn must be callable'

        self.pos_fn = pos_fn
        self.squash_fn = squash_fn
        self.init_conc = init_conc
        self.min_conc = min_conc
        self.eps = eps
        self.param_with_alpha_beta = param_with_alpha_beta

        # a float shorthand for unimodal sets the damping floor

        if not isinstance(unimodal, bool) and isinstance(unimodal, (int, float)):
            if unimodal <= 0:
                unimodal, max_unimodal_floor = False, None
            else:
                unimodal, max_unimodal_floor = True, float(unimodal)

        self.unimodal = unimodal
        self.max_unimodal_floor = max_unimodal_floor if unimodal else None
        self.detach_unimodal = detach_unimodal
        self.detach_entropy_mean = detach_entropy_mean

        if exists(clamp_log_conc):
            clamp_exp = (-float(clamp_log_conc), float(clamp_log_conc))
        elif isinstance(clamp_exp, (int, float)):
            clamp_exp = (-float(clamp_exp), float(clamp_exp))

        self.clamp_exp = clamp_exp

        # bounds / range

        val_range = default(bounds, default(range, val_range))
        self.low, self.high = parse_bounds(val_range)
        self.loc = self.low
        self.scale = self.high - self.low
        self.target_range = target_range

        # raw offset so concentration at raw_conc = 0 is exactly init_conc

        delta_conc = init_conc - min_conc
        self.raw_init_conc = math.log(delta_conc) if pos_fn == 'exp' else inv_softplus(delta_conc)

    @property
    def range(self):
        return self.bounds

    @property
    def bounds(self):
        return self.low, self.high

    @property
    def clamp_log_conc(self) -> float | None:
        return self.clamp_exp[1] if exists(self.clamp_exp) else None

    # decorate env.step to auto-rescale and clip actions

    def rescale_env_step(
        self,
        step_fn,
        target_range = None,
        clip = None
    ):
        target_range = default(target_range, self.target_range)

        if exists(target_range):
            target_range = parse_bounds(target_range)

        if clip is True:
            clip = target_range if exists(target_range) else self.bounds
        elif clip is False:
            clip = None
        elif exists(clip):
            clip = parse_bounds(clip)

        if not exists(target_range) and not exists(clip):
            return step_fn

        @wraps(step_fn)
        def rescaled_step(action, *args, **kwargs):
            if exists(target_range):
                action = rescale_from_to(action, self.bounds, target_range)

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
        params: Tensor,
        indexed = False
    ) -> Tensor:
        if self.param_with_alpha_beta and not indexed:
            alpha = self.concentration(params[..., 0], indexed = True)
            beta = self.concentration(params[..., 1], indexed = True)
            return alpha + beta

        raw_conc = params if indexed else params[..., 1]

        if self.pos_fn == 'exp':
            if exists(self.clamp_exp):
                min_val, max_val = self.clamp_exp
                raw_conc = raw_conc.clamp(min = min_val, max = max_val)

            return (raw_conc + self.raw_init_conc).exp() + self.min_conc

        elif self.pos_fn == 'softplus':
            return F.softplus(raw_conc + self.raw_init_conc) + self.min_conc

    def mean(
        self,
        params_or_dist: Tensor | Distribution,
        indexed = False
    ) -> Tensor:
        if isinstance(params_or_dist, Distribution):
            return params_or_dist.mean

        if self.param_with_alpha_beta:
            assert not indexed, 'indexed mean is only defined for the mean_conc parameterization'

            alpha = self.concentration(params_or_dist[..., 0], indexed = True)
            beta = self.concentration(params_or_dist[..., 1], indexed = True)
            unit_mean = alpha / (alpha + beta)
        else:
            raw_mean = params_or_dist if indexed else params_or_dist[..., 0]
            unit_mean = (self.squash_fn(raw_mean) + 1.) / 2.

        loc = as_tensor(self.loc, device = params_or_dist.device, dtype = params_or_dist.dtype)
        scale = as_tensor(self.scale, device = params_or_dist.device, dtype = params_or_dist.dtype)

        mean = loc + unit_mean * scale

        # straight through the eps clamp, so a saturated mean keeps its gradient

        clamped = mean.clamp(min = loc + self.eps, max = loc + scale - self.eps)
        return mean + (clamped - mean).detach()

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

    def kl_divergence(
        self,
        params_or_dist_p: Tensor | Distribution,
        params_or_dist_q: Tensor | Distribution,
        sum_action_dim = True,
        temperature: float = 1.
    ) -> Tensor:
        dist_p = self.to_dist(params_or_dist_p, temperature = temperature)
        dist_q = self.to_dist(params_or_dist_q, temperature = temperature)
        kl = kl_divergence(dist_p, dist_q)
        return kl.sum(dim = -1) if sum_action_dim else kl

    kl = kl_divergence

    def log_prob(
        self,
        params_or_dist: Tensor | Distribution,
        action: Tensor,
        sum_action_dim = True,
        eps = None,
        temperature: float = 1.
    ) -> Tensor:
        dist = self.to_dist(params_or_dist, temperature)

        if isinstance(dist, TransformedBeta):
            log_prob = dist.log_prob(action, eps = eps)
        else:
            eps = default(eps, self.eps)
            low, high = [
                as_tensor(b, device = action.device, dtype = action.dtype)
                for b in self.bounds
            ]
            action = action.clamp(min = low + eps, max = high - eps)
            log_prob = dist.log_prob(action)

        return log_prob.sum(dim = -1) if sum_action_dim else log_prob

    def sample(
        self,
        params: Tensor,
        sample_shape = (),
        temperature: float = 1.
    ) -> Tensor:
        return self(params, temperature = temperature).sample(parse_sample_shape(sample_shape))

    def rsample(
        self,
        params: Tensor,
        sample_shape = (),
        temperature: float = 1.
    ) -> Tensor:
        return self(params, temperature = temperature).rsample(parse_sample_shape(sample_shape))

    def forward(
        self,
        params: Tensor,
        temperature: float = 1.,
        detach_entropy_mean: bool | None = None
    ) -> Distribution:
        assert temperature >= 0., f'temperature must be non-negative, got {temperature}'

        detach_entropy_mean = default(detach_entropy_mean, self.detach_entropy_mean)

        loc = as_tensor(self.loc, device = params.device, dtype = params.dtype)
        scale = as_tensor(self.scale, device = params.device, dtype = params.dtype)

        transform = AffineTransform(loc = loc, scale = scale)

        # map mean onto unit interval

        mean = self.mean(params)
        unit_mean = (mean - loc) / scale

        # deterministic greedy action when temperature is 0

        if temperature == 0.:
            return TransformedBeta(Deterministic(unit_mean), transform)

        # temperature scales the concentration - lower temperature, sharper policy

        conc = self.concentration(params) / temperature

        # keep the beta unimodal without changing its mean, with optional damping

        if self.unimodal:
            min_unit_mean = torch.minimum(unit_mean, 1. - unit_mean)

            if self.detach_unimodal:
                min_unit_mean = min_unit_mean.detach()

            floor = 1. / min_unit_mean

            if exists(self.max_unimodal_floor):
                floor = floor.clamp(max = self.max_unimodal_floor)

            conc = conc + floor

        def to_beta(unit_mean, conc):
            return _Beta(unit_mean * conc, (1. - unit_mean) * conc)

        base_dist = to_beta(unit_mean, conc)

        # detach mean so entropy only regularizes concentration without penalizing non-zero mean actions

        entropy_base_dist = to_beta(unit_mean.detach(), conc) if detach_entropy_mean else None

        return TransformedBeta(base_dist, transform, entropy_base_dist = entropy_base_dist, eps = self.eps)
