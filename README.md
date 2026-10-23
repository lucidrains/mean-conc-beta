## mean-conc-beta

Beta distribution parameterized by mean and concentration for bounded continuous action spaces in reinforcement learning.

## Install

```bash
$ pip install mean-conc-beta
```

## Usage

```python
import torch
from mean_conc_beta import Beta

# defaults to (-1., 1.), most continuous action spaces are symmetric centered on 0, sometimes off by a scale
# but you can pass in custom bounds, e.g. Beta((-0.4, 0.4))

# the entropy bonus only regularizes concentration by default, without moving the mean

beta = Beta()

# network output: (batch, num_actions, 2) for raw mean and concentration

params = torch.randn(16, 4, 2, requires_grad = True)

# distribution on the action bounds

dist = beta(params)

# sample actions

actions = dist.sample()
actions_reparam = dist.rsample()

# joint log prob and entropy over the action dimensions

log_prob = dist.log_prob(actions).sum(dim = -1)
entropy = dist.entropy().sum(dim = -1)

# auto-rescale actions directly into env.step, with optional clipping

env_step = beta.rescale_env_step(env.step, target_range = (-2.1, 2.1), clip = (-2., 2.))
next_obs, reward, term, trunc, info = env_step(actions)

# behavior cloning with mse loss on mean

expert_actions = torch.rand(16, 4) * 2. - 1.

pred_mean = beta.mean(params)

bc_loss = (pred_mean - expert_actions).pow(2).mean()
bc_loss.backward()
```

## Greedy actions and uniform prior

```python
# temperature = 0. gives a point mass at the mean, useful for evaluation

action = beta(params, temperature = 0.).mean

# kl against the uniform on the bounds, for prior regularization

prior_loss = beta.kl_uniform(params).mean()
```

## Per-action bounds

Bounds must have shape `(2,)` for a single `(low, high)` pair, or `(num_actions, 2)` for a stack of per-action pairs.

```python
beta = Beta(bounds = [(-1., 1.), (0., 1.), (-2., 2.)])

params = torch.randn(16, 3, 2)
actions = beta(params).sample()
```

## Mean Squashing

The raw mean is squashed onto the bounds with `LeakyTanh` by default - the exact `tanh` forward, with the backward gradient floored at `leak` so a policy saturated at a bound can still be pulled back. It can be swapped for plain `tanh` or any other squash:

```python
from mean_conc_beta import Beta, LeakyTanh

beta = Beta(squash_fn = 'tanh')
beta = Beta(squash_fn = LeakyTanh(leak = 0.1))
beta = Beta(squash_fn = 'softsign')
beta = Beta(squash_fn = lambda x: x / (1. + x ** 2).sqrt())
```

## Mean perturbation

For update-time mean perturbation, as in Robust Policy Optimization, the concentration is held fixed and the mean is shifted by uniform noise in action space:

```python
dist = beta(params, perturb_mean_scale = 0.5)
log_prob = dist.log_prob(actions)
```

The explicit `beta.perturb_mean(params, scale = 0.5)` path also supports `apply_unimodal` and a `generator`.

## Citations

```bibtex
@article{Ferrari2004BetaRF,
    title   = {Beta Regression for Modelling Rates and Proportions},
    author  = {Silvia L. P. Ferrari and Francisco Cribari-Neto},
    journal = {Journal of Applied Statistics},
    year    = {2004},
    volume  = {31},
    pages   = {799 - 815}
}
```

```bibtex
@inproceedings{Chou2017TheBP,
    title   = {The Beta Policy for Continuous Reinforcement Learning},
    author  = {Po-Wei Chou and Daniel Maturana and Sebastian Scherer},
    booktitle = {International Conference on Machine Learning},
    year    = {2017}
}
```

```bibtex
@article{rahman2022robust,
    title   = {Robust Policy Optimization in Deep Reinforcement Learning},
    author  = {Rahman, Md Masudur and Xue, Yexiang},
    journal = {arXiv preprint arXiv:2212.07536},
    year    = {2022}
}
```
