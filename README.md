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
# but you can pass in the custom bounds of the action space

beta = Beta(bounds = (-2., 2.))

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

expert_actions = torch.rand(16, 4) * 4. - 2.

pred_mean = beta.mean(params)

bc_loss = (pred_mean - expert_actions).pow(2).mean()
bc_loss.backward()
```

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
