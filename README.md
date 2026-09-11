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

beta = Beta()

# network output: (batch, num_actions, 2) for raw mean and concentration

params = torch.randn(16, 4, 2, requires_grad = True)

# distribution on (-1, 1)

dist = beta(params)

# sample actions

actions = dist.sample()
actions_reparam = dist.rsample()

# log prob and entropy

log_prob = beta.log_prob(dist, actions)
entropy = beta.entropy(dist)

# behavior cloning with mse loss on mean

expert_actions = torch.rand(16, 4)

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
