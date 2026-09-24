# /// script
# dependencies = [
#     "fire",
#     "gymnasium[mujoco]",
#     "memmap-replay-buffer",
#     "numpy",
#     "torch>=2.5",
#     "x-ppo",
# ]
# ///

from __future__ import annotations
import tempfile
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW

import gymnasium as gym
import fire

from memmap_replay_buffer import ReplayBuffer
from x_ppo import calc_gae, dones_to_masks, ppo_actor_loss

from mean_conc_beta import Beta

# networks

class Actor(nn.Module):
    def __init__(
        self,
        bounds = (-1., 1.),
        pos_fn = 'softplus',
        dim_state = 4,
        dim_action = 1,
        dim_hidden = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_action * 2)
        )
        self.distr = Beta(bounds = bounds, pos_fn = pos_fn)

    def forward(self, x):
        params = self.net(x).view(*x.shape[:-1], -1, 2)
        return self.distr(params)

class Critic(nn.Module):
    def __init__(
        self,
        dim_state = 4,
        dim_hidden = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# evaluation

def evaluate(actor, env, num_episodes = 5):
    scores = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = 1000 + ep)
        ep_return = 0.

        for _ in range(1000):
            with torch.no_grad():
                params = actor.net(torch.from_numpy(obs).float()).view(1, 2)
                action = actor.distr.mean(params).numpy()

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward

            if terminated or truncated:
                break

        scores.append(ep_return)

    return float(np.mean(scores))

# ppo training sanity check

def train_inverted_pendulum(
    pos_fn: str = 'softplus',
    num_envs: int = 8,
    rollout_len: int = 256,
    batch_size: int = 64,
    epochs: int = 4,
    max_iterations: int = 20,
    lr: float = 3e-4,
    entropy_coef: float = 0.005,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make_vec('InvertedPendulum-v5', num_envs = num_envs)
    eval_env = gym.make('InvertedPendulum-v5')

    bounds = (float(env.single_action_space.low[0]), float(env.single_action_space.high[0]))

    actor = Actor(bounds = bounds, pos_fn = pos_fn)
    critic = Critic()
    optimizer = AdamW([*actor.parameters(), *critic.parameters()], lr = lr)

    temp_dir = tempfile.TemporaryDirectory()
    buffer = ReplayBuffer(
        temp_dir.name,
        max_episodes = num_envs,
        max_timesteps = rollout_len,
        fields = dict(
            obs = ('float', 4),
            action = ('float', 1),
            log_prob = 'float',
            advantage = 'float',
            returns = 'float'
        ),
        circular = True,
        overwrite = True
    )

    obs, _ = env.reset(seed = seed)
    score = 0.

    obs_seq = torch.zeros(rollout_len, num_envs, 4)
    action_seq = torch.zeros(rollout_len, num_envs, 1)
    log_prob_seq = torch.zeros(rollout_len, num_envs)
    val_seq = torch.zeros(rollout_len, num_envs)
    reward_seq = torch.zeros(rollout_len, num_envs)
    term_seq = torch.zeros(rollout_len, num_envs, dtype = torch.bool)
    trunc_seq = torch.zeros(rollout_len, num_envs, dtype = torch.bool)

    for iteration in range(max_iterations):
        for step in range(rollout_len):
            obs_tensor = torch.from_numpy(obs).float()

            with torch.no_grad():
                dist = actor(obs_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim = -1)
                value = critic(obs_tensor)

            next_obs, rewards, terms, truncs, _ = env.step(action.numpy())

            obs_seq[step] = obs_tensor
            action_seq[step] = action
            log_prob_seq[step] = log_prob
            val_seq[step] = value
            reward_seq[step] = torch.from_numpy(rewards)
            term_seq[step] = torch.from_numpy(terms)
            trunc_seq[step] = torch.from_numpy(truncs)

            obs = next_obs

        # generalized advantage estimation (gae)

        with torch.no_grad():
            last_value = critic(torch.from_numpy(obs).float())

        rewards = reward_seq.T
        values = val_seq.T
        terms = term_seq.T
        truncs = trunc_seq.T

        term_mask, done_mask = dones_to_masks(terms, truncs)
        returns, advantages = calc_gae(rewards, values, masks = term_mask, done_masks = done_mask, next_value = last_value, return_advantages = True)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        buffer.clear()
        for env_idx in range(num_envs):
            buffer.store_episode(
                obs = obs_seq[:, env_idx],
                action = action_seq[:, env_idx],
                log_prob = log_prob_seq[:, env_idx],
                advantage = advantages[env_idx],
                returns = returns[env_idx]
            )

        # batch update with replay buffer dataloader

        for _ in range(epochs):
            dataloader = buffer.dataloader(
                batch_size = batch_size,
                timestep_level = True,
                to_named_tuple = ('obs', 'action', 'log_prob', 'advantage', 'returns')
            )

            for batch in dataloader:
                dist = actor(batch.obs)
                new_log_probs = dist.log_prob(batch.action).sum(dim = -1)

                policy_loss = ppo_actor_loss(new_log_probs, batch.log_prob, batch.advantage).mean() - entropy_coef * dist.entropy().sum(dim = -1).mean()
                value_loss = 0.5 * ((critic(batch.obs) - batch.returns) ** 2).mean()

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        score = evaluate(actor, eval_env)

        if score >= 100.:
            break

    env.close()
    eval_env.close()
    temp_dir.cleanup()

    assert score >= 100., f'expected inverted pendulum to balance, final eval: {score}'
    print(f'inverted pendulum sanity check passed with {pos_fn} (score: {score:.1f})')
    return score

if __name__ == '__main__':
    fire.Fire(train_inverted_pendulum)
