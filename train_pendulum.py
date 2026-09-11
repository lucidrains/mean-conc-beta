from __future__ import annotations
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW

import gymnasium as gym
from tqdm import tqdm

from mean_conc_beta import Beta

# networks

class Actor(nn.Module):
    def __init__(
        self,
        dim_state = 3,
        dim_action = 1,
        dim_hidden = 64,
        detach_entropy_mean = True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_action * 2)
        )
        self.distr = Beta(init_conc = 2., detach_entropy_mean = detach_entropy_mean)

    def forward(self, x):
        params = self.net(x).view(*x.shape[:-1], -1, 2)
        return self.distr(params)

class Critic(nn.Module):
    def __init__(
        self,
        dim_state = 3,
        dim_hidden = 64
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# evaluation

def evaluate(actor, env, action_scale, num_episodes = 5):
    scores = []
    eval_step = actor.distr.rescale_env_step(env.step, action_scale)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed = 1000 + ep)
        ep_return = 0.

        for _ in range(200):
            with torch.no_grad():
                params = actor.net(torch.from_numpy(obs).float()).view(1, 2)
                action = actor.distr.mean(params).numpy()

            obs, reward, terminated, truncated, _ = eval_step(action)
            ep_return += reward

            if terminated or truncated:
                break

        scores.append(ep_return)

    return float(np.mean(scores))

# ppo training sanity check

def train_pendulum(
    detach_entropy_mean = True,
    entropy_coef = 0.1,
    max_iterations = 25
):
    torch.manual_seed(42)
    np.random.seed(42)

    num_envs = 16
    env = gym.make_vec('Pendulum-v1', num_envs = num_envs)
    eval_env = gym.make('Pendulum-v1')
    action_scale = float(env.single_action_space.high[0])

    actor = Actor(detach_entropy_mean = detach_entropy_mean)
    critic = Critic()
    optimizer = AdamW([*actor.parameters(), *critic.parameters()], lr = 1.5e-3)

    train_step = actor.distr.rescale_env_step(env.step, action_scale)

    obs, _ = env.reset(seed = 42)
    rollout_len = 200
    score = -1200.

    pbar = tqdm(range(max_iterations), desc = 'training pendulum')

    for iteration in pbar:
        obs_buf, action_buf, log_prob_buf, val_buf, reward_buf, done_buf = [], [], [], [], [], []

        for _ in range(rollout_len):
            obs_tensor = torch.from_numpy(obs).float()

            with torch.no_grad():
                dist = actor(obs_tensor)
                action = dist.sample()
                log_prob = actor.distr.log_prob(dist, action)
                value = critic(obs_tensor)

            obs_buf.append(obs)
            action_buf.append(action.numpy())
            log_prob_buf.append(log_prob.numpy())
            val_buf.append(value.numpy())

            next_obs, rewards, terms, truncs, _ = train_step(action.numpy())

            reward_buf.append(rewards * 0.1)
            done_buf.append(terms)

            obs = next_obs

        # generalized advantage estimation (gae)

        with torch.no_grad():
            last_value = critic(torch.from_numpy(obs).float()).numpy()

        all_values = [*val_buf, last_value]
        advantages = np.zeros((rollout_len, num_envs), dtype = np.float32)
        last_gae = np.zeros(num_envs, dtype = np.float32)

        for t in reversed(range(rollout_len)):
            non_terminal = 1. - done_buf[t].astype(np.float32)
            delta = reward_buf[t] + 0.99 * all_values[t + 1] * non_terminal - all_values[t]
            last_gae = delta + 0.99 * 0.95 * last_gae * non_terminal
            advantages[t] = last_gae

        returns = advantages + np.array(val_buf)

        # batch update

        flat_obs = torch.from_numpy(np.array(obs_buf).reshape(-1, 3)).float()
        flat_actions = torch.from_numpy(np.array(action_buf).reshape(-1, 1)).float()
        flat_log_probs = torch.from_numpy(np.array(log_prob_buf).reshape(-1)).float()
        flat_advantages = torch.from_numpy(advantages.reshape(-1)).float()
        flat_returns = torch.from_numpy(returns.reshape(-1)).float()

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        for _ in range(4):
            perm = torch.randperm(flat_obs.shape[0])

            for start in range(0, flat_obs.shape[0], 64):
                idx = perm[start:start + 64]

                dist = actor(flat_obs[idx])
                new_log_probs = actor.distr.log_prob(dist, flat_actions[idx])

                ratio = (new_log_probs - flat_log_probs[idx]).exp()
                surr1 = ratio * flat_advantages[idx]
                surr2 = ratio.clamp(0.8, 1.2) * flat_advantages[idx]

                policy_loss = -torch.min(surr1, surr2).mean() - entropy_coef * actor.distr.entropy(dist).mean()
                value_loss = 0.5 * ((critic(flat_obs[idx]) - flat_returns[idx]) ** 2).mean()

                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        score = evaluate(actor, eval_env, action_scale)
        pbar.set_description(f'iter {iteration:2d} | eval {score:.1f}')

        if score >= -250.:
            break

    env.close()
    eval_env.close()

    if detach_entropy_mean:
        assert score >= -250., f'expected pendulum to balance upright, final eval: {score}'
        print(f'pendulum converged successfully with detached entropy mean (score: {score:.1f})')
    else:
        print(f'pendulum failed to converge without detaching entropy mean (final score: {score:.1f})')

    return score

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = 'train pendulum with beta policy')
    parser.add_argument('--ablate', '--no-detach', dest = 'ablate', action = 'store_true', help = 'ablate detaching entropy mean to show failure to converge')
    args = parser.parse_args()

    train_pendulum(detach_entropy_mean = not args.ablate)


