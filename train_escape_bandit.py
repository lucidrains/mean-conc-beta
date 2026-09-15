# /// script
# dependencies = [
#     "fire",
#     "numpy",
#     "rich",
#     "torch>=2.5",
# ]
# ///

# escape bandit - reward depends only on the mean of the beta, target flips to the opposite bound
# halfway through. leaky tanh escapes within the budget, while tanh and the regular alpha / beta
# parameterization stall at the first bound

from __future__ import annotations

import fire
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import torch
from torch import nn
from torch.optim import AdamW

from mean_conc_beta import Beta

# training

def train(
    seed,
    squash_fn = 'leaky_tanh',
    param_with_alpha_beta = False,
    unimodal = True,
    iterations = 300,
    batch_size = 512,
    dim_hidden = 64,
    lr = 3e-3,
    entropy_coef = 0.02,
    detach_entropy_mean = True
):
    torch.manual_seed(seed)

    net = nn.Sequential(
        nn.Linear(1, dim_hidden), nn.Tanh(),
        nn.Linear(dim_hidden, dim_hidden), nn.Tanh(),
        nn.Linear(dim_hidden, 2)
    )

    distr = Beta(
        init_conc = 2.,
        squash_fn = squash_fn,
        param_with_alpha_beta = param_with_alpha_beta,
        unimodal = unimodal,
        detach_entropy_mean = detach_entropy_mean
    )
    optimizer = AdamW(net.parameters(), lr = lr)

    def params(obs):
        return net(obs).view(*obs.shape[:-1], -1, 2)

    # constant observation, only the mean action matters

    obs = torch.zeros(batch_size, 1)

    for iteration in range(iterations):
        target = 1. if iteration < iterations // 2 else -1.

        dist = distr(params(obs))
        action = dist.sample()

        # reward is linear in the mean, so the optimal policy must sit exactly on the target bound

        reward = -(action - target).abs().squeeze(-1)
        advantage = reward - reward.mean()

        loss = -(distr.log_prob(dist, action) * advantage).mean() - entropy_coef * distr.entropy(dist).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        return distr.mean(params(obs[:1])).item()

# main

def main(
    seeds: int = 10,
    iterations: int = 300,
    batch_size: int = 512,
    lr: float = 3e-3,
    entropy_coef: float = 0.02,
    detach_entropy_mean: bool = True
):
    console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold]Escape Bandit[/bold]\n\n"
        "A policy is trained to sit at the [cyan]+1.0[/cyan] bound until saturated.\n"
        "Halfway through, the target abruptly flips to the opposite bound ([cyan]-1.0[/cyan]).\n\n"
        "• [yellow]tanh[/yellow]: vanishing gradients (sech² ≈ 0) at the boundary leave the policy trapped.\n"
        "• [green]leaky_tanh[/green]: straight-through gradient floor allows the policy to escape to -1.0.\n"
        "• [red]alpha_beta[/red]: no separate mean parameter, so the saturated mean cannot be pulled back.",
        title = "Experiment",
        border_style = "bright_blue"
    ))
    console.print()

    variants = dict(
        tanh = dict(squash_fn = 'tanh'),
        leaky_tanh = dict(squash_fn = 'leaky_tanh'),
        alpha_beta = dict(param_with_alpha_beta = True)
    )

    results = {
        name: [
            train(
                seed,
                iterations = iterations,
                batch_size = batch_size,
                lr = lr,
                entropy_coef = entropy_coef,
                detach_entropy_mean = detach_entropy_mean,
                **kwargs
            )
            for seed in range(seeds)
        ]
        for name, kwargs in variants.items()
    }

    table = Table(title = f"Escape Rate ({seeds} Seeds)", border_style = "bright_black")
    table.add_column("Variant", style = "bold")
    table.add_column("Escaped", justify = "right")
    table.add_column("Trapped", justify = "right")

    for name in variants:
        escaped = sum(1 for m in results[name] if m < 0.)
        escape_pct = (escaped / seeds) * 100.
        trapped_pct = 100. - escape_pct

        escape_str = f"[green]{escape_pct:.0f}%[/green]" if escape_pct > 50 else f"[red]{escape_pct:.0f}%[/red]"
        trapped_str = f"[red]{trapped_pct:.0f}%[/red]" if trapped_pct > 50 else f"[green]{trapped_pct:.0f}%[/green]"

        table.add_row(name, escape_str, trapped_str)

    console.print(table)
    console.print()

if __name__ == '__main__':
    fire.Fire(main)
