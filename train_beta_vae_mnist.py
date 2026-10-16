# a minimal mnist vae with a beta posterior on (-1, 1) and the uniform prior, trained with the
# importance weighted bound (iwae, burda et al.) - needs einops, torch-einops-utils, torchvision, umap, matplotlib

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from einops import rearrange, repeat, einsum
from torch_einops_utils.shape import size, assert_shape

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import umap
from PIL import Image

from mean_conc_beta import Beta

torch.manual_seed(0)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
root = Path(__file__).parent
latent_dim, hidden, batch_size, epochs = 8, 256, 128, 10

# candidates explored per image, 1 reduces to the single sample bound

num_candidates = 8

# uniform prior on (-1, 1), a constant log density

log_prior = -latent_dim * math.log(2.)

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim * 2))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, 28 * 28))
        self.beta = Beta()

    @assert_shape('b p')
    def posterior(self, x: Tensor):
        return self.beta(rearrange(self.encoder(x), 'b (d p) -> b d p', p = 2))

vae = VAE().to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr = 3e-4)

# download both mnist splits into ./data if not already present

train_data = MNIST(root / 'data', train = True, download = True, transform = ToTensor())
test_data = MNIST(root / 'data', train = False, download = True, transform = ToTensor())

loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)

for epoch in range(1, epochs + 1):
    vae.train()

    for images, _ in loader:
        images = rearrange(images.to(device), 'b 1 h w -> b (h w)')
        pixels = size(images, 'b [p]')

        dist = vae.posterior(images)

        # log w = log p(x | z) + log p(z) - log q(z | x), then the log mean exp over the candidates

        z = dist.rsample((num_candidates,))
        assert_shape(z, 'k b d', k = num_candidates, d = latent_dim)

        logits = rearrange(vae.decoder(rearrange(z, 'k b d -> (k b) d')), '(k b) p -> k b p', k = num_candidates)

        log_px = -F.binary_cross_entropy_with_logits(logits, repeat(images, 'b p -> k b p', k = num_candidates), reduction = 'none')
        log_px = einsum(log_px, 'k b p -> k b')
        log_q = einsum(dist.log_prob(z), 'k b d -> k b')

        log_w = log_prior + log_px - log_q
        loss = -(torch.logsumexp(log_w, dim = 0) - math.log(num_candidates)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    best_recon = -log_px.max(dim = 0).values.mean().item()
    print(f'epoch {epoch}: bound {-loss.item() / pixels:.3f} nats/pixel, best recon {best_recon / pixels:.3f}')

# umap of the posterior means for 5000 test images

vae.eval()

images, labels = next(iter(DataLoader(test_data, batch_size = 5000)))

with torch.no_grad():
    images = rearrange(images.to(device), 'b 1 h w -> b (h w)')
    z = vae.posterior(images).mean.cpu().numpy()

embedding = umap.UMAP(n_neighbors = 15, min_dist = 0.1, random_state = 0).fit_transform(z)

fig, ax = plt.subplots(figsize = (9, 9))
scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c = labels.numpy(), cmap = 'tab10', s = 4, alpha = 0.8, vmin = -0.5, vmax = 9.5)
ax.set_title('beta vae mnist latent space (umap of the posterior means)')
ax.axis('off')
fig.colorbar(scatter, ax = ax, ticks = range(10), label = 'digit')

figure_path = root / 'beta_vae_mnist_umap.png'
fig.savefig(figure_path, dpi = 150, bbox_inches = 'tight')

# open the umap in the default image viewer

Image.open(figure_path).show()

print('saved', figure_path)
