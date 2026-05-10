import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from munkres import Munkres
from copulae.mixtures.gmc.gmc import GaussianMixtureCopula
from main import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch_geometric.utils import negative_sampling, add_self_loops
from torch_geometric.data import Data

from preprocessing import get_device
# Code below is developed and adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE here. We thank for the authors to make it publicly available

device = get_device()

"""
Variational Graph Autoencoder (VGAE) for scRNA-seq Clustering
Encoder: GraphSAGE + GATv2 hybrid (recent, scalable, inductive)
Decoder: Dot-product (standard for latent graph structure)
"""



class VGAEEncoder(nn.Module):
    """
    GraphSAGE  (inductive, scalable local aggregation)
      → GATv2  (attention-weighted, expressive neighbourhood)
      → parallel linear heads → mu, log_std
    """
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        self.sage     = SAGEConv(-1, hidden_dim)
        self.gat      = GATv2Conv(hidden_dim, hidden_dim // heads,
                                  heads=heads, dropout=dropout, concat=True)
        self.norm     = nn.LayerNorm(hidden_dim)
        self.mu_head  = nn.Linear(hidden_dim, latent_dim)
        self.std_head = nn.Linear(hidden_dim, latent_dim)
        self.dropout  = dropout

    def forward(self, x, edge_index):
        h = F.selu(self.sage(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.selu(self.gat(h, edge_index))
        h = self.norm(h)
        return self.mu_head(h), self.std_head(h)


class VGAE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 latent_dim: int = 32, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = VGAEEncoder(in_dim, hidden_dim, latent_dim, heads, dropout)

    def reparameterise(self, mu, log_std):
        if self.training:
            std = torch.exp(log_std.clamp(-10, 10))
            return mu + std * torch.randn_like(std)
        return mu                          # deterministic at eval/inference

    def decode(self, z, edge_index):
        """Dot-product decoder → edge logits."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index):
        mu, log_std = self.encoder(x, edge_index)
        z = self.reparameterise(mu, log_std)
        return z, mu, log_std


# ─────────────────────────────────────────────
# 2. LOSS
# ─────────────────────────────────────────────

def vgae_loss(model: VGAE, z, mu, log_std, edge_index,
              num_nodes: int, beta: float = 1.0) -> dict:
    """
    ELBO  =  Reconstruction (BCE on pos+neg edges)  −  beta * KL

    Returns dict with keys: 'loss', 'recon', 'kl'
    """
    # positive edges
    pos_logits = model.decode(z, edge_index)

    # negative sampling — equal count to positives
    neg_ei = negative_sampling(edge_index, num_nodes=num_nodes,
                               num_neg_samples=edge_index.size(1))
    neg_logits = model.decode(z, neg_ei)

    recon = (
        F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
      + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    )

    # KL:  −½ Σ(1 + 2·log_std − μ² − e^{2·log_std})  normalised by |E|
    kl = (-0.5 * (1 + 2*log_std - mu.pow(2) - (2*log_std).exp())
          .sum(dim=-1).mean() / edge_index.size(1))

    return {"loss": recon + beta * kl, "recon": recon, "kl": kl}


# ─────────────────────────────────────────────
# 3. TRAINING
# ─────────────────────────────────────────────

def train_vgae(model: VGAE, data: Data,
               epochs: int = 200, lr: float = 1e-3,
               beta: float = 1.0, device: str = "auto") -> dict:

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    x     = data.x.to(device)
    ei, _ = add_self_loops(data.edge_index.to(device), num_nodes=x.size(0))

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history = {"loss": [], "recon": [], "kl": []}

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        z, mu, log_std = model(x, ei)
        m = vgae_loss(model, z, mu, log_std, ei, x.size(0), beta)
        m["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        for k in history: history[k].append(m[k].item())
        if epoch % 20 == 0 or epoch == 1:
            print(f"[{epoch:>4}/{epochs}]  loss={m['loss']:.4f}  "
                  f"recon={m['recon']:.4f}  kl={m['kl']:.6f}")

    return history


if __name__ == "__main__":
    N, Fy = 1000, 2000                           # cells × genes (mock)
    data = Data(x=torch.randn(N, Fy),
                edge_index=torch.randint(0, N, (2, 5000)))

    model   = VGAE(in_dim=F, hidden_dim=256, latent_dim=32)
    history = train_vgae(model, data, epochs=100, lr=1e-3, beta=1.0)

    # downstream clustering: use mu (deterministic)
    model.eval()
    with torch.no_grad():
        _, mu, _ = model(data.x, data.edge_index)
    print("Cell embeddings:", mu.shape)         # [N, 32]
