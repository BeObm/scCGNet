from collections import defaultdict
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import math
import numpy as np
import torch
from torch.optim import Adam, SGD, RMSprop
from sklearn import metrics
from munkres import Munkres
from preprocessing import get_device
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import csv, os
from torch_geometric.nn import GCNConv as gnc_encoder
from torch import Tensor
from torch_geometric.typing import Adj



class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels,
                 activation=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.activation    = activation
        self.dropout       = nn.Dropout(p=dropout)
        self.conv_hidden_1 = gnc_encoder(in_channels,     hidden_channels)
        self.conv_hidden_2 = gnc_encoder(hidden_channels, hidden_channels // 2)  # derived
        self.conv_mu       = gnc_encoder(hidden_channels // 2, latent_channels)
        self.conv_logvar   = gnc_encoder(hidden_channels // 2, latent_channels)

    def _encode(self, x, edge_index):
        h = self.dropout(self.activation(self.conv_hidden_1(x, edge_index)))
        h = self.dropout(self.activation(self.conv_hidden_2(h, edge_index)))
        return h

    def forward(self, x, edge_index):
        h = self._encode(x, edge_index)
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)


class ZINBDecoder(nn.Module):
    """
    Decodes a latent representation z back into ZINB distribution parameters
    that reconstruct the original data.x.

    The Zero-Inflated Negative Binomial (ZINB) distribution is parameterized by:
        - pi    (π): zero-inflation probability     → sigmoid activation
        - mu    (μ): negative binomial mean         → softmax activation (scaled)
        - theta (θ): negative binomial dispersion   → softplus activation (strictly positive)

    Args:
        latent_channels:  Dimensionality of the encoder output z.
        hidden_channels:  Dimensionality of the shared hidden layer.
        out_channels:     Number of output features (must match data.x.shape[1]).
        dropout:          Dropout probability in the hidden layer (0 = disabled).
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        out_channels:    int,
        dropout:         float = 0.0,
    ) -> None:
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )

        self.head_pi    = nn.Linear(hidden_channels, out_channels)
        self.head_mu    = nn.Linear(hidden_channels, out_channels)
        self.head_theta = nn.Linear(hidden_channels, out_channels)

    def forward(self, z: Tensor) :
        h     = self.shared(z)
        pi    = torch.sigmoid(self.head_pi(h))                    # ∈ (0, 1)
        mu    = torch.softmax(self.head_mu(h), dim=-1) * z.shape[0]  # ∈ (0, N), scaled mean
        theta = torch.nn.functional.softplus(self.head_theta(h))  # ∈ (0, ∞)
        return pi, mu, theta


def zinb_loss(
    x:     Tensor,
    pi:    Tensor,
    mu:    Tensor,
    theta: Tensor,
    eps:   float = 1e-8,
) -> Tensor:
    """
    Computes the ZINB negative log-likelihood loss.

    The ZINB pmf mixes a point mass at zero with a Negative Binomial:

        P(x=0)  = π + (1-π) · NB(0 | μ, θ)
        P(x>0)  = (1-π)     · NB(x | μ, θ)

    where NB is parameterized by mean μ and dispersion θ:

        NB(x|μ,θ) = Γ(x+θ) / [Γ(θ)·x!] · (θ/(θ+μ))^θ · (μ/(θ+μ))^x

    Args:
        x:     Observed counts of shape (N, G).
        pi:    Zero-inflation probabilities, shape (N, G), ∈ (0,1).
        mu:    NB mean,       shape (N, G), > 0.
        theta: NB dispersion, shape (N, G), > 0.
        eps:   Small constant for numerical stability.

    Returns:
        Scalar mean negative log-likelihood.
    """
    # --- log NB probability at x=0 -------------------------------------------
    # log NB(0|μ,θ) = θ · log(θ/(θ+μ))
    log_nb_zero = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))

    # --- log NB probability at x>0 -------------------------------------------
    # log NB(x|μ,θ) = lgamma(x+θ) - lgamma(θ) - lgamma(x+1)
    #               + θ·log(θ/(θ+μ)) + x·log(μ/(θ+μ))
    log_nb_x = (
        torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        + x     * (torch.log(mu    + eps) - torch.log(theta + mu + eps))
    )

    # --- mix zero-inflation and NB -------------------------------------------
    # For x == 0: log[ π + (1-π)·NB(0) ]
    # For x >  0: log[ (1-π)·NB(x)     ]
    log_pi     = torch.log(pi + eps)
    log_1m_pi  = torch.log(1.0 - pi + eps)

    zero_case    = torch.logaddexp(log_pi, log_1m_pi + log_nb_zero)
    nonzero_case = log_1m_pi + log_nb_x

    nll = -torch.where(x < eps, zero_case, nonzero_case)
    return nll.mean()


class AdjDecoder(nn.Module):
    """
    Reconstructs the adjacency matrix from a latent representation z
    via an optional MLP projection followed by a scaled inner product.

    The inner product approach is grounded in VGAE (Kipf & Welling, 2016):

        Â = sigmoid(z_proj @ z_proj^T)

    where each entry Â[i,j] ∈ (0,1) is the predicted probability of an
    edge between nodes i and j.

    Args:
        latent_channels:  Dimensionality of z (encoder output).
        hidden_channels:  Hidden dim of the projection MLP (None = skip MLP,
                          use z directly for the inner product).
        dropout:          Dropout probability inside the MLP (0 = disabled).
    """

    def __init__(
        self,
        latent_channels: int,
        hidden_channels: int,
        dropout:         float      = 0.0,
    ) -> None:
        super().__init__()

        if hidden_channels is not None:
            self.mlp = nn.Sequential(
                nn.Linear(latent_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, latent_channels),
            )
        else:
            self.mlp = nn.Identity()

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Node embeddings of shape (N, latent_channels).

        Returns:
            adj_hat: Reconstructed adjacency matrix of shape (N, N),
                     each entry is a predicted edge probability ∈ (0, 1).
        """
        z_proj  = self.mlp(z)                          # (N, latent_channels)
        scale   = z_proj.size(-1) ** 0.5               # √d stabilises dot products
        logits  = (z_proj @ z_proj.T) / scale          # (N, N)
        return torch.sigmoid(logits)                   # Â ∈ (0, 1)

def adj_reconstruction_loss(
    adj_hat:   Tensor,
    edge_index: Tensor,
    num_nodes:  int,
    pos_weight: Tensor,
) -> Tensor:
    # Build dense binary ground-truth adjacency
    adj_true = torch.zeros(num_nodes, num_nodes, device=adj_hat.device)
    adj_true[edge_index[0], edge_index[1]] = 1.0

    # Compute positive weight from sparsity ratio if not provided
    if pos_weight is None:
        num_edges    = edge_index.size(1)
        num_non_edge = num_nodes ** 2 - num_edges
        pos_weight   = torch.tensor(
            num_non_edge / (num_edges + 1e-8),
            device=adj_hat.device,
            dtype=adj_hat.dtype,
        )

    loss = F.binary_cross_entropy_with_logits(
        input      = adj_hat.view(-1),
        target     = adj_true.view(-1),
        pos_weight = pos_weight,
        reduction  = "mean",
    )
    return loss


class GMCM(nn.Module):
    """
    Differentiable approximation of a Gaussian Mixture Copula Model.

    Main idea:
    - Replace hard empirical rank transform with a smooth empirical CDF
      based on pairwise sigmoids.
    - Transform smooth CDF values to Gaussian normal scores.
    - Fit a Gaussian mixture in that transformed space.

    This allows gradients to flow back into z.
    """

    def __init__(
        self,
        latent_dim: int,
        n_clusters: int,
        reg_covar: float = 1e-4,
        tau: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.K = n_clusters
        self.D = latent_dim
        self.reg_covar = reg_covar
        self.tau = tau
        self.eps = eps

        self.log_pi = nn.Parameter(torch.zeros(n_clusters))
        self.mu = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.1)

        # raw lower-triangular factors for covariance
        self.L_raw = nn.Parameter(
            torch.eye(latent_dim).unsqueeze(0).repeat(n_clusters, 1, 1)
        )

    def _cholesky(self) -> Tensor:
        """
        Build valid lower-triangular Cholesky factors.
        """
        L = torch.tril(self.L_raw)
        idx = torch.arange(self.D, device=L.device)
        L[:, idx, idx] = F.softplus(L[:, idx, idx]) + self.reg_covar
        return L

    def _smooth_empirical_cdf(self, z: Tensor) -> Tensor:
        """
        Differentiable approximation of the empirical CDF.

        For each dimension independently:
            F_hat(z_i) = mean_j sigmoid((z_i - z_j) / tau)

        z: (N, D)
        returns u: (N, D) in (0,1)
        """
        # pairwise differences per dimension
        # diff[n, m, d] = z[n, d] - z[m, d]
        diff = z.unsqueeze(1) - z.unsqueeze(0)   # (N, N, D)

        # smooth indicator I[z_j <= z_i]
        cdf_vals = torch.sigmoid(diff / self.tau).mean(dim=1)  # (N, D)

        # avoid exact 0 or 1 before inverse Gaussian CDF
        u = cdf_vals.clamp(self.eps, 1.0 - self.eps)
        return u

    def _gaussian_ppf(self, u: Tensor) -> Tensor:
        """
        Approximate inverse standard normal CDF:
            Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
        """
        return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)

    def _to_normal_scores(self, z: Tensor) -> Tensor:
        """
        Differentiable copula transform.
        """
        u = self._smooth_empirical_cdf(z)
        v = self._gaussian_ppf(u)
        return v

    def _log_component_density(self, v: Tensor) -> Tensor:
        """
        Log-density of each Gaussian component.

        v: (N, D)
        returns: (N, K)
        """
        L = self._cholesky()                                # (K, D, D)
        diff = v.unsqueeze(1) - self.mu.unsqueeze(0)       # (N, K, D)

        # reshape for triangular solve
        # solve L_k * a = diff_{n,k}
        rhs = diff.permute(1, 2, 0)                        # (K, D, N)
        alpha = torch.linalg.solve_triangular(
            L, rhs, upper=False
        )                                                  # (K, D, N)

        maha = (alpha ** 2).sum(dim=1).T                   # (N, K)
        log_det = torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (K,)

        return -0.5 * maha - log_det - 0.5 * self.D * math.log(2.0 * math.pi)

    def logits(self, z: Tensor) -> Tensor:
        """
        Unnormalized log posterior scores: (N, K)
        """
        v = self._to_normal_scores(z)
        log_pi = F.log_softmax(self.log_pi, dim=0)
        return log_pi + self._log_component_density(v)

    def loss(self, z: Tensor) -> Tensor:
        """
        Unsupervised negative log-likelihood.
        """
        logits = self.logits(z)
        return -torch.logsumexp(logits, dim=1).mean()

    def soft_assign(self, z: Tensor) -> Tensor:
        """
        Posterior responsibilities: (N, K)
        """
        return F.softmax(self.logits(z), dim=1)

    @torch.no_grad()
    def predict(self, z: Tensor) -> Tensor:
        """
        Hard cluster assignments: (N,)
        """
        return self.soft_assign(z).argmax(dim=1)







def contrastive_loss(z: Tensor, labels: Tensor, temperature: float = 0.5) -> Tensor:
    """
    Supervised contrastive loss.
    Pulls same-cluster embeddings together, pushes different ones apart.

    Args:
        z:           Cell embeddings (N, D) — L2-normalised internally.
        labels:      Cluster assignments (N,) from gmcm.predict().
        temperature: Scaling factor for similarity sharpness.

    Returns:
        Scalar loss.
    """
    z     = F.normalize(z, dim=1)                          # (N, D)
    sim   = (z @ z.T) / temperature                        # (N, N)

    # Mask: 1 where i,j share the same label (excluding diagonal)
    eq    = labels.unsqueeze(0) == labels.unsqueeze(1)     # (N, N)
    mask  = eq & ~torch.eye(len(labels), dtype=torch.bool, device=z.device)

    # Log-softmax over all negatives
    logits     = sim - sim.diagonal().unsqueeze(1)         # anchor each row
    log_prob   = logits - torch.logsumexp(sim, dim=1, keepdim=True)

    # Mean over positive pairs only
    n_pos = mask.sum(1).clamp(min=1)
    loss  = -(log_prob * mask).sum(1) / n_pos

    return loss.mean()

def kl_divergence(mu, log_var):
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()

def augment_graph(data, edge_drop=0.2, feat_mask=0.2):
    # randomly drop edges
    E    = data.edge_index.size(1)
    mask = torch.rand(E, device=data.edge_index.device) > edge_drop
    ei   = data.edge_index[:, mask]

    # randomly mask features
    x = data.x * (torch.rand_like(data.x) > feat_mask).float()

    return x, ei

def nt_xent_loss(z1, z2, temperature=0.5):
    N  = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    z    = torch.cat([z1, z2], dim=0)                        # (2N, D)
    sim  = (z @ z.T) / temperature                           # (2N, 2N)
    sim  = sim.masked_fill(torch.eye(2*N, dtype=torch.bool, device=z.device), float("-inf"))

    labels = torch.cat([torch.arange(N, device=z.device) + N,
                        torch.arange(N, device=z.device)])   # (2N,)

    return F.cross_entropy(sim, labels)