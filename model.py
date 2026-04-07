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
from typing import Tuple
from torch_geometric.data import Data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    accuracy_score,
)
from scipy.optimize import linear_sum_assignment



class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_channels_2, latent_channels,
                 activation=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.activation    = activation
        self.dropout       = nn.Dropout(p=dropout)
        self.conv_hidden_1 = gnc_encoder(in_channels,       hidden_channels)
        self.conv_hidden_2 = gnc_encoder(hidden_channels,   hidden_channels_2)
        self.conv_mu       = gnc_encoder(hidden_channels_2, latent_channels)
        self.conv_logvar   = gnc_encoder(hidden_channels_2, latent_channels)

    def _encode(self, x, edge_index):
        h = self.dropout(self.activation(self.conv_hidden_1(x, edge_index)))
        h = self.dropout(self.activation(self.conv_hidden_2(h, edge_index)))
        return h

    def forward(self, x, edge_index) :
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
    """
    Weighted binary cross-entropy between the reconstructed adjacency Â
    and the true binary adjacency A, with automatic positive-class reweighting
    to counter the severe class imbalance in sparse graphs.

    For a sparse graph with E edges on N nodes, there are only E positive
    entries but N²-E negatives — naïve BCE would collapse to predicting all
    zeros. The positive weight w = (N²-E)/E corrects for this.

    Args:
        adj_hat:    Predicted edge probabilities, shape (N, N) ∈ (0,1).
        edge_index: True edges as COO index, shape (2, E).
        num_nodes:  N — total number of nodes.
        pos_weight: Override automatic positive weight (scalar tensor).
                    Pass None to compute automatically from edge_index.

    Returns:
        Scalar weighted BCE loss.
    """
    # Build dense binary ground-truth adjacency ---------------------------------
    adj_true = torch.zeros(num_nodes, num_nodes, device=adj_hat.device)
    adj_true[edge_index[0], edge_index[1]] = 1.0

    # Compute positive weight from sparsity ratio if not provided ---------------
    if pos_weight is None:
        num_edges    = edge_index.size(1)
        num_non_edge = num_nodes ** 2 - num_edges
        pos_weight   = torch.tensor(
            num_non_edge / (num_edges + 1e-8),
            device=adj_hat.device,
            dtype=adj_hat.dtype,
        )

    # Weighted BCE on flattened (N²,) tensors -----------------------------------
    loss = F.binary_cross_entropy_with_logits(
        input      = adj_hat.view(-1),
        target     = adj_true.view(-1),
        pos_weight = pos_weight,
        reduction  = "mean",
    )
    return loss


class GaussianMixtureCopulaModule(nn.Module):
    """
    Gaussian Mixture Copula Model (GMCM) for unsupervised clustering in
    latent space.

    The copula separates the modelling of marginal distributions from the
    modelling of inter-dimensional dependence structure:

        1. Marginal transform (Sklar's theorem):
               z  →  u = Φ⁻¹( F̂(z) )
           Each dimension of z is mapped through its empirical CDF F̂ and
           then through the probit transform Φ⁻¹ (inverse standard normal
           CDF), yielding pseudo-normal scores u whose marginals are
           standard normal. This removes any marginal-specific shape and
           leaves only the dependency structure.

        2. GMM in copula space:
           A K-component Gaussian mixture is fitted on u. Each component
           has a learnable mean μ_k and a full (lower-triangular) Cholesky
           factor L_k that parameterises its covariance Σ_k = L_k L_kᵀ.
           Full covariance captures arbitrary linear dependencies between
           dimensions inside each cluster.

        3. Soft responsibilities:
               r_ik = π_k · p(u_i | k)  /  Σ_j π_j · p(u_i | j)
           where p(u|k) = N(u; μ_k, L_k L_kᵀ).

    Args:
        latent_channels:  Dimensionality of z (and u).
        num_clusters:     Number of Gaussian copula components K.
        eps:              Diagonal jitter added to Cholesky for stability.
    """

    def __init__(
        self,
        latent_channels: int,
        num_clusters:    int,
        eps:             float = 1e-4,
    ) -> None:
        super().__init__()

        self.K   = num_clusters
        self.D   = latent_channels
        self.eps = eps

        # --- learnable GMM parameters in copula space -------------------------
        self.mu        = nn.Parameter(torch.randn(num_clusters, latent_channels) * 0.1)
        self.logits_pi = nn.Parameter(torch.zeros(num_clusters))

        # Lower-triangular Cholesky factors L_k stored as flat vectors.
        # tril_indices gives positions of the D*(D+1)/2 lower-tri entries.
        n_tril = latent_channels * (latent_channels + 1) // 2

        rows, cols = torch.tril_indices(latent_channels, latent_channels)
        L_init = (
            torch.eye(latent_channels)
            .unsqueeze(0)
            .expand(num_clusters, -1, -1)
            .reshape(num_clusters, latent_channels, latent_channels)
        )
        self.L_flat = nn.Parameter(L_init[:, rows, cols].clone())  # index with plain variables

        self.register_buffer("tril_rows", rows)
        self.register_buffer("tril_cols", cols)

    # ------------------------------------------------------------------
    # Cholesky reconstruction
    # ------------------------------------------------------------------

    def _get_cholesky(self) -> Tensor:
        """
        Reconstructs (K, D, D) lower-triangular Cholesky matrices from
        L_flat. Diagonal entries are passed through softplus to ensure
        they are strictly positive (required for valid Cholesky factor).
        """
        L = torch.zeros(self.K, self.D, self.D, device=self.mu.device)
        L[:, self.tril_rows, self.tril_cols] = self.L_flat

        # softplus on diagonal entries ensures Σ_k = L_k L_kᵀ is PSD
        diag_idx = torch.arange(self.D, device=self.mu.device)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + self.eps
        return L                                               # (K, D, D)

    # ------------------------------------------------------------------
    # Marginal transform  z → u
    # ------------------------------------------------------------------

    @staticmethod
    def _empirical_cdf(z: Tensor) -> Tensor:
        """
        Computes per-dimension empirical CDF values (ranks / N).
        Uses mid-ranks to avoid boundary values of exactly 0 or 1.

        Args:
            z: (N, D)

        Returns:
            u_hat: (N, D)  values in (0, 1)
        """
        N    = z.size(0)
        # argsort twice gives the rank of each element
        ranks = z.argsort(dim=0).argsort(dim=0).float()       # (N, D)
        return (ranks + 1.0) / (N + 1.0)                      # mid-rank CDF

    @staticmethod
    def _probit(p: Tensor) -> Tensor:
        """Φ⁻¹(p) — inverse standard normal CDF via erfinv."""
        p     = p.clamp(1e-6, 1 - 1e-6)
        return torch.erfinv(2.0 * p - 1.0) * (2.0 ** 0.5)

    def _marginal_transform(self, z: Tensor) -> Tensor:
        """
        Full marginal transform: z → u = Φ⁻¹( F̂(z) ).

        Args:
            z: (N, D)

        Returns:
            u: (N, D)  pseudo-normal copula scores
        """
        cdf = self._empirical_cdf(z)
        return self._probit(cdf)

    # ------------------------------------------------------------------
    # GMM log-likelihood in copula space
    # ------------------------------------------------------------------

    def _log_component_density(self, u: Tensor) -> Tensor:
        """
        Log-likelihood of u under each Gaussian component using the
        Cholesky parameterisation for numerical efficiency:

            log N(u; μ_k, L_k L_kᵀ) =
                -D/2 log(2π)
                - Σ_d log L_k[d,d]          ← log|Σ|/2 via Cholesky
                - 1/2 ‖L_k⁻¹(u - μ_k)‖²   ← Mahalanobis via triangular solve

        Args:
            u: (N, D)

        Returns:
            log_p: (N, K)
        """
        L       = self._get_cholesky()                        # (K, D, D)
        diff    = u.unsqueeze(1) - self.mu.unsqueeze(0)       # (N, K, D)

        # Triangular solve: v_ik = L_k⁻¹ (u_i - μ_k)
        # torch.linalg.solve_triangular expects (..., D, D), (..., D, 1)
        diff_t  = diff.unsqueeze(-1)                          # (N, K, D, 1)
        L_exp   = L.unsqueeze(0).expand(u.size(0), -1, -1, -1)  # (N, K, D, D)
        v       = torch.linalg.solve_triangular(
            L_exp, diff_t, upper=False
        ).squeeze(-1)                                         # (N, K, D)

        # log|Σ_k|/2 = Σ_d log L_k[d,d]
        diag_idx   = torch.arange(self.D, device=u.device)
        log_det    = L[:, diag_idx, diag_idx].log().sum(dim=-1)  # (K,)

        mahalanobis = 0.5 * v.pow(2).sum(dim=-1)             # (N, K)
        log_2pi     = 0.5 * self.D * torch.log(
            torch.tensor(2 * torch.pi, device=u.device)
        )

        return -log_2pi - log_det.unsqueeze(0) - mahalanobis  # (N, K)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            z: (N, D) latent node embeddings.

        Returns:
            r:      Soft responsibilities (N, K), each row sums to 1.
            log_px: Per-node log marginal likelihood (N,) for the NLL loss.
        """
        u        = self._marginal_transform(z)                # (N, D)
        log_pi   = F.log_softmax(self.logits_pi, dim=0)       # (K,)
        log_p_k  = self._log_component_density(u)             # (N, K)
        log_w_p  = log_pi.unsqueeze(0) + log_p_k             # (N, K)

        log_px   = torch.logsumexp(log_w_p, dim=-1)           # (N,)  log Σ_k π_k p(u|k)
        r        = (log_w_p - log_px.unsqueeze(1)).exp()      # (N, K)  responsibilities

        return r, log_px

    @torch.no_grad()
    def predict(self, z: Tensor) -> Tensor:
        """Hard cluster labels via argmax over responsibilities."""
        r, _ = self.forward(z)
        return r.argmax(dim=-1)                               # (N,)

    @torch.no_grad()
    def initialize_means(self, z: Tensor) -> None:
        """
        Warm-starts cluster means using K-Means++ on the copula-transformed
        scores u. Always call before training to avoid degenerate initialisation.
        """
        u       = self._marginal_transform(z)
        centers = _kmeans_pp_init(u, self.K)
        self.mu.data.copy_(centers)


# -----------------------------------------------------------------------
# K-Means++ initialisation
# -----------------------------------------------------------------------

def _kmeans_pp_init(u: Tensor, K: int) -> Tensor:
    N       = u.size(0)
    idx     = torch.randint(N, (1,)).item()
    centers = [u[idx].detach().cpu()]

    for _ in range(1, K):
        stacked = torch.stack(centers, dim=0)
        dists   = torch.cdist(u.detach().cpu(), stacked).min(dim=1).values
        probs   = dists.pow(2)
        probs  /= probs.sum()
        idx     = torch.multinomial(probs, 1).item()
        centers.append(u[idx].detach().cpu())

    return torch.stack(centers).to(u.device)


# -----------------------------------------------------------------------
# GMCM loss
# -----------------------------------------------------------------------

def gmcm_loss(
    r:            Tensor,
    log_px:       Tensor,
    logits_pi:    Tensor,
    lambda_ent:   float = 0.1,
    lambda_prior: float = 0.01,
) -> Tensor:
    """
    GMCM clustering loss with three terms:

    1. NLL — negative log marginal likelihood of u under the mixture.
         Maximising this is the canonical copula model objective; it
         jointly optimises component parameters and mixture weights.

    2. Entropy regularisation — maximises H(r) across nodes to prevent
         cluster collapse where all nodes are assigned to one component.

    3. Uniform prior on π — KL(π ‖ Uniform) penalises dead components
         whose mixture weight decays to zero, maintaining diversity.

    Args:
        r:            Soft responsibilities (N, K).
        log_px:       Per-node log marginal likelihood (N,).
        logits_pi:    Raw mixture weight logits (K,) from the module.
        lambda_ent:   Weight on entropy regularisation term.
        lambda_prior: Weight on mixture prior regularisation term.

    Returns:
        Scalar total loss.
    """
    # 1. NLL: minimise negative log-likelihood --------------------------------
    nll = -log_px.mean()

    # 2. Entropy regularisation: maximise H(r) → minimise -H(r) --------------
    entropy = -(r * r.log().clamp(min=-1e8)).sum(dim=-1).mean()

    # 3. Uniform prior on π: KL(π ‖ Uniform(K)) ------------------------------
    K           = logits_pi.size(0)
    log_pi      = F.log_softmax(logits_pi, dim=0)
    log_uniform = torch.full_like(log_pi, -torch.log(torch.tensor(K, dtype=log_pi.dtype)))
    prior_kl    = F.kl_div(log_pi, log_uniform.exp(), reduction="sum")

    return nll - lambda_ent * entropy + lambda_prior * prior_kl





# -----------------------------------------------------------------------
# Reparameterisation trick
# -----------------------------------------------------------------------

def reparameterise(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    z = μ + ε·σ,  ε ~ N(0, I)

    During eval the encoder is deterministic (use mu directly).
    During training the stochastic sample enables the ELBO gradient.
    """
    std = (0.5 * log_var).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


# -----------------------------------------------------------------------
# KL divergence  KL( q(z|x) ‖ N(0,I) )
# -----------------------------------------------------------------------

def kl_divergence(mu: Tensor, log_var: Tensor) -> Tensor:
    """
    Analytical KL between a diagonal Gaussian q(z|x) = N(μ, diag(σ²))
    and the standard normal prior p(z) = N(0, I):

        KL = -½ Σ_d ( 1 + log σ²_d - μ²_d - σ²_d )

    Averaged over nodes and latent dimensions.
    """
    return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()


# -----------------------------------------------------------------------
# Loss weights
# -----------------------------------------------------------------------

class LossWeights:
    zinb:    float = 1.0
    adj:     float = 1.0
    kl:      float = 1e-3   # start small — KL tends to dominate early
    gmcm:    float = 0.1

    # KL annealing: ramp from 0 → kl_max over the pre-train phase
    kl_ramp_start:   int = 0
    kl_ramp_end:     int = 50

    # GMCM annealing: only active during fine-tuning
    gmcm_ramp_start: int = 100
    gmcm_ramp_end:   int = 300


# -----------------------------------------------------------------------
# Annealing helper
# -----------------------------------------------------------------------

def linear_anneal(epoch: int, start: int, end: int, max_val: float) -> float:
    """Linearly ramps a weight from 0 to max_val between start and end."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return max_val
    return max_val * (epoch - start) / (end - start)


# -----------------------------------------------------------------------
# Single training step
# -----------------------------------------------------------------------

def train_step(
    data:       Data,
    encoder:    nn.Module,
    zinb_dec:   nn.Module,
    adj_dec:    nn.Module,
    gmcm:       nn.Module,
    optimizer:  torch.optim.Optimizer,
    weights:    LossWeights,
    epoch:      int,
    device:     torch.device,
) :

    encoder.train()
    zinb_dec.train()
    adj_dec.train()
    gmcm.train()
    optimizer.zero_grad()

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    N          = x.size(0)

    # --- 1. Encode (VGAE) ----------------------------------------------------
    # GCNEncoder now returns (mu, log_var) — see updated encoder below
    mu, log_var = encoder(x, edge_index)              # (N, D), (N, D)
    z           = reparameterise(mu, log_var)         # (N, D)  stochastic sample

    # --- 2. KL divergence (annealed) -----------------------------------------
    loss_kl  = kl_divergence(mu, log_var)
    w_kl     = linear_anneal(epoch, weights.kl_ramp_start, weights.kl_ramp_end, weights.kl)

    # --- 3. ZINB feature reconstruction loss ---------------------------------
    pi, mu_zinb, theta = zinb_dec(z)
    loss_zinb          = zinb_loss(x, pi, mu_zinb, theta)

    # --- 4. Adjacency reconstruction loss ------------------------------------
    adj_hat  = adj_dec(z)
    loss_adj = adj_reconstruction_loss(adj_hat, edge_index, N,None)

    # --- 5. GMCM clustering loss (annealed) ----------------------------------
    r, log_px = gmcm(z)
    loss_gmcm = gmcm_loss(r, log_px, gmcm.logits_pi)
    w_gmcm    = linear_anneal(epoch, weights.gmcm_ramp_start, weights.gmcm_ramp_end, weights.gmcm)

    # --- 6. Total ELBO-style objective ---------------------------------------
    #
    #   L = E[log p(x|z)]          ← ZINB reconstruction
    #     + E[log p(A|z)]          ← adjacency reconstruction
    #     - KL(q(z|x) ‖ p(z))     ← VAE regularisation
    #     + λ_gmcm · L_gmcm       ← clustering (annealed)
    #
    loss = (
        weights.zinb * loss_zinb
        + weights.adj * loss_adj
        + w_kl        * loss_kl
        + w_gmcm      * loss_gmcm
    )

    loss.backward()
    nn.utils.clip_grad_norm_(
        list(encoder.parameters())
        + list(zinb_dec.parameters())
        + list(adj_dec.parameters())
        + list(gmcm.parameters()),
        max_norm=1.0,
    )
    optimizer.step()

    return {
        "loss/total": loss.item(),
        "loss/zinb":  loss_zinb.item(),
        "loss/adj":   loss_adj.item(),
        "loss/kl":    loss_kl.item(),
        "loss/gmcm":  loss_gmcm.item(),
        "w/kl":       w_kl,
        "w/gmcm":     w_gmcm,
    }


# -----------------------------------------------------------------------
# Evaluation step
# -----------------------------------------------------------------------

@torch.no_grad()
def eval_step(
    data:     Data,
    encoder:  nn.Module,
    zinb_dec: nn.Module,
    adj_dec:  nn.Module,
    gmcm:     nn.Module,
    device:   torch.device,
) :

    encoder.eval()
    zinb_dec.eval()
    adj_dec.eval()
    gmcm.eval()

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    N          = x.size(0)

    # Use mu directly at eval time — deterministic, lower variance estimate
    mu, log_var      = encoder(x, edge_index)
    z                = mu

    pi, mu_zinb, theta = zinb_dec(z)
    adj_hat            = adj_dec(z)
    r, log_px          = gmcm(z)

    return {
        "val/zinb": zinb_loss(x, pi, mu_zinb, theta).item(),
        "val/adj":  adj_reconstruction_loss(adj_hat, edge_index, N,None).item(),
        "val/kl":   kl_divergence(mu, log_var).item(),
        "val/gmcm": gmcm_loss(r, log_px, gmcm.logits_pi).item(),
        "val/nll":  (-log_px.mean()).item(),
    }




# -----------------------------------------------------------------------
# Cluster accuracy via Hungarian algorithm
# -----------------------------------------------------------------------

def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes clustering accuracy by solving the optimal label assignment
    between predicted cluster indices and true class labels using the
    Hungarian algorithm.

    A direct accuracy_score(y_true, y_pred) would be wrong because
    cluster index 0 may correspond to true class 3 — the Hungarian
    method finds the permutation of predicted labels that maximises
    agreement with ground truth.

    Args:
        y_true: Ground-truth integer labels (N,).
        y_pred: Predicted cluster indices   (N,).

    Returns:
        Accuracy in [0, 1].
    """
    assert y_true.shape == y_pred.shape
    K = max(y_true.max(), y_pred.max()) + 1

    # Build confusion matrix C where C[i,j] = # nodes with pred=i, true=j
    C = np.zeros((K, K), dtype=np.int64)
    for p, t in zip(y_pred, y_true):
        C[p, t] += 1

    # Hungarian algorithm finds the assignment that maximises the trace
    row_ind, col_ind = linear_sum_assignment(-C)
    return C[row_ind, col_ind].sum() / y_true.shape[0]


# -----------------------------------------------------------------------
# Clustering metrics
# -----------------------------------------------------------------------

@torch.no_grad()
def compute_metrics(
    encoder:  nn.Module,
    gmcm:     nn.Module,
    data:     Data,
    y_true:   np.ndarray,
    device:   torch.device,
) -> Dict[str, float]:
    """
    Computes ACC, NMI, and ARI against ground-truth labels.

    Args:
        encoder:  Trained GCNEncoder (VGAE — returns mu, log_var).
        gmcm:     Trained GaussianMixtureCopulaModule.
        data:     PyG Data object.
        y_true:   Ground-truth integer class labels (N,).
        device:   Target device.

    Returns:
        Dict with keys acc, nmi, ari.
    """
    encoder.eval()
    gmcm.eval()

    mu, _  = encoder(data.x.to(device), data.edge_index.to(device))
    r, _   = gmcm(mu)
    y_pred = r.argmax(dim=-1).cpu().numpy()

    return {
        "acc": cluster_accuracy(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic"),
        "ari": adjusted_rand_score(y_true, y_pred),
    }


# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------

def save_checkpoint(
    path:     str,
    epoch:    int,
    encoder:  nn.Module,
    zinb_dec: nn.Module,
    adj_dec:  nn.Module,
    gmcm:     nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics:  Dict[str, float],
) -> None:
    torch.save({
        "epoch":          epoch,
        "encoder":        encoder.state_dict(),
        "zinb_dec":       zinb_dec.state_dict(),
        "adj_dec":        adj_dec.state_dict(),
        "gmcm":           gmcm.state_dict(),
        "optimizer":      optimizer.state_dict(),
        "metrics":        metrics,
    }, path)


def load_checkpoint(
    path:     str,
    encoder:  nn.Module,
    zinb_dec: nn.Module,
    adj_dec:  nn.Module,
    gmcm:     nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    encoder.load_state_dict(ckpt["encoder"])
    zinb_dec.load_state_dict(ckpt["zinb_dec"])
    adj_dec.load_state_dict(ckpt["adj_dec"])
    gmcm.load_state_dict(ckpt["gmcm"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# -----------------------------------------------------------------------
# Pretty printer
# -----------------------------------------------------------------------

def print_epoch(
    epoch:       int,
    total:       int,
    stage:       str,
    train_m:     Dict[str, float],
    val_m:       Optional[Dict[str, float]],
    cluster_m:   Optional[Dict[str, float]],
    is_best:     bool,
) -> None:
    tag  = "★ BEST" if is_best else "      "
    sep  = "─" * 80

    print(sep)
    print(
        f"{tag}  [{stage.upper()}]  epoch {epoch:>4d}/{total}"
        f"   total={train_m['loss/total']:.4f}"
        f"   zinb={train_m['loss/zinb']:.4f}"
        f"   adj={train_m['loss/adj']:.4f}"
        f"   kl={train_m['loss/kl']:.4f}(w={train_m['w/kl']:.3f})"
        f"   gmcm={train_m['loss/gmcm']:.4f}(w={train_m['w/gmcm']:.3f})"
    )
    if val_m:
        print(
            f"         [VAL]"
            f"   zinb={val_m['val/zinb']:.4f}"
            f"   adj={val_m['val/adj']:.4f}"
            f"   kl={val_m['val/kl']:.4f}"
            f"   gmcm={val_m['val/gmcm']:.4f}"
            f"   nll={val_m['val/nll']:.4f}"
        )
    if cluster_m:
        print(
            f"         [CLUST]"
            f"   ACC={cluster_m['acc']:.4f}"
            f"   NMI={cluster_m['nmi']:.4f}"
            f"   ARI={cluster_m['ari']:.4f}"
        )


# -----------------------------------------------------------------------
# Full training loop
# -----------------------------------------------------------------------

def train(
    data:              Data,
    encoder:           nn.Module,
    zinb_dec:          nn.Module,
    adj_dec:           nn.Module,
    gmcm:              nn.Module,
    y_true:            Optional[np.ndarray]  = None,
    n_pretrain_epochs: int                   = 250,
    n_finetune_epochs: int                   = 800,
    lr:                float                 = 1e-4,
    weight_decay:      float                 = 1e-4,
    device:            Optional[torch.device] = None,
    val_data:          Optional[Data]        = None,
    log_every:         int                   = 10,
    ckpt_path:         str                   = "best_model.pt",
    best_metric:       str                   = "nmi",   # acc | nmi | ari | val/nll
) -> Dict[str, List]:
    """
    Two-stage VGAE training with checkpointing and clustering metrics.

    Best model selection:
        If y_true is provided, best model is selected by the clustering
        metric specified in best_metric (acc / nmi / ari), evaluated on
        the training graph (standard protocol for unsupervised graph
        clustering benchmarks where no held-out labels exist).

        If y_true is None, best model falls back to minimising val/nll
        (or total train loss when val_data is also None).

    Args:
        data:              Training PyG Data object.
        encoder:           GCNEncoder instance.
        zinb_dec:          ZINBDecoder instance.
        adj_dec:           AdjDecoder instance.
        gmcm:              GaussianMixtureCopulaModule instance.
        y_true:            Ground-truth integer labels (N,) for ACC/NMI/ARI.
        n_pretrain_epochs: Reconstruction-only warm-up epochs.
        n_finetune_epochs: Joint fine-tuning epochs.
        lr:                Learning rate.
        weight_decay:      AdamW L2 penalty.
        device:            Target device (auto if None).
        val_data:          Optional held-out graph for validation losses.
        log_every:         Print every N epochs.
        ckpt_path:         Where to save the best checkpoint.
        best_metric:       Metric used to decide "best" model.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.to(device)
    zinb_dec.to(device)
    adj_dec.to(device)
    gmcm.to(device)

    optimizer = Adam(
        list(encoder.parameters())
        + list(zinb_dec.parameters())
        + list(adj_dec.parameters())
        + list(gmcm.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=False)

    weights                  = LossWeights()
    weights.kl_ramp_end      = n_pretrain_epochs
    weights.gmcm_ramp_start  = n_pretrain_epochs
    weights.gmcm_ramp_end    = n_pretrain_epochs + n_finetune_epochs // 2

    history: Dict[str, List] = {k: [] for k in [
        "loss/total", "loss/zinb", "loss/adj", "loss/kl",  "loss/gmcm",
        "val/zinb",   "val/adj",   "val/kl",   "val/gmcm", "val/nll",
        "acc",        "nmi",       "ari",
    ]}

    total_epochs = n_pretrain_epochs + n_finetune_epochs

    # higher-is-better metrics; lower-is-better otherwise
    higher_is_better = best_metric in ("acc", "nmi", "ari")
    best_score       = -np.inf if higher_is_better else np.inf
    best_epoch       = 0

    print(f"\n{'═'*80}")
    print(f"  Training   device={device}   pretrain={n_pretrain_epochs}   finetune={n_finetune_epochs}")
    print(f"  Best-model criterion: {best_metric}   checkpoint → {ckpt_path}")
    print(f"{'═'*80}\n")

    for epoch in range(total_epochs):

        # --- GMCM warm start at stage transition ----------------------------
        if epoch == n_pretrain_epochs:
            print(f"\n{'─'*80}")
            print("  Pre-training complete — initialising GMCM via K-Means++ on μ")
            with torch.no_grad():
                encoder.eval()
                mu, _ = encoder(data.x.to(device), data.edge_index.to(device))
                gmcm.initialize_means(mu)
            print("  Beginning joint fine-tuning")
            print(f"{'─'*80}\n")

        # --- Training step --------------------------------------------------
        train_m = train_step(
            data, encoder, zinb_dec, adj_dec, gmcm,
            optimizer, weights, epoch, device,
        )
        for k, v in train_m.items():
            if k in history:
                history[k].append(v)

        # --- Validation losses ----------------------------------------------
        val_m = None
        if val_data is not None:
            val_m = eval_step(val_data, encoder, zinb_dec, adj_dec, gmcm, device)
            for k, v in val_m.items():
                history[k].append(v)
            monitor = val_m["val/zinb"] + val_m["val/adj"] + val_m["val/kl"]
            scheduler.step(monitor)

        # --- Clustering metrics (only during fine-tuning) -------------------
        cluster_m = None
        in_finetune = epoch >= n_pretrain_epochs
        if in_finetune and y_true is not None:
            cluster_m = compute_metrics(encoder, gmcm, data, y_true, device)
            for k, v in cluster_m.items():
                history[k].append(v)

        # --- Best model selection -------------------------------------------
        is_best = False
        if in_finetune:
            if best_metric in ("acc", "nmi", "ari") and cluster_m is not None:
                score = cluster_m[best_metric]
            elif val_m is not None:
                score = val_m.get(best_metric, val_m["val/nll"])
            else:
                score = -train_m["loss/total"]  # fallback: minimise total loss

            improved = score > best_score if higher_is_better else score < best_score
            if improved:
                best_score = score
                best_epoch = epoch
                is_best    = True
                save_checkpoint(
                    ckpt_path, epoch,
                    encoder, zinb_dec, adj_dec, gmcm, optimizer,
                    metrics={**(cluster_m or {}), **(val_m or {}),
                             "epoch": epoch, best_metric: score},
                )

        # --- Logging --------------------------------------------------------
        if epoch % log_every == 0 or epoch == total_epochs - 1 or is_best:
            stage = "pretrain" if epoch < n_pretrain_epochs else "finetune"
            print_epoch(epoch, total_epochs, stage, train_m, val_m, cluster_m, is_best)

    # --- Final summary ------------------------------------------------------
    print(f"\n{'═'*80}")
    print(f"  Training complete.")
    print(f"  Best epoch : {best_epoch}")
    print(f"  Best {best_metric:>4s}  : {best_score:.4f}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"{'═'*80}\n")

    return history