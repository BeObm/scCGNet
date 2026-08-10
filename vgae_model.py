import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ----------------------------- Step 1: encoder ------------------------------
class VariationalGraphEncoder(nn.Module):
    """Variational graph encoder with a swappable GNN backbone.

    Maps node features -> latent (mu, logvar) and returns a sampled z via the
    reparameterization trick. All three downstream branches consume the same z.

    Args:
        in_dim:     input features per node  (== features.shape[1])
        hidden_dim: width of the shared GNN layer
        latent_dim: latent dimension
        conv_layer: any PyG conv class with an (in_channels, out_channels)
                    signature, e.g. GCNConv, GraphConv, SAGEConv, GATConv.
                    For GATConv, set heads=1 (or concat=False) so dims line up.

    Tensor prep from your numpy arrays:
        x          = torch.tensor(features,   dtype=torch.float32)   # [N, in_dim]
        edge_index = torch.tensor(edge_index, dtype=torch.long)      # [2, E]
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, conv_layer=GCNConv):
        super().__init__()
        self.shared = conv_layer(in_dim, hidden_dim)
        self.conv_mu = conv_layer(hidden_dim, latent_dim)
        self.conv_logvar = conv_layer(hidden_dim, latent_dim)

    def encode(self, x, edge_index):
        h = torch.relu(self.shared(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


# --------------------- Step 2: topology (adjacency) branch -------------------
class InnerProductDecoder(nn.Module):
    """Reconstructs adjacency logits as z z^T."""

    def forward(self, z):
        return z @ z.t()                       # [N, N] logits


def topology_recon_loss(logits, adj):
    """Weighted BCE between reconstructed and true adjacency.

    adj: dense float tensor [N, N] with 0/1 entries.
    """
    n = adj.size(0)
    pos = adj.sum()
    pos_weight = (n * n - pos) / pos
    norm = n * n / ((n * n - pos) * 2)
    return norm * F.binary_cross_entropy_with_logits(logits, adj, pos_weight=pos_weight)


def kl_divergence(mu, logvar):
    """Standard VGAE KL term, per-node averaged. Added ONCE to the total loss."""
    n = mu.size(0)
    return -0.5 / n * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


# ----------------------- Step 3: feature (ZINB) branch -----------------------
class ZINBDecoder(nn.Module):
    """Decodes z into ZINB parameters for raw-count reconstruction.

    Returns:
        mean: expected counts (mu), positive
        disp: dispersion (theta), positive
        pi:   dropout / zero-inflation probability in (0, 1)
    """

    def __init__(self, latent_dim, hidden_dim, n_genes):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, n_genes)
        self.disp = nn.Linear(hidden_dim, n_genes)
        self.pi = nn.Linear(hidden_dim, n_genes)

    def forward(self, z):
        h = self.hidden(z)
        mean = torch.clamp(torch.exp(self.mean(h)), min=1e-5, max=1e6)
        disp = torch.clamp(F.softplus(self.disp(h)), min=1e-4, max=1e4)
        pi = torch.sigmoid(self.pi(h))
        return mean, disp, pi


def zinb_loss(x, mean, disp, pi, scale_factor=1.0, eps=1e-10):
    """Negative ZINB log-likelihood (DCA / scDeepCluster formulation).

    x:            raw count matrix [N, n_genes]
    scale_factor: optional per-cell size factors [N, 1] (library size); 1.0 if unused
    """
    mean = mean * scale_factor
    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + mean / (disp + eps)) \
        + x * (torch.log(disp + eps) - torch.log(mean + eps))
    nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
    result = torch.where(x < 1e-8, zero_case, nb_case)
    return result.mean()


# ------------------------- Step 4: GMCM clustering branch --------------------
class GMCMClustering(nn.Module):
    """Gaussian Mixture Copula clustering head over the latent z.

    Learnable mixture parameters {pi_k, mu_k, var_k} with DIAGONAL covariance.
    The clustering loss is the GMCM copula negative log-likelihood of z:

        NLL = - mean_i [ log g(z_i) - sum_j log g_j(z_ij) ]

        log g(z_i)    = logsumexp_k [ log pi_k + log N(z_i;  mu_k,  diag(var_k)) ]
        log g_j(z_ij) = logsumexp_k [ log pi_k + log N(z_ij; mu_kj, var_kj)      ]

    Subtracting the marginals log g_j is the copula term: it removes the
    marginal information and keeps only the dependence structure -- this is what
    makes it a copula model rather than a plain GMM. Because the mixture lives
    directly in z-space (y = z), the marginal inverse G_j^{-1} is the identity,
    so no CDF inversion is required and the whole term is differentiable.

    Warm-start note: cluster quality depends strongly on mu_k init -- run k-means
    on a pretrained z and copy centroids into self.mu before joint training.

    Args:
        n_clusters: K
        latent_dim: d (== encoder latent_dim)
    """

    def __init__(self, n_clusters, latent_dim):
        super().__init__()
        self.logits_pi = nn.Parameter(torch.zeros(n_clusters))            # -> softmax
        self.mu = nn.Parameter(torch.randn(n_clusters, latent_dim) * 0.1)
        self.logvar = nn.Parameter(torch.zeros(n_clusters, latent_dim))   # diag log-var

    def _component_log_prob(self, z):
        """Per-component marginal (and joint) Gaussian log-densities.

        Returns:
            log_joint: [N, K]     sum over dims  -> log N(z_i;  mu_k, diag(var_k))
            log_marg:  [N, K, d]  per dim        -> log N(z_ij; mu_kj, var_kj)
        """
        logvar = self.logvar.clamp(min=-10.0, max=10.0)                   # stability
        var = logvar.exp()
        z = z.unsqueeze(1)                                                # [N, 1, d]
        mu = self.mu.unsqueeze(0)                                         # [1, K, d]
        log_marg = -0.5 * (math.log(2 * math.pi) + logvar.unsqueeze(0)
                           + (z - mu) ** 2 / var.unsqueeze(0))            # [N, K, d]
        return log_marg.sum(-1), log_marg                                 # [N, K], [N, K, d]

    def forward(self, z):
        """Soft cluster responsibilities gamma [N, K] (for assignment/eval)."""
        log_pi = F.log_softmax(self.logits_pi, dim=0)                     # [K]
        log_joint, _ = self._component_log_prob(z)
        return torch.softmax(log_pi.unsqueeze(0) + log_joint, dim=1)      # [N, K]

    def gmcm_nll(self, z):
        """GMCM copula negative log-likelihood over z (the clustering loss)."""
        log_pi = F.log_softmax(self.logits_pi, dim=0)                     # [K]
        log_joint, log_marg = self._component_log_prob(z)                 # [N,K], [N,K,d]
        log_gz = torch.logsumexp(log_pi.unsqueeze(0) + log_joint, dim=1)  # [N]
        log_gj = torch.logsumexp(log_pi.view(1, -1, 1) + log_marg, dim=1) # [N, d]
        copula_ll = log_gz - log_gj.sum(1)                                # [N]
        return -copula_ll.mean()

    @torch.no_grad()
    def assign(self, z):
        """Hard cluster labels [N] for ARI/NMI evaluation."""
        return self.forward(z).argmax(1)


# ----------------------------- Step 5: full model ---------------------------
class GraphVAE(nn.Module):
    """Ties the encoder to the three branches and combines their losses.

    x_input:  encoder input  [N, in_dim]  (normalized / log expression)
    x_counts: ZINB target    [N, n_genes] (RAW counts)
    Keep these two distinct -- the encoder reads normalized features, the
    feature branch reconstructs raw counts.
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, n_genes, n_clusters,
                 conv_layer=GCNConv):
        super().__init__()
        self.encoder = VariationalGraphEncoder(in_dim, hidden_dim, latent_dim, conv_layer)
        self.adj_decoder = InnerProductDecoder()
        self.feat_decoder = ZINBDecoder(latent_dim, hidden_dim, n_genes)
        self.cluster = GMCMClustering(n_clusters, latent_dim)

    def forward(self, x_input, edge_index):
        z, mu, logvar = self.encoder(x_input, edge_index)
        adj_logits = self.adj_decoder(z)
        mean, disp, pi = self.feat_decoder(z)
        return z, mu, logvar, adj_logits, (mean, disp, pi)

    def loss(self, x_input, edge_index, adj, x_counts, scale_factor=1.0,
             w_adj=1.0, w_feat=1.0, w_clus=1.0, w_kl=1.0):
        z, mu, logvar, adj_logits, (mean, disp, pi) = self.forward(x_input, edge_index)
        L_adj = topology_recon_loss(adj_logits, adj)
        L_feat = zinb_loss(x_counts, mean, disp, pi, scale_factor)
        L_clus = self.cluster.gmcm_nll(z)
        L_kl = kl_divergence(mu, logvar)
        total = w_adj * L_adj + w_feat * L_feat + w_clus * L_clus + w_kl * L_kl
        parts = {"adj": L_adj.item(), "feat": L_feat.item(),
                 "clus": L_clus.item(), "kl": L_kl.item(), "total": total.item()}
        return total, parts