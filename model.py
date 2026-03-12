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
from torch_geometric.nn import GCNConv, VGAE, GraphSAGE

device = get_device()

gamma       = 1.0
alpha_init  = 1.0
beta_init   = 1.0
min_delta   = 1e-4
patience    = 50


# ─────────────────────────────────────────────────────────────────────────────
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels,
                 activation=torch.relu):
        super().__init__()
        self.conv1       = GraphSAGE(in_channels,     hidden_channels, num_layers=2)
        self.conv_mu     = GraphSAGE(hidden_channels, latent_channels, num_layers=2)
        self.conv_logvar = GraphSAGE(hidden_channels, latent_channels, num_layers=2)
        self.activation  = activation

    def forward(self, x, edge_index):
        h = self.activation(self.conv1(x, edge_index))
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)


# ─────────────────────────────────────────────────────────────────────────────
class GMCM_VGAE(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.num_neurons    = kwargs['num_neurons']
        self.num_features   = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters      = kwargs['nClusters']
        self.tau_rank       = kwargs['tau_rank']
        self.gmcm_dim       = kwargs['gmcm_dim']

        act_map = {
            "ReLU":    torch.relu,
            "Sigmoid": torch.sigmoid,
            "Tanh":    torch.tanh,
            "Linear":  lambda x: x,
        }
        self.activation = act_map[kwargs['activation']]

        self.seed = kwargs['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # ── FIX 1: build each sub-module exactly once ──────────────────────
        self.encoder      = GCNEncoder(self.num_features, self.num_neurons,
                                       self.embedding_size, self.activation).to(device)
        self.vgae         = VGAE(self.encoder).to(device)
        self.zinb_decoder = ZINBDecoder(self.embedding_size,
                                        self.num_features).to(device)
        self.projector    = GMCMProjector(in_dim=self.embedding_size,
                                          out_dim=self.gmcm_dim).to(device)
        self.gmcm         = GMCM(n_components=self.nClusters,
                                 n_features=self.gmcm_dim,
                                 tau_rank=self.tau_rank).to(device)
        self.weights      = LossWeights(alpha_init=alpha_init,
                                        beta_init=beta_init).to(device)

    # ─────────────────────────────────────────────────────────────────────
    def Calculate_Loss(self, z, data, mu, theta, pi):
        # Edge reconstruction
        pos_edge_index, _ = self.get_pos_neg_edges(data)
        recon_loss = self.vgae.recon_loss(z, pos_edge_index)

        # KL (per-node average)
        kl = (1.0 / data.num_nodes) * self.vgae.kl_loss()

        # ZINB
        zinb_loss = self.zinb_nll(data.x, mu, theta, pi)

        # GMCM: project → copula → GMM NLL
        zc = self.projector(z)
        zc = (zc - zc.mean(0)) / (zc.std(0) + 1e-6)
        resp, gmcm_nll = self.gmcm(zc)

        alpha, beta = self.weights()
        total = recon_loss + alpha * zinb_loss + beta * kl + gamma * gmcm_nll

        # ── FIX 3: entropy regularisation — MAXIMISE entropy to prevent collapse
        # H = -sum p log p  (positive); we SUBTRACT lam*H to add it as a bonus
        lam_ent = 0.01
        p       = resp.clamp_min(1e-9)
        entropy = -(p * p.log()).sum(dim=1).mean()   # >0
        total   = total - lam_ent * entropy           # reward high entropy

        # ── FIX 2: return order matches train() unpacking exactly ──────────
        # order: total, recon, zinb, kl, gmcm_nll, resp, alpha, beta
        return total, recon_loss, zinb_loss, kl, gmcm_nll, resp, alpha, beta

    # ─────────────────────────────────────────────────────────────────────
    def train_model(self, data, optimizer, epochs, lr, wd, momentum,
                    save_path, dataset):
        """Renamed from train() to avoid shadowing nn.Module.train()."""
        data = data.to(device)

        opt_cls = {"Adam": Adam, "SGD": SGD, "RMSProp": RMSprop}[optimizer]
        opti = opt_cls(
            list(self.vgae.parameters()) +
            list(self.zinb_decoder.parameters()) +
            list(self.projector.parameters()) +
            list(self.gmcm.parameters()) +
            list(self.weights.parameters()),
            lr=lr, weight_decay=wd
        )

        os.makedirs(save_path + dataset + '/cluster', exist_ok=True)
        logfile   = open(save_path + dataset + '/cluster/log.csv', 'w')
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['iter', 'ari', 'nmi', 'Loss_total'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))
        print('Training......')

        best_ari   = -1.0
        bad_epochs = 0
        best_state = None

        for epoch in epoch_bar:
            # ── set sub-modules to training mode explicitly ─────────────
            self.vgae.train()
            self.zinb_decoder.train()
            self.projector.train()
            self.gmcm.train()
            self.weights.train()

            opti.zero_grad()

            x          = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y          = data.y.to(device)

            z               = self.vgae.encode(x, edge_index)
            mu, theta, pi   = self.zinb_decoder(z)

            # ── FIX 2: unpack in the order Calculate_Loss actually returns ─
            (Loss_total, Loss_recons, Loss_zinb,
             Loss_kl, Loss_gmcm, resp, alpha, beta) = \
                self.Calculate_Loss(z, data, mu, theta, pi)

            Loss_total.backward()

            # ── FIX 7: gradient clipping prevents exploding gradients ──────
            torch.nn.utils.clip_grad_norm_(
                list(self.vgae.parameters()) +
                list(self.zinb_decoder.parameters()) +
                list(self.projector.parameters()) +
                list(self.gmcm.parameters()) +
                list(self.weights.parameters()),
                max_norm=5.0
            )

            opti.step()

            ari, nmi, acc = self.eval_clustering_from_resp(resp, y)

            # ── FIX 8: actually use bad_epochs for early stopping ──────────
            if ari > best_ari + min_delta:
                best_ari   = ari
                bad_epochs = 0
                best_state = {
                    "vgae": {k: v.detach().cpu().clone()
                             for k, v in self.vgae.state_dict().items()},
                    "zinb": {k: v.detach().cpu().clone()
                             for k, v in self.zinb_decoder.state_dict().items()},
                    "proj": {k: v.detach().cpu().clone()
                             for k, v in self.projector.state_dict().items()},
                    "gmcm": {k: v.detach().cpu().clone()
                             for k, v in self.gmcm.state_dict().items()},
                    "w":    {k: v.detach().cpu().clone()
                             for k, v in self.weights.state_dict().items()},
                }
                torch.save(best_state,
                           save_path + dataset + "/cluster/best_by_ari.pt")


            logwriter.writerow({'iter': epoch+1, 'ari': ari, 'nmi': nmi,
                                 'Loss_total': Loss_total.item()})

            if epoch == 0 or (epoch + 1) % 10 == 0:
                epoch_bar.write(
                    f"epoch={epoch+1:4d}  loss={Loss_total.item():.4f}  "
                    f"recon={Loss_recons.item():.4f}  zinb={Loss_zinb.item():.4f}  "
                    f"kl={Loss_kl.item():.4f}  gmcm={Loss_gmcm.item():.4f}  "
                    f"α={alpha:.3g}  β={beta:.3g}  "
                    f"ARI={ari:.4f}  NMI={nmi:.4f}  ACC={acc:.4f}"
                )

        logfile.close()

        if best_state is not None:
            self.vgae.load_state_dict(best_state["vgae"])
            self.zinb_decoder.load_state_dict(best_state["zinb"])
            self.projector.load_state_dict(best_state["proj"])
            self.gmcm.load_state_dict(best_state["gmcm"])
            self.weights.load_state_dict(best_state["w"])

        print(f"Best ARI={best_ari:.4f}")
        return ari, nmi, acc

    # ── helpers (unchanged logic, kept for completeness) ──────────────────
    def get_pos_neg_edges(self, data):
        if hasattr(data, "pos_edge_label_index") and \
           hasattr(data, "neg_edge_label_index"):
            return data.pos_edge_label_index, data.neg_edge_label_index
        if hasattr(data, "pos_edge_index") and \
           hasattr(data, "neg_edge_index"):
            return data.pos_edge_index, data.neg_edge_index
        if hasattr(data, "edge_label_index") and \
           hasattr(data, "edge_label"):
            idx = data.edge_label_index
            y   = data.edge_label
            return idx[:, y == 1], idx[:, y == 0]
        keys = data.keys() if callable(getattr(data, "keys", None)) else []
        raise RuntimeError(
            f"No pos/neg edge attributes found. Keys: {keys}")

    def zinb_nll(self, x, mu, theta, pi, eps=1e-8):
        t1 = (torch.lgamma(theta + x)
              - torch.lgamma(theta)
              - torch.lgamma(x + 1.0))
        t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        nb_log_prob = t1 + t2 + t3

        x_is_zero      = (x < 0.5).type_as(x)
        log_pi         = torch.log(pi + eps)
        log_1m_pi      = torch.log(1.0 - pi + eps)
        zero_case      = torch.logaddexp(log_pi, log_1m_pi + nb_log_prob)
        nonzero_case   = log_1m_pi + nb_log_prob
        zinb_log_prob  = x_is_zero * zero_case + (1.0 - x_is_zero) * nonzero_case
        return (-zinb_log_prob).mean()

    def clustering_accuracy(self, y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.int64)
        y_pred = np.asarray(y_pred).astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_true.size):
            w[y_pred[i], y_true[i]] += 1
        r, c = linear_sum_assignment(w.max() - w)
        return w[r, c].sum() / y_true.size

    def clustering_scores(self, y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.int64)
        y_pred = np.asarray(y_pred).astype(np.int64)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(
            y_true, y_pred, average_method="arithmetic")
        acc = self.clustering_accuracy(y_true, y_pred)
        return ari, nmi, acc

    @torch.no_grad()
    def eval_clustering_from_resp(self, resp, y_true):
        y_pred = resp.argmax(dim=1).cpu().numpy()
        y_true = y_true.cpu().numpy()
        return self.clustering_scores(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
class ZINBDecoder(nn.Module):
    def __init__(self, latent_dim, n_genes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head    = nn.Linear(hidden_dim, n_genes)
        self.theta_head = nn.Linear(hidden_dim, n_genes)
        self.pi_head    = nn.Linear(hidden_dim, n_genes)

    def forward(self, z):
        h     = self.net(z)
        mu    = F.softplus(self.mu_head(h))    + 1e-4
        theta = F.softplus(self.theta_head(h)) + 1e-4
        pi    = torch.sigmoid(self.pi_head(h))
        return mu, theta, pi


# ─────────────────────────────────────────────────────────────────────────────
class LossWeights(nn.Module):
    """
    FIX 4: store raw (unconstrained) parameters directly;
    softplus alone is the correct positive mapping.
    """
    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        # initialise so softplus(p) ≈ alpha_init
        self._a = nn.Parameter(torch.tensor(
            float(np.log(np.exp(alpha_init) - 1))))
        self._b = nn.Parameter(torch.tensor(
            float(np.log(np.exp(beta_init)  - 1))))

    def forward(self):
        alpha = F.softplus(self._a).clamp(1e-3, 1e3)
        beta  = F.softplus(self._b).clamp(1e-3, 1e3)
        return alpha, beta


# ─────────────────────────────────────────────────────────────────────────────
class GMCMProjector(nn.Module):
    """
    FIX 6: added hidden layer + nonlinearity so the projection is non-trivial.
    """
    def __init__(self, in_dim=256, out_dim=16):
        super().__init__()
        hidden = max(in_dim // 2, out_dim * 2)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z):
        return self.proj(z)


# ─────────────────────────────────────────────────────────────────────────────
def _soft_rank_1d(x: torch.Tensor, tau: float) -> torch.Tensor:
    x    = x.view(-1, 1)
    diff = (x - x.t()) / tau
    P    = torch.sigmoid(diff)
    return 1.0 + P.sum(dim=1)


def copula_normal_scores_soft(Z: torch.Tensor,
                               tau_rank: float = 0.1,
                               eps: float = 1e-4) -> torch.Tensor:
    N, D = Z.shape
    s2   = torch.sqrt(torch.tensor(2.0, device=Z.device, dtype=Z.dtype))
    cols = []
    for j in range(D):
        r = _soft_rank_1d(Z[:, j], tau=tau_rank)
        u = (r / (N + 1.0)).clamp(eps, 1.0 - eps)
        y = s2 * torch.erfinv(2.0 * u - 1.0)
        cols.append(y)
    return torch.stack(cols, dim=1)


class GMCM(nn.Module):
    def __init__(self, n_components, n_features,
                 tau_rank=0.1, eps=1e-4, jitter=1e-4):
        super().__init__()
        self.K        = int(n_components)
        self.D        = int(n_features)
        self.tau_rank = float(tau_rank)
        self.eps      = float(eps)
        self.jitter   = float(jitter)

        self.logits          = nn.Parameter(torch.zeros(self.K))
        self.means           = nn.Parameter(torch.randn(self.K, self.D) * 0.01)
        self.L_unconstrained = nn.Parameter(
            torch.zeros(self.K, self.D, self.D))
        nn.init.normal_(self.L_unconstrained, mean=0.0, std=0.01)

    def _cholesky(self):
        L        = torch.tril(self.L_unconstrained)
        diag     = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag) + self.jitter
        return L - torch.diag_embed(diag) + torch.diag_embed(diag_pos)

    def _log_prob_y_given_k(self, Y):
        N, D     = Y.shape
        L        = self._cholesky()                          # (K,D,D)
        diff     = Y[:, None, :] - self.means[None, :, :]   # (N,K,D)
        diff_kdn = diff.permute(1, 2, 0)                     # (K,D,N)
        v        = torch.linalg.solve_triangular(
            L, diff_kdn, upper=False)                        # (K,D,N)
        quad     = (v * v).sum(dim=1).permute(1, 0)         # (N,K)
        logdet   = 2.0 * torch.log(
            torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=1) # (K,)
        const    = D * math.log(2.0 * math.pi)
        return -0.5 * (quad + logdet[None, :] + const)

    def forward(self, Z):
        Y        = copula_normal_scores_soft(
            Z, tau_rank=self.tau_rank, eps=self.eps)
        log_pi   = F.log_softmax(self.logits, dim=0)
        log_p_yk = self._log_prob_y_given_k(Y)
        log_joint = log_p_yk + log_pi[None, :]
        nll      = -torch.logsumexp(log_joint, dim=1).mean()
        resp     = torch.softmax(log_joint, dim=1)
        return resp, nll


# ─────────────────────────────────────────────────────────────────────────────
class clustering_metrics:
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        l1 = list(set(self.true_label))
        l2 = list(set(self.pred_label))
        if len(l1) != len(l2):
            return 0
        D    = len(l1)
        cost = np.zeros((D, D), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                cost[i][j] = sum(1 for i1 in mps if self.pred_label[i1] == c2)
        m       = Munkres()
        indexes = m.compute((-cost).tolist())
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        return metrics.accuracy_score(self.true_label, new_predict)

    def evaluationClusterModelFromLabel(self):
        nmi      = metrics.normalized_mutual_info_score(
            self.true_label, self.pred_label)
        adjscore = adjusted_rand_score(self.true_label, self.pred_label)
        acc      = self.clusteringAcc()
        return acc, nmi, adjscore