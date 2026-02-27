from collections import defaultdict
import numpy as np
import torch
from scipy.stats import norm, rankdata
from sklearn.mixture import GaussianMixture
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import math
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from munkres import Munkres
from copulae.mixtures.gmc.gmc import GaussianMixtureCopula
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import csv, os
from torch_geometric.nn import GCNConv,VGAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gamma = 1.0
tau_rank = 0.1
alpha_init = 1.0
beta_init = 1.0
min_delta = 1e-4
patience = 30

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels,activation=torch.relu):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_channels)
        self.conv_logvar = GCNConv(hidden_channels, latent_channels)
        self.activation = activation

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation(h)
        return self.conv_mu(h, edge_index), self.conv_logvar(h, edge_index)


class GMCM_VGAE(nn.Module):
    """A Gaussian Mixture Copula Model based variational graph autoencoder

    Args:
        nn: Inputs for intialization
    """

    def __init__(self, **kwargs):
        super(GMCM_VGAE, self).__init__()
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        self.min_clamp_mean = kwargs['min_clamp_mean']
        self.max_clamp_mean = kwargs['max_clamp_mean']
        self.min_clamp_dis = kwargs['min_clamp_dis']
        self.max_clamp_dis = kwargs['max_clamp_dis']
        if kwargs['activation'] == "ReLU":
            self.activation = torch.relu
        elif kwargs['activation'] == "Sigmoid":
            self.activation = torch.sigmoid
        elif kwargs['activation'] == "Tanh":
            self.activation = torch.tanh
        elif kwargs["activation"]=="Linear":
            self.activation = torch.nn.Identity
        self.seed = kwargs['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # VGAE training parameters
        self.encoder= GCNEncoder(self.num_features, self.num_neurons, self.embedding_size, self.activation)
        self.vgae= VGAE(self.encoder)
        self.zinb_decoder= ZINBDecoder(self.embedding_size, self.num_features)

        # Clustering parameters initialization
        self.cluster_head = ClusterHead(self.embedding_size, self.nClusters)

        self.gmcm_dim = kwargs.get("gmcm_dim", 32)  # default 16

        # VGAE + ZINB
        self.encoder = GCNEncoder(self.num_features, self.num_neurons, self.embedding_size, self.activation)
        self.vgae = VGAE(self.encoder).to(device)
        self.zinb_decoder = ZINBDecoder(self.embedding_size, self.num_features).to(device)

        # GMCM components (end-to-end)
        self.projector = GMCMProjector(in_dim=self.embedding_size, out_dim=self.gmcm_dim).to(device)
        self.gmcm = DiagGMM(n_components=self.nClusters, n_features=self.gmcm_dim).to(device)

        # Learnable weights
        self.weights = LossWeights(alpha_init=alpha_init, beta_init=beta_init).to(device)

    def Calculate_Loss(self, z, data, mu, theta, pi):
        # Edge reconstruction
        pos_edge_index, _ = self.get_pos_neg_edges(data)
        recon_loss = self.vgae.recon_loss(z, pos_edge_index)

        # KL
        kl = (1.0 / data.num_nodes) * self.vgae.kl_loss()

        # ZINB
        zinb_loss = self.zinb_nll(data.x, mu, theta, pi)

        # GMCM (project -> copula -> gmm)
        zc = self.projector(z)  # (N, gmcm_dim)
        Y = self.copula_normal_scores_soft(zc, tau_rank=tau_rank)  # (N, gmcm_dim)
        resp, gmcm_nll = self.gmcm(Y)

        # Learnable alpha/beta
        alpha, beta = self.weights()

        total = recon_loss + alpha * zinb_loss + beta * kl + gamma * gmcm_nll
        return total, recon_loss, gmcm_nll, zinb_loss, kl, resp, alpha, beta

    def train(self, data, optimizer, epochs, lr,wd,momentum, save_path,
              dataset):

        if optimizer == "Adam":
            optim0 = Adam
        elif optimizer == "SGD":
            optim0 = SGD
        elif optimizer == "RMSProp":
            optim0= RMSprop

        opti = optim0(
            list(self.vgae.parameters()) +
            list(self.zinb_decoder.parameters()) +
            list(self.projector.parameters()) +
            list(self.gmcm.parameters()) +
            list(self.weights.parameters()),
            lr=lr, weight_decay=wd
        )


        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Logging the resluts
        os.makedirs(save_path + dataset + '/cluster',exist_ok=True)
        logfile = open(save_path + dataset + '/cluster/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ari', 'nmi', 'Loss_total'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))

        print('Training......')

        count = 0
        currmax = 0
        finalist = []
        acc_list = []
        loss_list=defaultdict(list)
        best_ari = -1.0
        bad_epochs = 0
        best_state = None

        for epoch in epoch_bar:
            self.vgae.train()
            self.zinb_decoder.train()
            self.projector.train()
            self.gmcm.train()
            self.weights.train()

            opti.zero_grad()

            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            y = data.y.to(device)

            z = self.vgae.encode(x, edge_index)  # (N, embedding_size)
            mu, theta, pi = self.zinb_decoder(z)

            Loss_total, Loss_recons, Loss_gmcm, Loss_zinb, Loss_kl, resp, alpha, beta = \
                self.Calculate_Loss(z, data, mu, theta, pi)

            Loss_total.backward()
            opti.step()

            # metrics (train). If you have val_data, evaluate on val_data instead.
            ari, nmi, acc = self.eval_clustering_from_resp(resp, y)

            # early stopping on ARI
            improved = (ari > best_ari + min_delta)
            if improved:
                best_ari = ari
                bad_epochs = 0
                best_state = {
                    "vgae": {k: v.detach().cpu().clone() for k, v in self.vgae.state_dict().items()},
                    "zinb": {k: v.detach().cpu().clone() for k, v in self.zinb_decoder.state_dict().items()},
                    "proj": {k: v.detach().cpu().clone() for k, v in self.projector.state_dict().items()},
                    "gmcm": {k: v.detach().cpu().clone() for k, v in self.gmcm.state_dict().items()},
                    "w": {k: v.detach().cpu().clone() for k, v in self.weights.state_dict().items()},
                }
                torch.save(best_state, save_path + dataset + "/cluster/best_by_ari.pt")
            else:
                bad_epochs += 1

            if epoch == 0 or (epoch + 1) % 10 == 0:
                epoch_bar.write(
                    f"epoch={epoch + 1} loss={Loss_total.item():.4f} "
                    f"recon={Loss_recons.item():.4f} zinb={Loss_zinb.item():.4f} "
                    f"kl={Loss_kl.item():.4f} gmcm={Loss_gmcm.item():.4f} "
                    f"alpha={alpha:.3g} beta={beta:.3g} | ARI={ari:.4f} NMI={nmi:.4f} ACC={acc:.4f}"
                )

            if bad_epochs >= patience:
                epoch_bar.write(f"Early stopping at epoch {epoch + 1}. Best ARI={best_ari:.4f}")
                break

        # restore best
        if best_state is not None:
            self.vgae.load_state_dict(best_state["vgae"])
            self.zinb_decoder.load_state_dict(best_state["zinb"])
            self.projector.load_state_dict(best_state["proj"])
            self.gmcm.load_state_dict(best_state["gmcm"])
            self.weights.load_state_dict(best_state["w"])

        torch.save(best_state, save_path + dataset + "/cluster/best_by_ari.pt")
        print(f"Best ARI={best_ari:.4f}")
        return ari, nmi, acc


    def soft_rank_1d(self,x, tau=0.1):
        """
        x: (N,) tensor
        Returns approx ranks in (1..N), differentiable.
        rank_i = 1 + sum_j sigmoid((x_i - x_j)/tau)
        """
        x = x.view(-1, 1)  # (N,1)
        diff = (x - x.t()) / tau  # (N,N)
        P = torch.sigmoid(diff)  # approx I[x_i > x_j]
        return 1.0 + P.sum(dim=1)  # (N,)

    def copula_normal_scores_soft(self,Z, tau_rank=0.1, eps=1e-6):
        """
        Z: (N,d) torch
        Returns Y: (N,d) with Gaussian marginals via soft ranks.
        """
        N, d = Z.shape
        Y_cols = []
        for j in range(d):
            r = self.soft_rank_1d(Z[:, j], tau=tau_rank)  # (N,)
            u = r / (N + 1.0)  # (0,1)
            u = u.clamp(eps, 1.0 - eps)
            # Phi^{-1}(u) = sqrt(2)*erfinv(2u-1)
            y = torch.sqrt(torch.tensor(2.0, device=Z.device)) * torch.erfinv(2.0 * u - 1.0)
            Y_cols.append(y)
        return torch.stack(Y_cols, dim=1)  # (N,d)

    def fit_gmcm_clusters(self, Z_torch, n_clusters, covariance_type="full", random_state=0):
        """
        Z_torch: (N, d) torch tensor
        Returns:
          resp: (N, K) numpy responsibilities
          labels: (N,) numpy hard labels
          gmm: fitted sklearn GMM
        """
        Z_np = Z_torch.detach().cpu().numpy()
        Y = self.copula_normal_scores(Z_np)

        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(Y)
        resp = gmm.predict_proba(Y)
        labels = resp.argmax(axis=1)
        return resp, labels, gmm

    def gmcm_cluster_loss(self,logits, resp_np):
        """
        logits: (N,K) torch
        resp_np: (N,K) numpy responsibilities from GMCM/GMM
        """
        target = torch.from_numpy(resp_np).to(logits.device).float()
        logp = F.log_softmax(logits, dim=1)
        # cross-entropy soft: -sum_k q_k log p_k
        return -(target * logp).sum(dim=1).mean()
    def get_pos_neg_edges(self,data):
        # Newer PyG:
        if hasattr(data, "pos_edge_label_index") and hasattr(data, "neg_edge_label_index"):
            return data.pos_edge_label_index, data.neg_edge_label_index

        # Older PyG:
        if hasattr(data, "pos_edge_index") and hasattr(data, "neg_edge_index"):
            return data.pos_edge_index, data.neg_edge_index

        # Unified label format:
        if hasattr(data, "edge_label_index") and hasattr(data, "edge_label"):
            idx = data.edge_label_index
            y = data.edge_label
            pos = idx[:, y == 1]
            neg = idx[:, y == 0]
            return pos, neg

        # If nothing matched, fail loudly with keys:
        keys = data.keys() if callable(getattr(data, "keys", None)) else []
        raise RuntimeError(f"No pos/neg edge attributes found. Available keys: {keys}")

    def zinb_nll(self,x, mu, theta, pi, eps=1e-8):
        """
        x: (N, G) raw counts (non-negative, integer-valued recommended)
        mu: (N, G) mean > 0
        theta: (N, G) dispersion > 0
        pi: (N, G) dropout probability in (0,1)
        returns: scalar loss (mean over entries)
        """
        # NB log-prob (using stable log-gamma form)
        # log NB(x | mu, theta)
        t1 = torch.lgamma(theta + x) - torch.lgamma(theta) - torch.lgamma(x + 1.0)
        t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        nb_log_prob = t1 + t2 + t3  # (N,G)

        # ZINB: if x==0: log( pi + (1-pi)*exp(nb_log_prob) ), else: log(1-pi)+nb_log_prob
        x_is_zero = (x < 0.5).type_as(x)  # safe for float x

        log_pi = torch.log(pi + eps)
        log_1m_pi = torch.log(1.0 - pi + eps)

        zero_case = torch.logaddexp(log_pi, log_1m_pi + nb_log_prob)  # (N,G)
        nonzero_case = log_1m_pi + nb_log_prob  # (N,G)

        zinb_log_prob = x_is_zero * zero_case + (1.0 - x_is_zero) * nonzero_case
        return (-zinb_log_prob).mean()


    def decodeZINB(self, z):
        m = self.Mean(z)
        d = self.Dispersion(z)
        p = self.Dropout(z)
        # extra=(m,d,p)
        extra = (m, d, p)
        return extra


    def clustering_accuracy(self,y_true, y_pred):
        """
        ACC via Hungarian matching.
        y_true, y_pred: 1D arrays of ints
        """
        y_true = np.asarray(y_true).astype(np.int64)
        y_pred = np.asarray(y_pred).astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_true.size):
            w[y_pred[i], y_true[i]] += 1
        r, c = linear_sum_assignment(w.max() - w)
        return w[r, c].sum() / y_true.size

    def clustering_scores(self,y_true, y_pred):
        y_true = np.asarray(y_true).astype(np.int64)
        y_pred = np.asarray(y_pred).astype(np.int64)
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
        acc = self.clustering_accuracy(y_true, y_pred)
        return ari, nmi, acc

    @torch.no_grad()
    def eval_clustering_from_resp(self,resp, y_true):
        y_pred = resp.argmax(dim=1).detach().cpu().numpy()
        y_true = y_true.detach().cpu().numpy()
        return self.clustering_scores(y_true, y_pred)


def random_uniform_init(input_dim, output_dim, seed):
    np.random.seed(seed)
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    torch.manual_seed(seed)
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.true_label))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        acc = metrics.accuracy_score(self.true_label, new_predict)

        return acc

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = adjusted_rand_score(self.true_label, self.pred_label)
        acc = self.clusteringAcc()

        return acc, nmi, adjscore



class ClusterHead(nn.Module):
    def __init__(self, latent_dim, n_clusters):
        super().__init__()
        self.lin = nn.Linear(latent_dim, n_clusters)

    def forward(self, z):
        return self.lin(z)  # logits (N, K)

class ZINBDecoder(nn.Module):
    """
    Takes latent Z (N, d) and outputs ZINB parameters per gene (N, G):
    - mu: mean (positive)
    - theta: dispersion (positive)
    - pi: dropout prob (0..1)
    """
    def __init__(self, latent_dim, n_genes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, n_genes)
        self.theta_head = nn.Linear(hidden_dim, n_genes)
        self.pi_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, z):
        h = self.net(z)
        mu = F.softplus(self.mu_head(h)) + 1e-4         # >0
        theta = F.softplus(self.theta_head(h)) + 1e-4   # >0
        pi = torch.sigmoid(self.pi_head(h))             # (0,1)
        return mu, theta, pi

class UncertaintyWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_edge = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_zinb = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_kl   = nn.Parameter(torch.tensor(0.0))

    def forward(self, edge_loss, zinb_loss, kl_loss):
        s1 = self.log_sigma_edge
        s2 = self.log_sigma_zinb
        s3 = self.log_sigma_kl
        loss = (
            0.5 * torch.exp(-2*s1) * edge_loss + s1 +
            0.5 * torch.exp(-2*s2) * zinb_loss + s2 +
            0.5 * torch.exp(-2*s3) * kl_loss   + s3
        )
        return loss

#Diagonal covariance (stable, simple)
class DiagGMM(nn.Module):
    def __init__(self, n_components, n_features):
        super().__init__()
        self.K = n_components
        self.D = n_features
        self.logits = nn.Parameter(torch.zeros(self.K))                 # mixing logits
        self.means  = nn.Parameter(torch.randn(self.K, self.D) * 0.01)  # mu_k
        self.log_vars = nn.Parameter(torch.zeros(self.K, self.D))       # log(sigma^2)

    def log_prob_per_comp(self, Y):
        """
        Y: (N,D)
        returns log p(Y | k) : (N,K)
        """
        N, D = Y.shape
        log_vars = self.log_vars.clamp(-10.0, 10.0)
        vars_ = torch.exp(log_vars)                                      # (K,D)

        # (N,1,D) - (1,K,D) -> (N,K,D)
        diff = Y[:, None, :] - self.means[None, :, :]
        quad = (diff * diff) / (vars_[None, :, :] + 1e-8)               # (N,K,D)

        log_det = log_vars.sum(dim=1)                                   # (K,)
        const = D * math.log(2.0 * math.pi)

        return -0.5 * (quad.sum(dim=2) + log_det[None, :] + const)      # (N,K)

    def forward(self, Y):
        """
        returns:
          resp: (N,K) responsibilities
          nll:  scalar negative log-likelihood
        """
        log_pi = F.log_softmax(self.logits, dim=0)                      # (K,)
        log_p_yk = self.log_prob_per_comp(Y)                            # (N,K)
        log_joint = log_p_yk + log_pi[None, :]                          # (N,K)
        log_p_y = torch.logsumexp(log_joint, dim=1)                     # (N,)
        nll = -log_p_y.mean()
        resp = torch.softmax(log_joint, dim=1)                          # (N,K)
        return resp, nll

class LossWeights(nn.Module):
    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self._a = nn.Parameter(torch.tensor(float(alpha_init)).log())
        self._b = nn.Parameter(torch.tensor(float(beta_init)).log())

    def forward(self):
        alpha = (F.softplus(self._a) + 1e-8).clamp(1e-3, 1e3)
        beta  = (F.softplus(self._b) + 1e-8).clamp(1e-3, 1e3)
        return alpha, beta

class GMCMProjector(nn.Module):
    def __init__(self, in_dim=256, out_dim=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, z):
        return self.proj(z)