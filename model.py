import os
from collections import defaultdict

os.environ["OMP_NUM_THREADS"] = "15"

import torch
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy import stats
from scipy.stats import norm
from munkres import Munkres

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set all random seeds for reproducibility - call ONCE at program start"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaussianMixtureCopulaModel:
    """Gaussian Mixture Copula Model with reproducible random state."""

    def __init__(self, n_clusters, random_state=None, use_copula=True, covariance_type='full'):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_copula = use_copula
        self.covariance_type = covariance_type
        self.gmm = None
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def _to_pseudo_observations(self, X):
        n_samples, n_features = X.shape
        U = np.zeros_like(X)
        for j in range(n_features):
            ranks = stats.rankdata(X[:, j], method='ordinal')
            U[:, j] = ranks / (n_samples + 1)
        return U

    def _to_normal(self, U):
        U_clipped = np.clip(U, 1e-10, 1 - 1e-10)
        return norm.ppf(U_clipped)

    def _transform(self, X):
        if self.use_copula:
            U = self._to_pseudo_observations(X)
            return self._to_normal(U)
        return X

    def fit(self, X, method='kmeans', eps=1e-4, max_iter=300):
        Z = self._transform(X)

        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=3,
            init_params='k-means++' if method == 'kmeans' else 'random',
            tol=eps,
            max_iter=max_iter,
            reg_covar=1e-6  # Add regularization to avoid singular covariance
        )
        self.gmm.fit(Z)

        self.weights_ = self.gmm.weights_
        self.means_ = self.gmm.means_

        if self.covariance_type == 'full':
            self.covariances_full_ = self.gmm.covariances_
            self.covariances_ = np.array([np.diag(self.gmm.covariances_[k]) for k in range(self.n_clusters)])
        else:
            self.covariances_ = self.gmm.covariances_
            self.covariances_full_ = None

        return self


class GraphConvSparse(nn.Module):
    def __init__(self, rng, input_dim, output_dim, adj, activation=torch.sigmoid, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = rng.uniform(-init_range, init_range, (input_dim, output_dim)).astype(np.float32)
        self.weight = nn.Parameter(torch.from_numpy(initial))
        self.adj = adj
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class GMCM_VGAE(nn.Module):
    """A Gaussian Mixture Copula Model based variational graph autoencoder"""

    def __init__(self, **kwargs):
        super(GMCM_VGAE, self).__init__()
        self.adj = kwargs['adj']
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        self.use_copula = kwargs.get('use_copula', True)
        self.seed = kwargs['seed']

        if kwargs['activation'] == "ReLU":
            self.activation = torch.relu
        elif kwargs['activation'] == "Sigmoid":
            self.activation = torch.sigmoid
        elif kwargs['activation'] == "Tanh":
            self.activation = torch.tanh

        # Create a master RNG for all initialization
        self.master_rng = np.random.RandomState(self.seed)

        # Pre-generate ALL random values needed for training
        # This ensures complete reproducibility
        self._init_model()

    def _init_model(self):
        """Initialize all model components with deterministic randomness"""
        rng = self.master_rng

        # VGAE layers
        self.base_gcn = GraphConvSparse(rng, self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(rng, self.num_neurons, self.embedding_size, self.adj, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(rng, self.num_neurons, self.embedding_size, self.adj,
                                             activation=lambda x: x)

        # Clustering parameters
        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        mu_c_init = rng.randn(self.nClusters, self.embedding_size).astype(np.float32)
        self.mu_c = nn.Parameter(torch.from_numpy(mu_c_init), requires_grad=True)
        log_sigma2_c_init = rng.randn(self.nClusters, self.embedding_size).astype(np.float32)
        self.log_sigma2_c = nn.Parameter(torch.from_numpy(log_sigma2_c_init), requires_grad=True)

        # ZINB decoder - create with deterministic weights
        self.Mean = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), MeanAct())
        self.Dispersion = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), DispAct())
        self.Dropout = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), nn.Sigmoid())

        # Initialize ZINB layers
        for module in [self.Mean, self.Dispersion, self.Dropout]:
            linear = module[0]
            fan_in = linear.weight.size(1)
            fan_out = linear.weight.size(0)
            std = np.sqrt(2.0 / fan_in)
            bound = np.sqrt(3.0) * std

            w = rng.uniform(-bound, bound, (fan_out, fan_in)).astype(np.float32)
            linear.weight.data = torch.from_numpy(w)

            if linear.bias is not None:
                b_bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                b = rng.uniform(-b_bound, b_bound, (fan_out,)).astype(np.float32)
                linear.bias.data = torch.from_numpy(b)

    def ZINB_loss(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge
        result = torch.mean(result)
        return result

    def Calculate_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb):
        nClusters = self.nClusters
        pi = self.pi
        mu_c = self.mu_c
        log_sigma2_c = self.log_sigma2_c

        eps = 1e-10
        det = 1e-2

        # Loss weights for balancing
        w_recons = 1.0
        w_gmcm = 0.1  # Reduce GMCM weight since it dominates
        w_zinb = 1.0

        # Reconstructed Loss
        Loss_recons = norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight=weight_tensor)
        Loss_recons = Loss_recons * features.size(0)

        # Cluster GMCM loss
        log_pi = torch.log(pi.unsqueeze(0) + eps)
        log_pdfs = self.gmcm_gaussian_pdfs_log(emb, nClusters, mu_c, log_sigma2_c, pi)
        log_yita = log_pi + log_pdfs

        log_yita_max = torch.max(log_yita, dim=1, keepdim=True)[0]
        yita_c = torch.exp(log_yita - log_yita_max) + det
        yita_c = yita_c / (yita_c.sum(1, keepdim=True) + eps)

        log_sigma2_c_clamped = torch.clamp(log_sigma2_c, min=-20, max=20)

        KL1 = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2_c_clamped.unsqueeze(0) +
                                                            torch.exp(z_sigma2_log.unsqueeze(
                                                                1) - log_sigma2_c_clamped.unsqueeze(0)) +
                                                            (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / (torch.exp(
            log_sigma2_c_clamped.unsqueeze(0)) + eps), 2), 1))

        KL2 = torch.mean(torch.sum(yita_c * torch.log((pi.unsqueeze(0) + eps) / (yita_c + eps)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1))
        Loss_gmcm = KL1 - KL2

        # ZINB loss
        m, d, p = self.decodeZINB(emb)
        Loss_zinb = self.ZINB_loss(features.to_dense().squeeze(0), m, d, p)

        # Apply weights
        Loss_total = w_recons * Loss_recons + w_gmcm * Loss_gmcm + w_zinb * Loss_zinb

        return Loss_total, Loss_recons, Loss_gmcm, Loss_zinb

    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path,
              dataset, features_new):
        # Create a separate RNG for training noise - seeded deterministically
        self.train_rng = np.random.RandomState(self.seed + 999999)

        # Pre-generate all noise for all epochs upfront
        n_samples = features.size(0)
        self.all_noise = {}
        for ep in range(-1, epochs + 1):  # -1 for initial GMCM fit
            noise_rng = np.random.RandomState(self.seed + ep + 100000)
            self.all_noise[ep] = torch.from_numpy(
                noise_rng.randn(n_samples, self.embedding_size).astype(np.float32)
            ).to(device)

        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)

        import csv
        log_dir = os.path.join(save_path, dataset, 'cluster')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logfile = open(os.path.join(log_dir, 'log.csv'), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ari', 'nmi','Loss_total'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))
        print('Training......')

        # Initial GMCM fit
        with torch.no_grad():
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm, epoch=-1)
            emb_np = emb.detach().cpu().numpy()

            gmcm = GaussianMixtureCopulaModel(
                n_clusters=self.nClusters,
                random_state=self.seed,
                use_copula=self.use_copula,
                covariance_type='full' if self.use_copula else 'diag'
            )
            gmcm.fit(emb_np, method='kmeans', eps=0.0001, max_iter=300)

            self.pi.data = torch.from_numpy(gmcm.weights_.astype(np.float32)).to(device)
            self.mu_c.data = torch.from_numpy(gmcm.means_.astype(np.float32)).to(device)
            self.log_sigma2_c.data = torch.from_numpy(
                np.log(np.maximum(gmcm.covariances_, 1e-10)).astype(np.float32)).to(device)

        currmax = 0
        finalist = []
        loss_list = defaultdict(list)
        for epoch in epoch_bar:
            opti.zero_grad()

            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm, epoch=epoch)
            x_ = self.decode(emb)

            # Refit GMCM periodically
            if epoch > 0 and epoch % 50 == 0:
                with torch.no_grad():
                    emb_np = emb.detach().cpu().numpy()
                    gmcm = GaussianMixtureCopulaModel(
                        n_clusters=self.nClusters,
                        random_state=self.seed + epoch,
                        use_copula=self.use_copula,
                        covariance_type='full' if self.use_copula else 'diag'
                    )
                    gmcm.fit(emb_np, method='kmeans', eps=0.0001, max_iter=300)

                    self.pi.data = torch.from_numpy(gmcm.weights_.astype(np.float32)).to(device)
                    self.mu_c.data = torch.from_numpy(gmcm.means_.astype(np.float32)).to(device)
                    self.log_sigma2_c.data = torch.from_numpy(
                        np.log(np.maximum(gmcm.covariances_, 1e-10)).astype(np.float32)).to(device)

            Loss_total, Loss_recons, Loss_gmcm, Loss_zinb = self.Calculate_Loss(
                features, adj_norm, x_, adj_label.to_dense().view(-1), y, weight_tensor, norm, z_mu, z_sigma2_log, emb
            )

            loss_list['total'].append(Loss_total.detach().cpu().numpy())
            loss_list['Loss_recons'].append(Loss_recons.detach().cpu().numpy())
            loss_list['Loss_gmcm'].append(Loss_gmcm.detach().cpu().numpy())
            loss_list['Loss_zinb'].append(Loss_zinb.detach().cpu().numpy())

            if torch.isnan(Loss_total) or torch.isinf(Loss_total):
                print(f"Warning: NaN/Inf loss at epoch {epoch}")
                Loss_total = torch.tensor(1e6, device=device, requires_grad=True)

            epoch_bar.write('Loss={:.4f} (recons={:.4f}, gmcm={:.4f}, zinb={:.4f})'.format(
                Loss_total.detach().cpu().numpy(),
                Loss_recons.detach().cpu().numpy(),
                Loss_gmcm.detach().cpu().numpy(),
                Loss_zinb.detach().cpu().numpy()
            ))

            y_pred = self.predict_gmcm(emb, self.nClusters, self.mu_c, self.log_sigma2_c, self.pi)

            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
            acc_list.append(acc)

            logdict = dict(iter=epoch, ari=adjscore, nmi=nmi, Loss_total=Loss_total.detach().cpu().numpy())
            logwriter.writerow(logdict)
            logfile.flush()

            Loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

            opti.step()
            lr_s.step()

            if adjscore > currmax:
                finalist = [acc, adjscore, nmi, Loss_recons.detach().cpu().numpy(), Loss_gmcm.detach().cpu().numpy(),
                            Loss_zinb.detach().cpu().numpy(), Loss_total.detach().cpu().numpy(), epoch]
                currmax = adjscore

        return finalist, y_pred, y,loss_list

    def gmcm_gaussian_pdfs_log(self, x, nClusters, mus, log_sigma2s, pies):
        G = []
        for c in range(nClusters):
            G.append(self.gmcm_gaussian_pdf_log(x, mus[c, :], log_sigma2s[c, :], pies[c]).view(-1, 1))
        return torch.cat(G, 1)

    def gmcm_gaussian_pdf_log(self, x, mu, log_sigma2, pi):
        log_sigma2 = torch.clamp(log_sigma2, min=-20, max=20)
        c = -0.5 * torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / (torch.exp(log_sigma2) + 1e-10), 1)
        return c

    def predict_gmcm(self, x, nClusters, mu_c, log_sigma2_c, pi_c):
        eps = 1e-10
        log_pi = torch.log(pi_c.unsqueeze(0) + eps)
        log_pdfs = self.gmcm_gaussian_pdfs_log(x, nClusters, mu_c, log_sigma2_c, pi_c)

        log_prob = log_pi + log_pdfs
        log_prob_max = torch.max(log_prob, dim=1, keepdim=True)[0]
        log_prob_stable = log_prob - log_prob_max

        kappa_c = torch.exp(log_prob_stable)
        kappa_c = kappa_c / (kappa_c.sum(dim=1, keepdim=True) + eps)

        kappa = kappa_c.detach().cpu().numpy()
        return np.argmax(kappa, axis=1)

    def encode(self, x_features, adj, epoch=None):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)

        self.logstd = torch.clamp(self.logstd, min=-10, max=10)

        # Use pre-generated noise
        if epoch is not None and hasattr(self, 'all_noise') and epoch in self.all_noise:
            gaussian_noise = self.all_noise[epoch]
        else:
            # Fallback - should not happen during training
            gaussian_noise = torch.zeros(x_features.size(0), self.embedding_size, device=device)

        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z

    @staticmethod
    def decode(z):
        A_pred = torch.matmul(z, z.t())
        A_pred = torch.clamp(A_pred, min=-10, max=10)
        A_pred = torch.sigmoid(A_pred)
        A_pred = torch.clamp(A_pred, min=1e-7, max=1 - 1e-7)
        return A_pred

    def decodeZINB(self, z):
        m = self.Mean(z)
        d = self.Dispersion(z)
        p = self.Dropout(z)
        return m, d, p


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label

    def clusteringAcc(self):
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

        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c
        acc = metrics.accuracy_score(self.true_label, new_predict)

        return acc

    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = adjusted_rand_score(self.true_label, self.pred_label)
        acc = self.clusteringAcc()
        return acc, nmi, adjscore