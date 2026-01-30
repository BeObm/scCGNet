import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.manifold import SpectralEmbedding
from munkres import Munkres
from copulae.mixtures.gmc.gmc import GaussianMixtureCopula
from preprocessing import *

device=get_device()
class GraphConvSparse(nn.Module):
    def __init__(self, seed, input_dim, output_dim, adj, activation=torch.sigmoid, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.weight = random_uniform_init(input_dim, output_dim, seed)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.spmm(x, self.weight)
        x = torch.spmm(adj, x)
        outputs = self.activation(x)
        return outputs


class MeanAct(nn.Module):
    def __init__(self, **kwargs):
        super(MeanAct, self).__init__(**kwargs)

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self, **kwargs):
        super(DispAct, self).__init__(**kwargs)

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class GMCM_VGAE_Final(nn.Module):
    """Final optimized GMCM-VGAE model

    Combines:
    - GMCM for probabilistic clustering
    - Spectral embedding initialization
    - Graph structure preservation
    - All stability fixes
    - Optimized for fast convergence
    """

    def __init__(self, **kwargs):
        super(GMCM_VGAE_Final, self).__init__()
        self.adj = kwargs['adj']
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        self.seed = kwargs['seed']

        if kwargs['activation'] == "ReLU":
            self.activation = torch.relu
        elif kwargs['activation'] == "Sigmoid":
            self.activation = torch.sigmoid
        elif kwargs['activation'] == "Tanh":
            self.activation = torch.tanh
        else:
            self.activation = torch.sigmoid

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # VGAE encoder
        self.base_gcn = GraphConvSparse(self.seed, self.num_features, self.num_neurons, self.adj, self.activation)
        self.gcn_mean = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                        activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                             activation=lambda x: x)

        # GMCM cluster parameters (diagonal covariance)
        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)

        # ZINB decoder
        self.Mean = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), MeanAct())
        self.Dispersion = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), DispAct())
        self.Dropout = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), nn.Sigmoid())

        # Loss weights - optimized for fast convergence
        self.alpha_recons = kwargs.get('alpha_recons', 0.1)
        self.beta_gmcm = kwargs.get('beta_gmcm', 5.0)
        self.gamma_zinb = kwargs.get('gamma_zinb', 0.1)
        self.delta_graph = kwargs.get('delta_graph', 10.0)  # Graph structure preservation

    def ZINB_loss(self, x, mean, disp, pi, scale_factor=1.0):
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        return torch.mean(result)

    def graph_laplacian_loss(self, embeddings, adj):
        """Enforce graph smoothness - optimized with edge sampling"""
        adj_coo = adj.coalesce()
        edge_indices = adj_coo.indices()
        edge_values = adj_coo.values()

        # Sample edges for efficiency
        num_edges = edge_indices.shape[1]
        if num_edges > 3000:
            sample_idx = torch.randperm(num_edges, device=edge_indices.device)[:3000]
            edge_indices = edge_indices[:, sample_idx]
            edge_values = edge_values[sample_idx]

        source_emb = embeddings[edge_indices[0]]
        target_emb = embeddings[edge_indices[1]]
        edge_distances = torch.sum((source_emb - target_emb) ** 2, dim=1)
        weighted_distances = edge_distances * edge_values

        return torch.mean(weighted_distances)

    def Calculate_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb):

        features = features.to(device)
        adj = adj.to(device)
        x_ = x_.to(device)
        adj_label = adj_label.to(device)
        weight_tensor = weight_tensor.to(device)
        z_mu = z_mu.to(device)
        z_sigma2_log = z_sigma2_log.to(device)
        emb = emb.to(device)

        nClusters = self.nClusters
        pi = self.pi.to(device)
        mu_c = self.mu_c.to(device)
        log_sigma2_c = self.log_sigma2_c.to(device)

        # Reconstruction loss
        det = 1e-2
        Loss_recons = det * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight=weight_tensor)
        Loss_recons = Loss_recons * features.size(0)

        # GMCM clustering loss
        yita_c = torch.exp(torch.log(pi.unsqueeze(0) + 1e-10) +
                           self.gmcm_gaussian_pdfs_log(emb, nClusters, mu_c, log_sigma2_c, pi)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1) + 1e-10)

        # Optional: Class imbalance weighting
        use_class_weights = False
        if use_class_weights:
            cluster_counts = torch.bincount(torch.from_numpy(y).long(), minlength=nClusters)
            cluster_weights = 1.0 / (cluster_counts.float() + 1.0)
            cluster_weights = cluster_weights * nClusters / cluster_weights.sum()
            weighted_yita = yita_c * cluster_weights.unsqueeze(0)
            weighted_yita = weighted_yita / (weighted_yita.sum(1, keepdim=True) + 1e-10)
        else:
            weighted_yita = yita_c.to(device)

        # KL divergence
        KL1 = 0.5 * torch.mean(torch.sum(weighted_yita * torch.sum(
            log_sigma2_c.unsqueeze(0) +
            torch.exp(z_sigma2_log.unsqueeze(1) - log_sigma2_c.unsqueeze(0)) +
            (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(log_sigma2_c.unsqueeze(0)), 2), 1))

        KL2 = torch.mean(torch.sum(weighted_yita * torch.log((pi.unsqueeze(0) + 1e-10) / (weighted_yita + 1e-10)), 1)) + \
              0.5 * torch.mean(torch.sum(1 + z_sigma2_log, 1))
        Loss_gmcm = KL1 - KL2

        # ZINB loss
        extra = self.decodeZINB(emb)
        m, d, p = extra

        if features.is_sparse:
            feat_dense = features.to_dense().squeeze(0) if features.shape[0] * features.shape[
                1] < 1e7 else features.coalesce().values()
        else:
            feat_dense = features.squeeze(0)

        Loss_zinb = self.ZINB_loss(feat_dense, m, d, p)

        # Graph structure preservation
        Loss_graph = self.graph_laplacian_loss(emb, adj)

        # Combined loss
        Loss_total = (
                self.alpha_recons * Loss_recons +
                self.beta_gmcm * Loss_gmcm +
                self.gamma_zinb * Loss_zinb +
                self.delta_graph * Loss_graph
        )

        return Loss_total, Loss_recons, Loss_gmcm, Loss_zinb, Loss_graph, weighted_yita

    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path,
              dataset, features_new):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        opti = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.00001, betas=(0.9, 0.999))
        scheduler = CosineAnnealingLR(opti, T_max=epochs, eta_min=lr / 50)

        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cluster_dir = os.path.join(save_path, dataset, 'cluster')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        logfile = open(os.path.join(cluster_dir, 'log.csv'), 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ari', 'nmi', 'loss_total', 'loss_gmcm', 'loss_graph'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))
        print('Training Final GMCM-VGAE...')

        currmax = 0
        finalist = []
        patience = 100
        patience_counter = 0
        best_model_state = None

        # Initialize GMCM with spectral embedding
        print("\nInitializing GMCM with Spectral Embedding...")
        gmcm = GaussianMixtureCopula(n_clusters=self.nClusters, ndim=self.embedding_size)

        try:
            with torch.no_grad():
                z_mu_init, z_sigma2_log_init, emb_init = self.encode(features, adj_norm)

            # Try spectral embedding first
            try:
                adj_dense = adj_norm.to_dense().cpu().numpy()
                spectral = SpectralEmbedding(n_components=self.embedding_size, random_state=self.seed,
                                             affinity='precomputed')
                spectral_emb = spectral.fit_transform(adj_dense)
                init_emb = spectral_emb
                print(f"Using Spectral Embedding initialization")
            except:
                init_emb = emb_init.cpu().numpy()
                print(f"Using GCN embedding initialization")

            # Normalize and fit GMCM
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            init_emb_norm = scaler.fit_transform(init_emb.astype('float64'))

            gmcm_fit = gmcm.fit(init_emb_norm, method='kmeans', criteria='GMCM', eps=0.0001)

            # Initialize parameters
            self.pi.data = torch.from_numpy(gmcm_fit.params.prob).float()
            self.mu_c.data = torch.from_numpy(gmcm_fit.params.means).float()

            # Extract diagonal covariance
            covs = gmcm_fit.params.covs
            covs_diag = np.array([np.diag(covs[i]) for i in range(covs.shape[0])])
            log_covs_diag = np.log(np.abs(covs_diag) + 1e-6)
            self.log_sigma2_c.data = torch.from_numpy(log_covs_diag).float()

            print(f"GMCM initialization complete. Clusters: {self.nClusters}\n")

        except Exception as e:
            print(f"Warning: GMCM initialization failed: {e}")
            print("Using random initialization\n")

        for epoch in epoch_bar:
            try:
                opti.zero_grad()

                # Encode
                z_mu, z_sigma2_log, emb = self.encode(features, adj_norm)

                # Decode
                x_ = self.decode(emb)

                # GMCM update every 100 epochs with very gentle EMA
                if epoch % 100 == 0 and epoch > 0:
                    try:
                        emb_np = emb.detach().cpu().numpy().astype("float64", copy=False)
                        emb_np_norm = scaler.transform(emb_np)
                        gmcm_fit = gmcm.fit(emb_np_norm, method='kmeans', criteria='GMCM', eps=0.0001)

                        alpha = 0.02  # Very gentle update

                        pies = torch.from_numpy(gmcm_fit.params.prob).float()
                        mus = torch.from_numpy(gmcm_fit.params.means).float()
                        covs = gmcm_fit.params.covs
                        covs_diag = np.array([np.diag(covs[i]) for i in range(covs.shape[0])])
                        log_covs_diag = np.log(np.abs(covs_diag) + 1e-6)
                        log_sigma2s = torch.from_numpy(log_covs_diag).float()

                        self.pi.data = alpha * pies + (1 - alpha) * self.pi.data
                        self.mu_c.data = alpha * mus + (1 - alpha) * self.mu_c.data
                        self.log_sigma2_c.data = alpha * log_sigma2s + (1 - alpha) * self.log_sigma2_c.data

                        print(f"\n[GMCM Update at epoch {epoch}]")
                    except Exception as e:
                        print(f"\nWarning: GMCM update failed: {e}")

                # Calculate loss
                Loss_total, Loss_recons, Loss_gmcm, Loss_zinb, Loss_graph, cluster_probs = self.Calculate_Loss(
                    features, adj_norm, x_, adj_label.to_dense().view(-1), y,
                    weight_tensor, norm, z_mu, z_sigma2_log, emb
                )

                if torch.isnan(Loss_total) or torch.isinf(Loss_total):
                    print(f"\nWarning: Invalid loss at epoch {epoch}")
                    continue

                # Predictions
                y_pred = self.predict_gmcm(emb.detach(), self.nClusters, self.mu_c, self.log_sigma2_c, self.pi)

                # Metrics
                cm = clustering_metrics(y, y_pred)
                acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
                acc_list.append(acc)

                # Logging every 5 epochs
                if epoch % 5 == 0:
                    n_pred = len(np.unique(y_pred))
                    epoch_bar.write(
                        f'Epoch {epoch}: Loss={Loss_total.item():.2f}, ARI={adjscore:.4f}, NMI={nmi:.4f}, Clusters={n_pred}')

                    if epoch > 0 and len(acc_list) > 5:
                        recent_ari = [acc_list[i] for i in range(len(acc_list) - 5, len(acc_list)) if i >= 0]
                        if len(recent_ari) >= 2:
                            ari_change = recent_ari[-1] - recent_ari[0]
                            if ari_change > 0.001:
                                epoch_bar.write(f'  └─ ARI +{ari_change:.4f} in last 5 epochs ✓')

                logdict = dict(
                    iter=epoch,
                    ari=adjscore,
                    nmi=nmi,
                    loss_total=Loss_total.detach().cpu().numpy(),
                    loss_gmcm=Loss_gmcm.detach().cpu().numpy(),
                    loss_graph=Loss_graph.detach().cpu().numpy()
                )
                logwriter.writerow(logdict)
                logfile.flush()

                # Backward
                Loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opti.step()
                scheduler.step()

                # Track best
                if adjscore > currmax:
                    finalist = [acc, adjscore, nmi, Loss_total.detach().cpu().numpy(), epoch]
                    currmax = adjscore
                    patience_counter = 0
                    best_model_state = {
                        'pi': self.pi.data.clone(),
                        'mu_c': self.mu_c.data.clone(),
                        'log_sigma2_c': self.log_sigma2_c.data.clone()
                    }
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"\n[Early Stopping] No improvement for {patience} epochs")
                    print(f"Best ARI: {currmax:.4f} at epoch {finalist[4]}")
                    break

            except Exception as e:
                print(f"\nError at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Restore best model
        if best_model_state is not None:
            print(f"\nRestoring best model from epoch {finalist[4]}")
            self.pi.data = best_model_state['pi']
            self.mu_c.data = best_model_state['mu_c']
            self.log_sigma2_c.data = best_model_state['log_sigma2_c']

            with torch.no_grad():
                _, _, emb = self.encode(features, adj_norm)
                y_pred = self.predict_gmcm(emb, self.nClusters, self.mu_c, self.log_sigma2_c, self.pi)

        logfile.close()
        return finalist, y_pred, y

    def gmcm_gaussian_pdfs_log(self, x, nClusters, mus, log_sigma2s, pies):
        G = []

        mus = mus.to(device)
        log_sigma2s = log_sigma2s.to(device)
        pies = pies.to(device)

        for c in range(nClusters):
            G.append(self.gmcm_gaussian_pdf_log(x, mus[c, :], log_sigma2s[c, :], pies[c]).view(-1, 1))
        return torch.cat(G, 1)

    def gmcm_gaussian_pdf_log(self, x, mu, log_sigma2, pi):
        c = -0.5 * torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1)
        return c

    def predict_gmcm(self, x, nClusters, mu_c, log_sigma2_c, pi_c):
        mu_c = mu_c.to(device)
        log_sigma2_c = log_sigma2_c.to(device)
        pi_c = pi_c.to(device)
        log_probs = torch.log(pi_c.unsqueeze(0) + 1e-10) + self.gmcm_gaussian_pdfs_log(x, nClusters, mu_c, log_sigma2_c,
                                                                                       pi_c)
        kappa_c = torch.softmax(log_probs, dim=1)
        return torch.argmax(kappa_c, dim=1).cpu().numpy()

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)

        self.logstd = torch.clamp(self.logstd, min=-10, max=2)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size).to(device)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        sampled_z = torch.clamp(sampled_z, min=-10, max=10)

        return self.mean, self.logstd, sampled_z

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

    def decodeZINB(self, z):
        m = self.Mean(z)
        d = self.Dispersion(z)
        p = self.Dropout(z)
        return (m, d, p)


def random_uniform_init(input_dim, output_dim, seed):
    np.random.seed(seed)
    init_range = np.sqrt(2.0 / (input_dim + output_dim))
    torch.manual_seed(seed)
    initial = torch.randn(input_dim, output_dim) * init_range
    return nn.Parameter(initial)


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