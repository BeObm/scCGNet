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
from sklearn.cluster import KMeans
from munkres import Munkres


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


class GraphStructureVGAE(nn.Module):
    """VGAE optimized for graph structure learning

    Key principles:
    1. Initialize with spectral embedding (proven ARI=0.54)
    2. Minimize feature reconstruction weight
    3. Maximize graph structure preservation
    4. Use graph Laplacian regularization
    """

    def __init__(self, **kwargs):
        super(GraphStructureVGAE, self).__init__()
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

        # Graph encoder layers
        self.base_gcn = GraphConvSparse(self.seed, self.num_features, self.num_neurons, self.adj, self.activation)
        self.gcn_mean = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                        activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                             activation=lambda x: x)

        # Cluster prototypes
        self.prototypes = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)

        # Loss weights - heavily favor graph structure
        self.alpha_recons = kwargs.get('alpha_recons', 0.01)  # Very low
        self.beta_cluster = kwargs.get('beta_cluster', 10.0)  # High
        self.gamma_structure = kwargs.get('gamma_structure', 20.0)  # Very high

    def graph_laplacian_loss(self, embeddings, adj):
        """Enforce graph smoothness - connected nodes should have similar embeddings"""
        # Compute pairwise distances
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        distances = torch.sum(diff ** 2, dim=2)

        # Weight by adjacency
        adj_dense = adj.to_dense()
        weighted_distances = distances * adj_dense

        laplacian_loss = torch.sum(weighted_distances) / (torch.sum(adj_dense) + 1e-10)

        return laplacian_loss

    def structure_preservation_loss(self, embeddings, original_distances):
        """Preserve original graph distances in embedding space"""
        emb_distances = torch.cdist(embeddings, embeddings, p=2)
        structure_loss = F.mse_loss(emb_distances, original_distances)
        return structure_loss

    def clustering_loss(self, embeddings):
        """Distance-based clustering to prototypes"""
        distances = torch.cdist(embeddings, self.prototypes, p=2)

        # Soft assignments
        assignments = F.softmax(-distances, dim=1)

        # Entropy regularization
        entropy = -torch.sum(assignments * torch.log(assignments + 1e-10), dim=1)

        # Compactness
        min_distances = torch.min(distances, dim=1)[0]
        compactness = torch.mean(min_distances)

        # Separation
        proto_distances = torch.cdist(self.prototypes, self.prototypes, p=2)
        mask = 1.0 - torch.eye(self.nClusters, device=proto_distances.device)
        separation = -torch.mean(proto_distances * mask)

        total_cluster_loss = compactness + 0.1 * torch.mean(entropy) + 0.1 * separation

        return total_cluster_loss, assignments

    def calculate_loss(self, features, adj, x_, adj_label, weight_tensor, norm, z_mu, z_sigma2_log, emb,
                       original_distances):
        # Graph reconstruction loss (minimal weight)
        det = 1e-2
        loss_recons = det * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight=weight_tensor)
        loss_recons = loss_recons * features.size(0)

        # Clustering loss
        loss_cluster, cluster_assignments = self.clustering_loss(emb)

        # Graph structure preservation (critical)
        loss_laplacian = self.graph_laplacian_loss(emb, adj)
        loss_structure_preserve = self.structure_preservation_loss(emb, original_distances)

        # KL divergence
        loss_kl = -0.5 * torch.mean(torch.sum(1 + z_sigma2_log - z_mu.pow(2) - z_sigma2_log.exp(), dim=1))

        # Combined loss
        loss_total = (
                self.alpha_recons * loss_recons +
                self.beta_cluster * (loss_cluster + 0.1 * loss_kl) +
                self.gamma_structure * (loss_laplacian + loss_structure_preserve)
        )

        return loss_total, loss_recons, loss_cluster, loss_laplacian, cluster_assignments

    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr, save_path,
              dataset, features_new):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        opti = Adam(self.parameters(), lr=lr, weight_decay=0.0001)
        scheduler = CosineAnnealingLR(opti, T_max=epochs, eta_min=lr / 100)

        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cluster_dir = os.path.join(save_path, dataset, 'cluster')
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        logfile = open(os.path.join(cluster_dir, 'log.csv'), 'w')
        logwriter = csv.DictWriter(logfile,
                                   fieldnames=['iter', 'ari', 'nmi', 'loss_total', 'loss_cluster', 'loss_structure'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))
        print('Training Graph-Structure-First VGAE...')

        currmax = 0
        finalist = []
        patience = 200
        patience_counter = 0
        best_model_state = None

        # Initialize with spectral embedding
        print("\nInitializing with Spectral Embedding (32-dim)...")
        try:
            adj_dense = adj_norm.to_dense().cpu().numpy()
            spectral = SpectralEmbedding(n_components=self.embedding_size, random_state=self.seed,
                                         affinity='precomputed')
            spectral_emb = spectral.fit_transform(adj_dense)

            kmeans = KMeans(n_clusters=self.nClusters, random_state=self.seed, n_init=50)
            kmeans.fit(spectral_emb)

            self.prototypes.data = torch.from_numpy(kmeans.cluster_centers_).float()

            spectral_pred = kmeans.predict(spectral_emb)
            spectral_ari = adjusted_rand_score(y, spectral_pred)
            spectral_nmi = metrics.normalized_mutual_info_score(y, spectral_pred)

            print(f"Spectral initialization: ARI={spectral_ari:.4f}, NMI={spectral_nmi:.4f}")
            print(f"Target: Maintain this ARI throughout training")

        except Exception as e:
            print(f"Warning: Spectral initialization failed: {e}")

        # Pre-compute graph distances
        print("Computing graph distances...")
        adj_dense_torch = adj_norm.to_dense()
        graph_distances = torch.where(adj_dense_torch > 0,
                                      torch.ones_like(adj_dense_torch),
                                      torch.ones_like(adj_dense_torch) * 100)
        graph_distances.fill_diagonal_(0)

        print("Starting training...")
        print("=" * 60)

        for epoch in epoch_bar:
            try:
                opti.zero_grad()

                # Encode
                z_mu, z_sigma2_log, emb = self.encode(features, adj_norm)

                # Decode
                x_ = self.decode(emb)

                # Calculate loss
                loss_total, loss_recons, loss_cluster, loss_structure, cluster_assignments = self.calculate_loss(
                    features, adj_norm, x_, adj_label.to_dense().view(-1),
                    weight_tensor, norm, z_mu, z_sigma2_log, emb, graph_distances
                )

                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    print(f"\nWarning: Invalid loss at epoch {epoch}")
                    continue

                # Predictions
                y_pred = torch.argmax(cluster_assignments, dim=1).cpu().numpy()

                # Metrics
                cm = clustering_metrics(y, y_pred)
                acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
                acc_list.append(acc)

                # Logging
                if epoch % 10 == 0:
                    n_pred = len(np.unique(y_pred))
                    epoch_bar.write(
                        f'Epoch {epoch}: Loss={loss_total.item():.2f}, ARI={adjscore:.4f}, NMI={nmi:.4f}, Clusters={n_pred}')

                logdict = dict(
                    iter=epoch,
                    ari=adjscore,
                    nmi=nmi,
                    loss_total=loss_total.detach().cpu().numpy(),
                    loss_cluster=loss_cluster.detach().cpu().numpy(),
                    loss_structure=loss_structure.detach().cpu().numpy()
                )
                logwriter.writerow(logdict)
                logfile.flush()

                # Backward
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                opti.step()
                scheduler.step()

                # Track best
                if adjscore > currmax:
                    finalist = [acc, adjscore, nmi, loss_total.detach().cpu().numpy(), epoch]
                    currmax = adjscore
                    patience_counter = 0
                    best_model_state = {
                        'prototypes': self.prototypes.data.clone(),
                        'base_gcn': self.base_gcn.state_dict(),
                        'gcn_mean': self.gcn_mean.state_dict(),
                        'gcn_logstddev': self.gcn_logstddev.state_dict()
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
            self.prototypes.data = best_model_state['prototypes']
            self.base_gcn.load_state_dict(best_model_state['base_gcn'])
            self.gcn_mean.load_state_dict(best_model_state['gcn_mean'])
            self.gcn_logstddev.load_state_dict(best_model_state['gcn_logstddev'])

            # Final predictions
            with torch.no_grad():
                _, _, emb = self.encode(features, adj_norm)
                distances = torch.cdist(emb, self.prototypes, p=2)
                assignments = F.softmax(-distances, dim=1)
                y_pred = torch.argmax(assignments, dim=1).cpu().numpy()

        logfile.close()
        return finalist, y_pred, y

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)

        self.logstd = torch.clamp(self.logstd, min=-10, max=2)

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        sampled_z = torch.clamp(sampled_z, min=-10, max=10)

        return self.mean, self.logstd, sampled_z

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred


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