import os
os.environ["OMP_NUM_THREADS"] = "15"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import scipy.sparse as sp
from model_old import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
import time
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from sklearn.metrics.cluster import adjusted_rand_score
from munkres import Munkres
from copulae.mixtures.gmc.gmc import GaussianMixtureCopula

device = get_device()
dataset = "baron4"
epochs_cluster = 800
lr_cluster = 0.01
embedding_size = 64
num_neurons = 256
activation = "Sigmoid"
seed = 8


# Code below is developed and adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE here. We thank for the authors to make it publicly available
device = torch.device('cpu')


class GraphConvSparse(nn.Module):
    def __init__(self, seed, input_dim, output_dim, adj, activation=torch.sigmoid, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)

        self.weight = random_uniform_init(input_dim, output_dim, seed)
        self.adj = adj
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
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


class Sigmoid(nn.Module):
    def __init__(self, seed, input_dim, output_dim, activation=torch.sigmoid, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

        self.weight = random_uniform_init(input_dim, output_dim, seed)
        self.activation = activation

    def forward(self, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        outputs = self.activation(x)
        return outputs


class GMCM_VGAE(nn.Module):
    """A Gaussian Mixture Copula Model based variational graph autoencoder

    Args:
        nn: Inputs for intialization
    """

    def __init__(self, **kwargs):
        super(GMCM_VGAE, self).__init__()
        self.adj = kwargs['adj']
        self.num_neurons = kwargs['num_neurons']
        self.num_features = kwargs['num_features']
        self.embedding_size = kwargs['embedding_size']
        self.nClusters = kwargs['nClusters']
        if kwargs['activation'] == "ReLU":
            self.activation = torch.relu
        if kwargs['activation'] == "Sigmoid":
            self.activation = torch.sigmoid
        if kwargs['activation'] == "Tanh":
            self.activation = torch.tanh
        self.seed = kwargs['seed']
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # VGAE training parameters
        self.base_gcn = GraphConvSparse(self.seed, self.num_features, self.num_neurons, self.activation)
        self.gcn_mean = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                        activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(self.seed, self.num_neurons, self.embedding_size, self.adj,
                                             activation=lambda x: x)

        # Clustering parameters initialization
        self.pi = nn.Parameter(torch.ones(self.nClusters) / self.nClusters, requires_grad=True)
        self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)
        self.log_sigma2_c = nn.Parameter(torch.randn(self.nClusters, self.embedding_size), requires_grad=True)

        # ZINB decoder
        self.Mean = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), MeanAct())
        self.Dispersion = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), DispAct())
        self.Dropout = nn.Sequential(nn.Linear(self.embedding_size, self.num_features), nn.Sigmoid())

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

    def Calculate_Loss(self, features, adj, x_, adj_label, y, weight_tensor, norm, z_mu, z_sigma2_log, emb, L=1):
        nClusters = self.nClusters
        features = features.to(device)
        x_ = x_.to(device)
        adj_label = adj_label.to(device)
        weight_tensor = weight_tensor.to(device)
        z_sigma2_log = z_sigma2_log.to(device)
        z_mu = z_mu.to(device)
        emb = emb.to(device)


        pi = self.pi.to(device)
        mu_c = self.mu_c.to(device)
        log_sigma2_c = self.log_sigma2_c

        # Reconstructed Loss
        det = 1e-2
        Loss_recons = det * norm * F.binary_cross_entropy(x_.view(-1), adj_label, weight=weight_tensor)
        Loss_recons = Loss_recons * features.size(0)

        # Cluster GMCM loss
        yita_c = torch.exp(
            torch.log(pi.unsqueeze(0)) + self.gmcm_gaussian_pdfs_log(emb, nClusters, mu_c, log_sigma2_c, pi)) + det
        yita_c = yita_c / (yita_c.sum(1).view(-1, 1))
        y_pred = self.predict_gmcm(emb, nClusters, mu_c, log_sigma2_c, pi)
        for c in range(self.nClusters):
            log_sigma2c = torch.diagonal(log_sigma2_c[c, :]).to(device)
        KL1 = 0.5 * torch.mean(torch.sum(yita_c * torch.sum(log_sigma2c.unsqueeze(0) +
                                                            torch.exp(
                                                                z_sigma2_log.unsqueeze(1) - log_sigma2c.unsqueeze(0)) +
                                                            (z_mu.unsqueeze(1) - mu_c.unsqueeze(0)).pow(2) / torch.exp(
            log_sigma2c.unsqueeze(0)), 2), 1))

        KL2 = torch.mean(torch.sum(yita_c * torch.log(pi.unsqueeze(0) / (yita_c)), 1)) + 0.5 * torch.mean(
            torch.sum(1 + z_sigma2_log, 1))
        Loss_gmcm = KL1 - KL2

        # ZINB loss
        extra = self.decodeZINB(emb)
        m, d, p = extra
        Loss_zinb = self.ZINB_loss(features.to_dense().squeeze(0), m, d, p)

        Loss_total = Loss_recons + Loss_gmcm + Loss_zinb
        return Loss_total, Loss_recons, Loss_gmcm, Loss_zinb

    def train(self, acc_list, adj_norm, features, adj_label, y, weight_tensor, norm, optimizer, epochs, lr,wd, momentum,save_path,
              dataset):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        adj_norm = adj_norm.to(device)
        features = features.to(device)
        adj_label = adj_label.to(device)
        weight_tensor = weight_tensor.to(device)

        if optimizer == "Adam":
            opti = Adam(self.parameters(), lr=lr, weight_decay=0.01)
        elif optimizer == "SGD":
            opti = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
        elif optimizer == "RMSProp":
            opti = RMSprop(self.parameters(), lr=lr, weight_decay=0.01)
        lr_s = StepLR(opti, step_size=10, gamma=0.9)

        import csv, os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Logging the resluts
        logfile = open(save_path + dataset + '/cluster/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['iter', 'ari', 'nmi', 'Loss_total'])
        logwriter.writeheader()

        epoch_bar = tqdm(range(epochs))

        print('Training......')

        count = 0
        currmax = 0
        finalist = []
        for epoch in epoch_bar:
            opti.zero_grad()
            # Encoding
            z_mu, z_sigma2_log, emb = self.encode(features, adj_norm)

            # Decoding
            x_ = self.decode(emb).to(device)

            # Use GMCM to model the clusters
            _, dim = emb.detach().cpu().numpy().shape
            z_mu = z_mu.to(device)
            z_sigma2_log = z_sigma2_log.to(device)
            emb = emb.to(device)

            gmcm = GaussianMixtureCopula(n_clusters=self.nClusters, ndim=dim)
            gmcm_fit = gmcm.fit(emb.detach().cpu().numpy(), method='kmeans', criteria='GMCM', eps=0.0001)
            pies = torch.from_numpy(gmcm_fit.params.prob)
            mus = torch.from_numpy(gmcm_fit.params.means)
            log_sigma2s = torch.from_numpy(np.log(gmcm_fit.params.covs))
            # emb = torch.from_numpy(emb.detach().cpu().numpy())
            emb = emb.to(device)
            n_clusters = gmcm_fit.clusters
            self.pi.data = pies
            self.mu_c.data = mus
            self.log_sigma2_c.data = log_sigma2s
            self.nClusters = n_clusters

            Loss_total, Loss_recons, Loss_gmcm, Loss_zinb = self.Calculate_Loss(features, adj_norm, x_,
                                                                                adj_label.to_dense().view(-1), y,
                                                                                weight_tensor, norm, z_mu, z_sigma2_log,
                                                                                emb)
            epoch_bar.write('Loss={:.4f}'.format(Loss_total.detach().cpu().numpy()))

            # Prediction and metrics
            nClusters = self.nClusters
            mu_c = self.mu_c
            log_sigma2_c = self.log_sigma2_c
            pi = self.pi

            y_pred = self.predict_gmcm(emb, nClusters, mu_c, log_sigma2_c, pi)

            cm = clustering_metrics(y, y_pred)
            acc, nmi, adjscore = cm.evaluationClusterModelFromLabel()
            acc_list.append(acc)

            logdict = dict(iter=epoch, ari=adjscore, nmi=nmi, Loss_total=Loss_total.detach().cpu().numpy())
            logwriter.writerow(logdict)
            logfile.flush()

            Loss_total.backward()
            opti.step()
            lr_s.step()
            count += 1
            if adjscore > currmax:
                finalist = [acc, adjscore, nmi, Loss_recons.detach().cpu().numpy(), Loss_gmcm.detach().cpu().numpy(),
                            Loss_zinb.detach().cpu().numpy(), Loss_total.detach().cpu().numpy(), epoch]
                currmax = adjscore
        return finalist, y_pred, y

    def gmcm_gaussian_pdfs_log(self, x, nClusters, mus, log_sigma2s, pies):
        G = []
        x = x.to(device)
        mus = mus.to(device)
        log_sigma2s = log_sigma2s.to(device)
        pies = pies.to(device)
        for c in range(nClusters):
            G.append(self.gmcm_gaussian_pdf_log(x, mus[c, :], torch.diagonal(log_sigma2s[c, :]), pies[c]).view(-1, 1))
            # covariance from GMCM is full convariance
        return torch.cat(G, 1).to(device)

    def gmcm_gaussian_pdf_log(self, x, mu, log_sigma2, pi):
        x = x.to(device)
        mu = mu.to(device)
        log_sigma2 = log_sigma2.to(device)
        pi = pi.to(device)
        c = -0.5 * torch.sum(np.log(np.pi * 2) + log_sigma2 + (x - mu).pow(2) / torch.exp(log_sigma2), 1)
        return c.to(device)

    def predict_gmcm(self, x, nClusters, mu_c, log_sigma2_c, pi_c):
        g = torch.log(pi_c.unsqueeze(0)) * self.gmcm_gaussian_pdfs_log(x, nClusters, mu_c, log_sigma2_c, pi_c)
        kappa_c = g / torch.sum(g, dim=0)
        kappa = kappa_c.detach().cpu().numpy()
        return np.argmax(kappa, axis=1)

    def encode(self, x_features, adj):
        hidden = self.base_gcn(x_features, adj)
        self.mean = self.gcn_mean(hidden, adj)
        self.logstd = self.gcn_logstddev(hidden, adj)

        gaussian_noise = torch.randn(x_features.size(0), self.embedding_size)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return self.mean, self.logstd, sampled_z

    @staticmethod
    def decode(z):
        A_pred = torch.sigmoid(torch.matmul(z, z.t()))
        return A_pred

    def decodeZINB(self, z):
        m = self.Mean(z)
        d = self.Dispersion(z)
        p = self.Dropout(z)
        # extra=(m,d,p)
        extra = (m, d, p)
        return extra


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




def main():
    save_path = "./results/"

    optimizer = "Adam"
    seed = 8
    wd = 0.01
    momentum = 0.9
    min_clamp_mean = 1e-5
    max_clamp_mean = 1e6
    min_clamp_dis = 1e-4
    max_clamp_dis = 1e4


    data, nClusters = load_data(dataset=dataset,
                                 data_path=f'./data/{dataset}',
                                 n_top_genes=1200,
                                 n_neighbors=5,
                                 n_pcs=50)

    A, X, labels = extract_graph_components(data)
    adj = csr_matrix(A) if not hasattr(A, 'eliminate_zeros') else A
    features = csr_matrix(X)

    # Data processing
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj_norm = preprocess_graph(adj)
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    def to_sparse_tensor(data):
        indices = torch.LongTensor(data[0].T).to(device)
        values = torch.FloatTensor(data[1]).to(device)
        shape = torch.Size(data[2])
        return torch.sparse.FloatTensor(indices, values, shape).to(device)

    adj_norm = to_sparse_tensor(adj_norm)
    adj_label = to_sparse_tensor(adj_label)
    features = to_sparse_tensor(features)

    weight_mask_orig = adj_label.to_dense().view(-1) == 1
    weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
    weight_tensor_orig[weight_mask_orig] = pos_weight_orig

    print("start")
    # Start the timer
    start = time.perf_counter()
    # Training


    network = GMCM_VGAE(adj=adj_norm, num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size,
                        nClusters=nClusters, activation=activation, seed=seed, min_clamp_dis=min_clamp_dis,
                        max_clamp_dis=max_clamp_dis, min_clamp_mean=min_clamp_mean, max_clamp_mean=max_clamp_mean)
    network.to(device)
    res, y_pred, y = network.train([], adj_norm, features, adj_label, labels, weight_tensor_orig, norm,
                                        optimizer=optimizer, epochs=epochs_cluster, lr=lr_cluster, wd=wd,
                                        momentum=momentum, save_path=save_path, dataset=dataset)
    end = time.perf_counter()

    print(f"Total time: {end - start:0.4f} seconds")
    print(f"Training results: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")


if __name__ == '__main__':

    main()