import os
os.environ["OMP_NUM_THREADS"] = "15"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import scipy.sparse as sp
from model import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph,get_device
import time

save_path = "./results/"
dataset = "baron3"

# Network hyperparameters
embedding_size = 64
num_neurons = 256
activation = "Sigmoid"
optimizer = "Adam"
seed = 8
wd = 0.01
momentum = 0.9
min_clamp_mean = 1e-5
max_clamp_mean = 1e6
min_clamp_dis = 1e-4
max_clamp_dis = 1e4

# Clustering hyperparameters
epochs_cluster = 350
lr_cluster = 0.01
n_top_genes = 2000
n_neighbors = 15
n_pcs = 5

device = get_device()
print(torch.cuda.is_available())

# ------------------------------------------------------------------ #
# Load data                                                            #
# ------------------------------------------------------------------ #
adj, features, labels, nClusters = load_data(
    dataset=dataset,
    data_path=f"./data/{dataset}",
    n_top_genes=n_top_genes,
    n_neighbors=n_neighbors,
    n_pcs=n_pcs,
)

# ------------------------------------------------------------------ #
# Helper: convert adj tensor → scipy sparse csr                       #
# ------------------------------------------------------------------ #
def tensor_to_scipy_sparse(t: torch.Tensor) -> sp.csr_matrix:
    return sp.csr_matrix(t.detach().cpu().numpy())

# ------------------------------------------------------------------ #
# Preprocess adjacency                                                 #
# ------------------------------------------------------------------ #
features_new = features.numpy()          # features is a tensor
num_nodes    = features.shape[0]
num_features = features.shape[1]

# Convert adj tensor → scipy sparse for all downstream scipy ops
adj_sp = tensor_to_scipy_sparse(adj)

# Remove self-loops
adj_sp = adj_sp - sp.dia_matrix(
    (adj_sp.diagonal()[np.newaxis, :], [0]), shape=adj_sp.shape
)
adj_sp.eliminate_zeros()

# Recompute loss weights (on scipy sparse)
pos_weight_orig = float(adj_sp.shape[0] * adj_sp.shape[0] - adj_sp.nnz) / adj_sp.nnz
norm = adj_sp.shape[0] * adj_sp.shape[0] / float((adj_sp.shape[0] * adj_sp.shape[0] - adj_sp.nnz) * 2)

# Normalised adjacency for GCN
adj_norm = preprocess_graph(adj_sp)

# Labels and features as sparse tuples
adj_label = adj_sp + sp.eye(adj_sp.shape[0])
adj_label  = sparse_to_tuple(adj_label)
features   = sparse_to_tuple(sp.csr_matrix(features_new))   # numpy → sparse → tuple
num_features = features[2][1]

# ------------------------------------------------------------------ #
# Convert to sparse tensors                                            #
# ------------------------------------------------------------------ #
def to_sparse_tensor(data):
    indices = torch.LongTensor(data[0].T).to(device)
    values  = torch.FloatTensor(data[1]).to(device)
    shape   = torch.Size(data[2])
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

adj_norm  = to_sparse_tensor(adj_norm)
adj_label = to_sparse_tensor(adj_label)
features  = to_sparse_tensor(features)

weight_mask_orig   = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig

# ------------------------------------------------------------------ #
# Train                                                                #
# ------------------------------------------------------------------ #
print("start")
start = time.perf_counter()

network = GMCM_VGAE(
    adj=adj_norm, num_neurons=num_neurons, num_features=num_features,
    embedding_size=embedding_size, nClusters=nClusters, activation=activation,
    seed=seed, min_clamp_dis=min_clamp_dis, max_clamp_dis=max_clamp_dis,
    min_clamp_mean=min_clamp_mean, max_clamp_mean=max_clamp_mean
)
network.to(device)

res, y_pred, y = network.train(
    [], adj_norm, features, adj_label, labels, weight_tensor_orig,
    norm, optimizer=optimizer, epochs=epochs_cluster, lr=lr_cluster,
    wd=wd, momentum=momentum, save_path=save_path, dataset=dataset,
    features_new=features_new
)

end = time.perf_counter()
print(f"Total time: {end - start:0.4f} seconds")
print(f"Training results for {dataset}: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")


















# import os
# os.environ["OMP_NUM_THREADS"] = "15"
# import warnings
# warnings.filterwarnings("ignore")
# import numpy as np
# import torch
# import scipy.sparse as sp
# from model import GMCM_VGAE
# from preprocessing import load_data, sparse_to_tuple, preprocess_graph
# import time
# from random import randint
# import math
#
# # Code below is adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE. We thank for the authors to make it publicly available
# save_path = "./results/"
# dataset = "Klein"
#
#
# # Network hyperparameter
# embedding_size = 64
# num_neurons = 256
# activation="Sigmoid"
# optimizer="Adam"
# seed=8
# wd=0.01
# momentum=0.9
# min_clamp_mean=1e-5
# max_clamp_mean=1e6
# min_clamp_dis=1e-4
# max_clamp_dis=1e4
# # Clustering hyperparameters
# epochs_cluster = 350
# lr_cluster = 0.01
# n_top_genes=2000
# n_neighbors=15
# n_pcs=5
# # Configure the device to cuda
# # torch.set_default_tensor_type('torch.cuda.FloatTensor')
# device = torch.device("cpu")
# print(torch.cuda.is_available())
#
#
# # adj, features, labels,nClusters = load_data('baron3', './data/baron3', True)
# adj, features, labels,nClusters = load_data(
#         dataset=dataset,
#         data_path=f"./data/{dataset}",
#         n_top_genes=n_top_genes,
#         n_neighbors=n_neighbors,
#         n_pcs=n_pcs,
#     )
#
#
# features_new = features.numpy()
# num_nodes = features.shape[0]
# num_features = features.shape[1]
# # Data processing
# adj_np = adj.detach().cpu().numpy()
# adj_np = adj_np - sp.dia_matrix((adj_np.diagonal()[np.newaxis, :], [0]), shape=adj_np.shape).toarray()
# adj = torch.tensor(adj_np, dtype=torch.float)
#
# # adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
#
# adj.eliminate_zeros()
# adj_norm = preprocess_graph(adj)
# features = sparse_to_tuple(features.tocoo())
# num_features = features[2][1]
# pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
# norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
# adj_label = adj + sp.eye(adj.shape[0])
# adj_label = sparse_to_tuple(adj_label)
#
#
#
# def to_sparse_tensor(data):
#     indices = torch.LongTensor(data[0].T).to(device)
#     values = torch.FloatTensor(data[1]).to(device)
#     shape = torch.Size(data[2])
#     return torch.sparse.FloatTensor(indices, values, shape).to(device)
#
# adj_norm = to_sparse_tensor(adj_norm)
# adj_label = to_sparse_tensor(adj_label)
# features = to_sparse_tensor(features)
#
# weight_mask_orig = adj_label.to_dense().view(-1) == 1
# weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
# weight_tensor_orig[weight_mask_orig] = pos_weight_orig
#
#
# print("start")
# # Start the timer
# start = time.perf_counter()
# # Training
# acc_array = []
#
# ress = []
#
#
# network = GMCM_VGAE(adj = adj_norm , num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation=activation, seed=seed,min_clamp_dis=min_clamp_dis,max_clamp_dis=max_clamp_dis,min_clamp_mean=min_clamp_mean,max_clamp_mean=max_clamp_mean)
# network.to(device)
# res, y_pred, y = network.train([], adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer=optimizer, epochs=epochs_cluster, lr=lr_cluster,wd=wd,momentum=momentum, save_path=save_path, dataset=dataset, features_new=features_new)
# end = time.perf_counter()
#
# print(f"Total time: {end - start:0.4f} seconds")
# print(f"Training results for {dataset}: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")
