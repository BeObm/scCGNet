import os
os.environ["OMP_NUM_THREADS"] = "15"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import scipy.sparse as sp
from model import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
import time
from random import randint
import math

# Code below is adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE. We thank for the authors to make it publicly available
save_path = "./results/"
dataset = "baron3"
nClusters = 14


# Network hyperparameter
embedding_size = 64
num_neurons = 256
activation="Sigmoid"
optimizer="Adam"
seed=8
wd=0.01
momentum=0.9
min_clamp_mean=1e-5
max_clamp_mean=1e6
min_clamp_dis=1e-4
max_clamp_dis=1e4
# Clustering hyperparameters
epochs_cluster = 350
lr_cluster = 0.01

# Configure the device to cuda
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
device = torch.device("cpu")
print(torch.cuda.is_available())


adj, features, labels = load_data('baron3', './data/baron3', True)

features_new = features.toarray()
num_nodes = features.shape[0]
num_features = features.shape[1]
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
acc_array = []

ress = []


network = GMCM_VGAE(adj = adj_norm , num_neurons=num_neurons, num_features=num_features, embedding_size=embedding_size, nClusters=nClusters, activation=activation, seed=seed,min_clamp_dis=min_clamp_dis,max_clamp_dis=max_clamp_dis,min_clamp_mean=min_clamp_mean,max_clamp_mean=max_clamp_mean)
network.to(device)
res, y_pred, y = network.train([], adj_norm, features, adj_label, labels, weight_tensor_orig, norm, optimizer=optimizer, epochs=epochs_cluster, lr=lr_cluster,wd=wd,momentum=momentum, save_path=save_path, dataset=dataset, features_new=features_new)
end = time.perf_counter()

print(f"Total time: {end - start:0.4f} seconds")
print(f"Training results: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")