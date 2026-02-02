import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
import scipy.sparse as sp
from model import GMCM_VGAE_Final
from preprocessing import load_data, sparse_to_tuple, preprocess_graph, get_device
import time

print("="*60)
print("Final Optimized GMCM-VGAE")
print("Combines: GMCM + Spectral Init + Graph Structure + All Fixes")
print("="*60)

# Dataset
dataset = "baron4"
nClusters = 14

# Load data
print("\n[1/6] Loading data...")
adj, features, labels = load_data('baron4', './data/baron4', True)

features_new = features.toarray()
num_nodes = features.shape[0]
num_features = features.shape[1]

print(f"  Nodes: {num_nodes}")
print(f"  Features: {num_features}")
print(f"  Clusters: {nClusters}")
print(f"  Label distribution: {np.bincount(labels)}")

# Network hyperparameters
embedding_size = 32  # Match spectral-32 (ARI=0.54)
num_neurons = 128
save_path = "./results/"

# Training hyperparameters - optimized for fast convergence
epochs_cluster = 500
lr_cluster = 0.005

# Device
device = get_device()
print(f"\n[2/6] Device: {device}")

# Data processing
print("\n[3/6] Preprocessing...")
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()
adj_norm0 = preprocess_graph(adj)
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
pos_weight_orig = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
adj_label = adj + sp.eye(adj.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Create tensors
print("\n[4/6] Creating tensors...")
adj_norm = torch.sparse_coo_tensor(
    indices=torch.LongTensor(adj_norm0[0].T).to(device),
    values=torch.FloatTensor(adj_norm0[1]).to(device),
    size=torch.Size(adj_norm0[2]),
    device=device
).coalesce()

adj_label = torch.sparse_coo_tensor(
    indices=torch.LongTensor(adj_label[0].T).to(device),
    values=torch.FloatTensor(adj_label[1]).to(device),
    size=torch.Size(adj_label[2]),
    device=device
).coalesce()

features = torch.sparse_coo_tensor(
    indices=torch.LongTensor(features[0].T).to(device),
    values=torch.FloatTensor(features[1]).to(device),
    size=torch.Size(features[2]),
    device=device
).coalesce()

weight_mask_orig = adj_label.to_dense().view(-1) == 1
weight_tensor_orig = torch.ones(weight_mask_orig.size(0))
weight_tensor_orig[weight_mask_orig] = pos_weight_orig

# Create model
print("\n[5/6] Creating Final GMCM-VGAE model...")
seed = 42

epochs_cluster_0 = [500,750]
lr_cluster_0 = [0.001,0.01] # Increased from 0.001 for faster learning
alpha_recons_0=[ 0.1, 0.8]
beta_cluster_0=[0.1, 0.001]
gamma_structure_0=[0.1, 0.001]
delta_graph_0=[0.1, 0.001]
embedding_size_0 = [32]
num_neurons_0 = [128,512]    # More capacity for graph learning

result_dict=defaultdict(list)
start_time = time.perf_counter()

for epochs_cluster in tqdm(epochs_cluster_0):
    for lr_cluster in lr_cluster_0:
        for alpha_recons in alpha_recons_0:
            for beta_cluster in beta_cluster_0:
                for gamma_structure in gamma_structure_0:
                    for embedding_size in embedding_size_0:
                        for delta_graph in delta_graph_0:
                           for num_neurons in num_neurons_0:
                              print(
                                  f"\n Starting training... with "
                                  f"epoch_cluster({epochs_cluster}) | "
                                  f"lr_cluster({lr_cluster}) | "
                                  f"alpha_recons({alpha_recons}) | "
                                  f"beta_cluster({beta_cluster}) | "
                                  f"gamma_gmcm({gamma_structure}) | "
                                  f"embedding_size({embedding_size}) | "
                                  f"delta_graph({delta_graph}) | "
                                  f"num_neurons({num_neurons})"
                              )
                              network = GMCM_VGAE_Final(
                                    adj=adj_norm,
                                    num_neurons=num_neurons,
                                    num_features=num_features,
                                    embedding_size=embedding_size,
                                    nClusters=nClusters,
                                    activation="Sigmoid",
                                    seed=seed,
                                    # Optimized loss weights
                                    alpha_recons=alpha_recons,     # Reconstruction
                                    beta_gmcm=beta_cluster,        # GMCM clustering
                                    gamma_zinb=gamma_structure,       # ZINB
                                    delta_graph=delta_graph      # Graph structure preservation
                                )
                              network.to(device)

                              start_time = time.perf_counter()

                              res, y_pred, y = network.train(
                                    acc_list=[],
                                    adj_norm=adj_norm.to(device),
                                    features=features.to(device),
                                    adj_label=adj_label.to(device),
                                    y=labels,
                                    weight_tensor=weight_tensor_orig.to(device),
                                    norm=norm,
                                    optimizer="Adam",
                                    epochs=epochs_cluster,
                                    lr=lr_cluster,
                                    save_path=save_path,
                                    dataset=dataset,
                                    features_new=features_new
                                )

                              result_dict["epoch_cluster"].append(epochs_cluster)
                              result_dict["lr_cluster"].append(lr_cluster)
                              result_dict["alpha_recons_cluster"].append(alpha_recons)
                              result_dict["beta_cluster"].append(beta_cluster)
                              result_dict["gamma_structure_cluster"].append(gamma_structure)
                              result_dict["delta_graph_cluster"].append(delta_graph)
                              result_dict["embedding_size_cluster"].append(embedding_size)
                              result_dict["num_neurons_cluster"].append(num_neurons)
                              result_dict["ACC"].append(res[0])
                              result_dict["ARI"].append(res[1])
                              result_dict["NMI"].append(res[2])
result_data=pd.DataFrame(result_dict)
result_data.to_csv(f"{save_path}{dataset}/cluster/results.csv")

end_time = time.perf_counter()
print("\nDone!")






