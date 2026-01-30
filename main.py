import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import scipy.sparse as sp
from model_final import GMCM_VGAE_Final
from preprocessing import load_data, sparse_to_tuple, preprocess_graph, get_device
import time

print("="*60)
print("Final Optimized GMCM-VGAE")
print("Combines: GMCM + Spectral Init + Graph Structure + All Fixes")
print("="*60)

# Dataset
dataset = "baron3"
nClusters = 14

# Load data
print("\n[1/6] Loading data...")
adj, features, labels = load_data('baron3', './data/baron3', True)

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

network = GMCM_VGAE_Final(
    adj=adj_norm,
    num_neurons=num_neurons,
    num_features=num_features,
    embedding_size=embedding_size,
    nClusters=nClusters,
    activation="Sigmoid",
    seed=seed,
    # Optimized loss weights
    alpha_recons=0.1,     # Reconstruction
    beta_gmcm=5.0,        # GMCM clustering
    gamma_zinb=0.1,       # ZINB
    delta_graph=10.0      # Graph structure preservation
)

network.to(device)
print(f"  Model parameters: {sum(p.numel() for p in network.parameters())}")

# Training
print("\n[6/6] Starting training...")
print("="*60)

start_time = time.perf_counter()

res, y_pred, y = network.train(
    acc_list=[],
    adj_norm=adj_norm,
    features=features,
    adj_label=adj_label,
    y=labels,
    weight_tensor=weight_tensor_orig,
    norm=norm,
    optimizer="Adam",
    epochs=epochs_cluster,
    lr=lr_cluster,
    save_path=save_path,
    dataset=dataset,
    features_new=features_new
)

end_time = time.perf_counter()

# Results
print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"\nTraining time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.2f} minutes)")
print(f"\nBest Results (at epoch {res[4]}):")
print(f"  Accuracy:     {res[0]:.4f}")
print(f"  ARI:          {res[1]:.4f}")
print(f"  NMI:          {res[2]:.4f}")
print(f"\nComparison to Baselines:")
print(f"  K-means on features:     ARI ~0.24")
print(f"  Your original model:     ARI ~0.18 (decreasing)")
print(f"  Spectral-32 (target):    ARI ~0.54")
print(f"  Final GMCM-VGAE:         ARI  {res[1]:.4f}")

if res[1] >= 0.50:
    print(f"\n   EXCELLENT: Matches or exceeds spectral baseline!")
elif res[1] >= 0.40:
    print(f"\n  ✓ GOOD: Strong improvement, approaching spectral baseline")
elif res[1] >= 0.30:
    print(f"\n  ~ MODERATE: Better than feature-based, room for improvement")
else:
    print(f"\n  NEEDS WORK: Below expectations")

print(f"\nPrediction Distribution:")
print(f"  Predicted clusters: {len(np.unique(y_pred))}")
print(f"  True clusters:      {len(np.unique(y))}")
print(f"  Cluster sizes: {np.bincount(y_pred)}")
print("="*60)

print(f"\nResults saved to: {save_path}{dataset}/cluster/log.csv")
print("\nDone!")






# epochs_cluster_0 = [200,500,750,1000,1500,2000,2500,3000,3500,4000,4500,5000]
# lr_cluster_0 = [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.0005,0.00005] # Increased from 0.001 for faster learning
# alpha_recons_0=[0.00001, 0.01,0.002, 0.003, 0.004, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# beta_cluster_0=[0.1,1,10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
# gamma_structure_0=[0.1,1,10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
# embedding_size_0 = [8,16,32,64,128]
# num_neurons_0 = [16,32,64,128,512,1024]    # More capacity for graph learning
#
# result_dict=defaultdict(list)
# start_time = time.perf_counter()
#
# for epochs_cluster in tqdm(epochs_cluster_0):
#     for lr_cluster in lr_cluster_0:
#         for alpha_recons in alpha_recons_0:
#             for beta_cluster in beta_cluster_0:
#                 for gamma_structure in gamma_structure_0:
#                         for embedding_size in embedding_size_0:
#                             for num_neurons in num_neurons_0:

