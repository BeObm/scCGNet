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
from model import GraphStructureVGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph, get_device
import time

print("="*60)
print("Graph-Structure-First VGAE")
print("Target: ARI ~0.50-0.54 (based on Spectral-32 baseline)")
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

# Network hyperparameters - optimized for graph structure
embedding_size = 32  # Match Spectral-32 which got ARI=0.54
num_neurons = 512    # More capacity for graph learning
save_path = "./results/"

# Training hyperparameters - OPTIMIZED FOR FASTER CONVERGENCE
epochs_cluster = 500
lr_cluster = 0.0005  # Increased from 0.001 for faster learning

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
print("\n[5/6] Creating model...")
seed = 42

network = GraphStructureVGAE(
    adj=adj_norm,
    num_neurons=num_neurons,
    num_features=num_features,
    embedding_size=embedding_size,
    nClusters=nClusters,
    activation="Sigmoid",
    seed=seed,
    # Loss weights OPTIMIZED for faster convergence
    alpha_recons=0.6,       # Increased from 0.01 - allow some feature learning
    beta_cluster=50.0,      # Increased from 10.0 - stronger clustering drive
    gamma_structure=10.0    # Reduced from 20.0 - less constraint, faster movement
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
print(f"\nTraining time: {end_time - start_time:.2f} seconds")
print(f"\nBest Results (at epoch {res[4]}):")
print(f"  Accuracy:     {res[0]:.4f}")
print(f"  ARI:          {res[1]:.4f}")
print(f"  NMI:          {res[2]:.4f}")
print(f"\nComparison to Baselines:")
print(f"  K-means on features:     ARI ~0.24")
print(f"  Spectral-32 embedding:   ARI ~0.54 (target)")
print(f"  Your final ARI:          {res[1]:.4f}")
if res[1] >= 0.45:
    print(f"  ✓ SUCCESS: Close to spectral baseline!")
elif res[1] >= 0.35:
    print(f"  ~ MODERATE: Better than features, below spectral")
else:
    print(f"  ✗ POOR: Not learning graph structure effectively")
print("="*60)

print(f"\nResults saved to: {save_path}{dataset}/cluster/log.csv")