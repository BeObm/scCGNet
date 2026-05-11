import os
os.environ["OMP_NUM_THREADS"] = "15"
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from collections import defaultdict
import scipy.sparse as sp
import pandas as pd
from model import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph,get_device
import time
save_path = "./results/"
datasets = ["baron3","baron4","Klein","Chung","YAN","facs_lung","droplet_lung","10X_PMBC","lps_int2","human_kidney","Muraro","Mouse","mouse_ES","worm_neuron","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Trachea","Quake_10x_Limb_Muscle","Quake_10x_Spleen","Quake_Smart-seq2_Diaphragm","Quake_Smart-seq2_Lung","Romanov"]
result=defaultdict(list)
for dataset in datasets:
    try:
        # Network hyperparameters
        embedding_size = 32
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
            if dataset in ["baron3", "baron4", "baron5"]:
                return sp.csr_matrix(t)
            else:
                return sp.csr_matrix(t.detach().cpu().numpy())

        # ------------------------------------------------------------------ #
        # Preprocess adjacency                                                 #
        # ------------------------------------------------------------------ #
        if dataset in ["baron3", "baron4", "baron5"]:
            features_new = features.astype(np.float32)
        else:
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
        result[dataset].append(dataset)
        result["ACC"].append(res[0])
        result["ARI"].append(res[1])
        result["NMI"].append(res[2])
        result["Epoch"].append(epochs_cluster)
        result["LR"].append(lr_cluster)
        result["WD"].append(wd)
        result["Momentum"].append(momentum)
        result["n_top_genes"].append(n_top_genes)
        result["n_neighbors"].append(n_neighbors)
        result["n_pcs"].append(n_pcs)
        result["num_features"].append(num_features)
        result["num_neurons"].append(num_neurons)
        result["embedding_size"].append(embedding_size)
        result["activation"].append(activation)
        result["optimizer"].append(optimizer)
        result["seed"].append(seed)
        result["min_clamp_dis"].append(min_clamp_dis)
        result["max_clamp_dis"].append(max_clamp_dis)
        result["min_clamp_mean"].append(min_clamp_mean)
        result["max_clamp_mean"].append(max_clamp_mean)
        result["nClusters"].append(nClusters)
        result["device"].append(device)

        end = time.perf_counter()
        print(f"Total time: {end - start:0.4f} seconds")
        print(f"Training results for {dataset}: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")

    except:
             pass

    pd.DataFrame(result).to_csv(f"./results/Results.csv")

