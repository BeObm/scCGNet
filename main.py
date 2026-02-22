import os

from toolz.tests.test_dicttoolz import defaultdict

os.environ["OMP_NUM_THREADS"] = "15"
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import scipy.sparse as sp
from model import GMCM_VGAE
from preprocessing import load_data, sparse_to_tuple, preprocess_graph
import time
import itertools
import pandas as pd

save_path = "./results/"
dataset = "baron3"
nClusters = 14

# # ---- SEARCH SPACE ----
embedding_sizeL = [45,50,64]
num_neuronsL = [64,90,256]
activationL = ["Sigmoid","Linear"]
optimizerL = ["Adam","SGD"]
seedL = [82,8,42]
wdL = [0.01]
momentumL = [0.9]
min_clamp_meanL = [1e-5]
max_clamp_meanL = [1e6]
min_clamp_disL = [1e-4]
max_clamp_disL = [1e4]

epochs_clusteL = [300,800,1000]
lr_clusterL = [0.0001,0.001,0.01]
# -----------------------


device = torch.device("cpu")
print("CUDA available:", torch.cuda.is_available())

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

print("Start grid search")
start = time.perf_counter()

results = defaultdict(list)
total_configs = 1

grid = itertools.product(
    embedding_sizeL,
    num_neuronsL,
    activationL,
    optimizerL,
    seedL,
    wdL,
    momentumL,
    min_clamp_meanL,
    max_clamp_meanL,
    min_clamp_disL,
    max_clamp_disL,
    epochs_clusteL,
    lr_clusterL,
)





total_configs = (
    len(embedding_sizeL)
    * len(num_neuronsL)
    * len(activationL)
    * len(optimizerL)
    * len(seedL)
    * len(wdL)
    * len(momentumL)
    * len(min_clamp_meanL)
    * len(max_clamp_meanL)
    * len(min_clamp_disL)
    * len(max_clamp_disL)
    * len(epochs_clusteL)
    * len(lr_clusterL)
)

print("Total number of configurations:", total_configs)

i=1
for combo in grid:

    try:
        (embedding_size,
         num_neurons,
         activation,
         optimizer,
         seed,
         wd,
         momentum,
         min_clamp_mean,
         max_clamp_mean,
         min_clamp_dis,
         max_clamp_dis,
         epochs_cluster,
         lr_cluster) = combo

        config = {
            "embedding_size": embedding_size,
            "num_neurons": num_neurons,
            "activation": activation,
            "optimizer": optimizer,
            "seed": seed,
            "wd": wd,
            "momentum": momentum,
            "min_clamp_mean": min_clamp_mean,
            "max_clamp_mean": max_clamp_mean,
            "min_clamp_dis": min_clamp_dis,
            "max_clamp_dis": max_clamp_dis,
            "epochs_cluster": epochs_cluster,
            "lr_cluster": lr_cluster,
        }

        torch.manual_seed(seed)
        run_start = time.perf_counter()
        print(f"\n ###{i} -- Training with: {config}")
        i+=1
        network = GMCM_VGAE(
            adj=adj_norm,
            num_neurons=num_neurons,
            num_features=num_features,
            embedding_size=embedding_size,
            nClusters=nClusters,
            activation=activation,
            seed=seed,
            min_clamp_dis=min_clamp_dis,
            max_clamp_dis=max_clamp_dis,
            min_clamp_mean=min_clamp_mean,
            max_clamp_mean=max_clamp_mean
        ).to(device)

        res, y_pred, y = network.train(
            [],
            adj_norm,
            features,
            adj_label,
            labels,
            weight_tensor_orig,
            norm,
            optimizer=optimizer,
            epochs=epochs_cluster,
            lr=lr_cluster,
            wd=wd,
            momentum=momentum,
            save_path=save_path,
            dataset=dataset,
            features_new=features_new
        )
        print(f"Training results: Acc={res[0]} | ARI={res[1]}, NMI={res[2]}")

        run_time = time.perf_counter() - run_start
        results["ACC"].append(res[0])
        results["ARI"].append(res[1])
        results["NMI"].append(res[2])
        for k,v in config.items():
            results[k].append(v)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except:
        print(f"Error with config: {i}")
        results["ACC"].append(0)
        results["ARI"].append(0)
        results["NMI"].append(0)
        for k, v in config.items():
            results[k].append(v)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        continue
end = time.perf_counter()

print(f"Total grid time: {end - start:0.4f} seconds")

df = pd.DataFrame(results)
df.to_excel("grid_search_results.xlsx")

print("Saved to grid_search_results.xlsx")
print(df.head())

