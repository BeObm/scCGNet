import os
from preprocessing import to_pyg_data,build_pyg_data
from toolz.tests.test_dicttoolz import defaultdict
from torch_geometric.transforms import RandomLinkSplit
from preprocessing import get_device
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
dataset = "Chung"

# # ---- SEARCH SPACE ----
embedding_sizeL = [64,256]
num_neuronsL = [64,256]
activationL = ["ReLU"]
optimizerL = ["Adam"]
seedL = [8]
wdL = [0.0,0.001]
tau_rankL = [0.1]
momentumL = [0.9]
min_clamp_meanL = [1e-5]
max_clamp_meanL = [1e6]
min_clamp_disL = [1e-4]
max_clamp_disL = [1e4]
gmcm_dimL = [32,64]
epochs_clusteL = [800]
lr_clusterL = [0.001,0.0001]
# -----------------------


device = get_device()

adj, features, labels = load_data(dataset, f'./data/{dataset}')
data,nClusters=build_pyg_data(adj,features,labels)

print("-----------------------------------------------")
print(features.min(), features.max())
print(np.allclose(features, np.round(features)))  # should be True for counts



print(f"The dataset has {data.num_nodes} nodes, {data.x.shape[1]} feature, {data.num_edges} edges and {nClusters} clusters")
splitter = RandomLinkSplit(
    num_val=0.0,
    num_test=0.0,
    is_undirected=True,
    add_negative_train_samples=True,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = splitter(data)

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
    tau_rankL,
    momentumL,
    min_clamp_meanL,
    max_clamp_meanL,
    min_clamp_disL,
    max_clamp_disL,
    epochs_clusteL,
    lr_clusterL,
    gmcm_dimL
)
total_configs = (
    len(embedding_sizeL)
    * len(num_neuronsL)
    * len(activationL)
    * len(optimizerL)
    * len(seedL)
    * len(wdL)
    * len(momentumL)
    * len(tau_rankL)
    * len(min_clamp_meanL)
    * len(max_clamp_meanL)
    * len(min_clamp_disL)
    * len(max_clamp_disL)
    * len(epochs_clusteL)
    * len(lr_clusterL)
    * len(gmcm_dimL)
)

print("Total number of configurations:", total_configs)

i=1
for combo in grid:

    # try:
        (embedding_size,
         num_neurons,
         activation,
         optimizer,
         seed,
         wd,
         tau_rank,
         momentum,
         min_clamp_mean,
         max_clamp_mean,
         min_clamp_dis,
         max_clamp_dis,
         epochs_cluster,
         lr_cluster,
         gmcm_dim) = combo

        config = {
            "embedding_size": embedding_size,
            "num_neurons": num_neurons,
            "activation": activation,
            "optimizer": optimizer,
            "seed": seed,
            "wd": wd,
            "tau_rank": tau_rank,
            "momentum": momentum,
            "min_clamp_mean": min_clamp_mean,
            "max_clamp_mean": max_clamp_mean,
            "min_clamp_dis": min_clamp_dis,
            "max_clamp_dis": max_clamp_dis,
            "epochs_cluster": epochs_cluster,
            "lr_cluster": lr_cluster,
            "gmcm_dim":gmcm_dim
        }

        torch.manual_seed(seed)
        run_start = time.perf_counter()
        print(f"\n ###{i} -- Training with: {config}")
        i+=1
        network = GMCM_VGAE(
            data=train_data,
            num_neurons=num_neurons,
            gmcm_dim=gmcm_dim,
            num_features=features.shape[1],
            embedding_size=embedding_size,
            nClusters=nClusters,
            activation=activation,
            tau_rank=tau_rank,
            seed=seed,
            min_clamp_dis=min_clamp_dis,
            max_clamp_dis=max_clamp_dis,
            min_clamp_mean=min_clamp_mean,
            max_clamp_mean=max_clamp_mean
        ).to(device)

        ari, nmi, acc  = network.train(
            train_data,
            optimizer=optimizer,
            epochs=epochs_cluster,
            lr=lr_cluster,
            wd=wd,
            momentum=momentum,
            save_path=save_path,
            dataset=dataset
        )
        print(f"Training results: Acc={acc} | ARI={ari}, NMI={nmi}")

        run_time = time.perf_counter() - run_start
        results["ACC"].append(acc)
        results["ARI"].append(ari)
        results["NMI"].append(nmi)
        for k,v in config.items():
            results[k].append(v)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # except:
    #     print(f"Error with config: {i}")
    #     results["ACC"].append(0)
    #     results["ARI"].append(0)
    #     results["NMI"].append(0)
    #     for k, v in config.items():
    #         results[k].append(v)
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
    #     continue
end = time.perf_counter()

print(f"Total grid time: {end - start:0.4f} seconds")

df = pd.DataFrame(results)
df.to_excel("grid_search_results.xlsx")

print("Saved to grid_search_results.xlsx")
print(df.head())

