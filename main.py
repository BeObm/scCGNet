import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "15"

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import argparse
from vgaemodel import *
from preprocessing import load_h5_data,get_device,load_data1
import time

save_path = "./results/"
datasets = ["Adam","Muraro","worm_neuron","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Trachea","Quake_10x_Limb_Muscle","Quake_10x_Spleen","Quake_Smart-seq2_Diaphragm","Quake_Smart-seq2_Lung","Romanov","Young","baron3"]
#datasetse = ["Adam","baron3","baron4","baron3","Muraro","Campbell","Quake_Smart-seq2_Diaphragm","Quake_10x_Limb_Muscle_raw","Shekar","Tosches_turtle","Wang_Large_Intestine","Young"]
datasets=["Quake_10x_Bladder"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=" scRNA-seq clustering with GMCM-VGAE")

    parser.add_argument("--dataset_name", type=str, default="muraro_preprocessed")
    args = parser.parse_args()


    # datasets=[args.dataset_name]
    epochs_clusters = [800]
    lr_clusters = [0.001]
    embedding_sizes = [32]
    num_neuronss = [512]
    seed=8
    # epochs_clusters = [1]
    # lr_clusters = [0.0001, 0.01]
    # embedding_sizes = [32, 40]
    # num_neuronss = [64, 80, 90, 256, 512]
    # seed = 82
    optimizers=["Adam"]
    n_eighborss=15
    n_top_genes=2000


    n_top_genes = 2000
    if args.dataset_name in ["baron3", "baron4", "baron5"]:
        datapath = f"./data/{args.dataset_name}"
        adj, features, labels, nClusters= load_data1(args.dataset_name, datapath, True)
    else:
        datapath = f"./data/{args.dataset_name}.h5ad"
        data, adata = load_h5_data(dataPath=datapath,
                                   dataset=args.dataset_name,
                                   hvg=n_top_genes,
                                   n_neighbors=n_eighborss,
                                   ts=[0, 0],
                                   metric='cosine')
        adj=data["adj"]
        features=np.asarray(data["features"])
        labels=data["label"]
        K= len(np.unique(labels))
        n_genes=features[0].shape[1]
        x_input=torch.from_numpy(features).float()
        edge_index=data["edge_index"]
        x_counts=data["features"]
        X_log= normalize_log_counts(features[0])
        scaler = StandardScaler()
        x_counts = scaler.fit_transform(X_log)
        size_factors=0.05

        model = GraphVAE(in_dim=F, hidden_dim=256, latent_dim=32,
                         n_genes=n_genes, n_clusters=K, conv_layer=GCNConv)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts,
                                 scale_factor=size_factors)  # adj dense float [N,N]
        loss.backward();
        opt.step()
