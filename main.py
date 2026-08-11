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
from collections import defaultdict
save_path = "./results/"
datasets = ["Adam","Muraro","worm_neuron","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Trachea","Quake_10x_Limb_Muscle","Quake_10x_Spleen","Quake_Smart-seq2_Diaphragm","Quake_Smart-seq2_Lung","Romanov","Young","baron3"]
#datasetse = ["Adam","baron3","baron4","baron3","Muraro","Campbell","Quake_Smart-seq2_Diaphragm","Quake_10x_Limb_Muscle_raw","Shekar","Tosches_turtle","Wang_Large_Intestine","Young"]
datasets=["Quake_10x_Bladder"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=" scRNA-seq clustering with GMCM-VGAE")

    parser.add_argument("--dataset_name", type=str, default="Adam")
    args = parser.parse_args()

    result = defaultdict(list)
    # optimizers = ["Adam","AdamW","SGD","RMSProp"]
    optimizers = ["Adam"]
    n_eighborss = 15
    n_top_genes = 2000
    hidden_dims = [128, 256, 512]
    latent_dims = [32, 64, 128]
    conv_layers = [GCNConv, GCNConv, GCNConv, GCNConv]
    epochs_clusters = [500, 800]
    lr_clusters = [0.001, 0.01, 0.005]
    seed = 8

    datapath = f"./data/{args.dataset_name}.h5ad"
    data, adata = load_h5_data(dataPath=datapath,
                               dataset=args.dataset_name,
                               hvg=n_top_genes,
                               n_neighbors=n_eighborss,
                               ts=[5, 0],
                               metric='cosine')

    for hidden_dim in range(len(hidden_dims)):
        for latent_dim in range(len(latent_dims)):
            for conv_layer in range(len(conv_layers)):
                for epochs in range(len(epochs_clusters)):
                    for lr in range(len(lr_clusters)):
                        for optimizer in range(len(optimizers)):
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
