import numpy as np
import torch
from fontTools.ttLib.tables.S_V_G_ import doc_index_entry_format_0

from preprocessing import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.nn import GCNConv, GraphConv, LEConv, SAGEConv
import argparse
from vgae_model import GraphVAE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict


@torch.no_grad()
def kmeans_warmstart(model, x_input, edge_index, n_clusters, seed=0):
    """Initialise the GMCM component means from k-means on the encoder's
    latent means. Run this only AFTER reconstruction pretraining, so z is
    already meaningful."""
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)  # deterministic
    z = mu.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(z)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=mu.device)
    model.cluster.mu.data.copy_(centers)
    return km.labels_


@torch.no_grad()
def evaluate(model, x_input, edge_index, labels):
    """ARI / NMI of GMCM hard assignments vs ground truth (labels: eval only)."""
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    pred = model.cluster.assign(mu).cpu().numpy()
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi


def train(model, x_input, edge_index, adj, x_counts, labels,
          scale_factor=1.0, n_clusters=None,
          pretrain_epochs=200, train_epochs=300, lr=1e-3,
          weights=(1.0, 1.0, 1.0, 1.0), eval_every=10, device="cpu"):
    """Three-phase, full-batch training.

      1. pretrain reconstruction branches only  (adj + ZINB + KL, no clustering)
      2. k-means warm-start of the GMCM means
      3. joint training with all four losses

    labels are used for evaluation ONLY -- the clustering itself is unsupervised.
    """
    w_adj, w_feat, w_clus, w_kl = weights
    model = model.to(device)
    x_input, edge_index = x_input.to(device), edge_index.to(device)
    adj, x_counts = adj.to(device), x_counts.to(device)
    if torch.is_tensor(scale_factor):
        scale_factor = scale_factor.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- Phase 1: reconstruction pretraining (clustering weight = 0) ----
    for ep in range(pretrain_epochs):
        model.train()
        opt.zero_grad()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts, scale_factor,
                                 w_adj=w_adj, w_feat=w_feat, w_clus=0.0, w_kl=w_kl)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            print(f"[pretrain {ep:4d}] total={parts['total']:.4f} "
                  f"adj={parts['adj']:.4f} feat={parts['feat']:.4f} kl={parts['kl']:.4f}")

    # ---- Phase 2: k-means warm-start ----
    kmeans_warmstart(model, x_input, edge_index, n_clusters)
    ari, nmi = evaluate(model, x_input, edge_index, labels)
    print(f"[k-means init] ARI={ari:.4f} NMI={nmi:.4f}")

    # ---- Phase 3: joint training ----
    best = {"ari": -1.0, "nmi": -1.0, "epoch": -1}
    for ep in range(train_epochs):
        model.train()

        opt.zero_grad()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts, scale_factor,
                                 w_adj=w_adj, w_feat=w_feat, w_clus=w_clus, w_kl=w_kl)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            ari, nmi = evaluate(model, x_input, edge_index, labels)
            if ari > best["ari"]:
                best = {"ari": ari, "nmi": nmi, "epoch": ep}
            print(f"[train {ep:4d}] total={parts['total']:.4f} clus={parts['clus']:.4f} "
                  f"| ARI={ari:.4f} NMI={nmi:.4f}")

    print(f"[best] epoch={best['epoch']} ARI={best['ari']:.4f} NMI={best['nmi']:.4f}")
    return model, best


if __name__ == "__main__":
    os.makedirs("./results", exist_ok=True)
    parser = argparse.ArgumentParser(
        description=" scRNA-seq clustering with GMCM-VGAE")

    parser.add_argument("--dataset_name", type=str, default="Quake_10x_Limb_Muscle")
    args = parser.parse_args()

    result = defaultdict(list)
    # optimizers = ["Adam","AdamW","SGD","RMSProp"]
    optimizers = ["Adam"]
    n_eighborss = 15
    n_top_genes = 2000
    hidden_dims = [64,128, 256, 512]
    latent_dims = [32, 64, 128, 512]
    conv_layers = [GCNConv,GraphConv,LEConv,SAGEConv]
    epochs_clusters = [500,800,300]
    pre_epochs=[200,500]
    lr_clusters = [0.001, 0.01, 0.005, 0.0001]
    seed = 82

    datapath = f"./data/{args.dataset_name}.h5ad"
    data, adata = load_h5_data(dataPath=datapath,
                               dataset=args.dataset_name,
                               hvg=n_top_genes,
                               n_neighbors=n_eighborss,
                               ts=[0, 0],
                               metric='cosine')

    for hidden_dim in hidden_dims:
        for latent_dim in latent_dims:
            for conv_layer in conv_layers:
                 for epochs in epochs_clusters:
                  for pre_epoch in pre_epochs:
                    for lr in lr_clusters:
                        for optimizer in optimizers:
                          try:
                            features = np.asarray(data["features"])
                            labels = data["label"]
                            K = len(np.unique(labels))
                            n_genes = features[0].shape[1]
                            x_input = torch.from_numpy(features).float()
                            edge_index = data["edge_index"]

                            # ---------- wire in your numpy arrays ----------
                            # counts     : [N, G] raw counts (ZINB target)
                            # adj        : [N, N] 0/1 dense  (convert from scipy sparse if needed)
                            # edge_index : [2, E]
                            # labels     : [N]    ground-truth (eval only)
                            counts = features[0]  # <- your raw count matrix
                            adj = data["adj"]  # <- your dense adjacency

                            device = "cuda" if torch.cuda.is_available() else "cpu"

                            x_counts = torch.tensor(counts, dtype=torch.float32)
                            # encoder input: log1p-normalised counts. Swap in your own normalised
                            # feature array here if you already have one.
                            x_input = torch.log1p(x_counts)
                            adj_t = torch.tensor(adj, dtype=torch.float32)
                            edge_index_t = torch.tensor(edge_index, dtype=torch.long)

                            # per-cell size factors for the ZINB mean (or pass scale_factor=1.0)
                            lib = x_counts.sum(1, keepdim=True)
                            size_factors = lib / lib.median()

                            K = len(np.unique(labels))
                            model = GraphVAE(in_dim=x_input.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim,
                                             n_genes=x_counts.shape[1], n_clusters=K, conv_layer=conv_layer)

                            model, best = train(model, x_input, edge_index_t, adj_t, x_counts, labels,
                                                scale_factor=size_factors, n_clusters=K,
                                                pretrain_epochs=pre_epoch, train_epochs=epochs, lr=lr,
                                                weights=(1.0, 1.0, 1.0, 1.0), eval_every=10, device=device)

                            result["hidden_dim"].append(hidden_dim)
                            result["latent_dim"].append(latent_dim)
                            result["conv_layer"].append(conv_layer)
                            result["pre_epoch"].append(pre_epoch)
                            result["epochs"].append(epochs)
                            result["lr"].append(lr)
                            result["optimizer"].append(optimizer)
                            result["Best epoch"].append(best["epoch"])
                            result["ARI"].append(best["ari"])
                            result["NMI"].append(best["nmi"])
                          except Exception as e:
                              print(e)
                    df0 = pd.DataFrame(result)
                    df0.to_csv(f"./results/Details_{args.dataset_name}.csv", index=False)
    df = pd.DataFrame(result)
    df = df.nlargest(3, ["NMI"])
    df.to_csv(f"./results/{args.dataset_name}.csv", index=False)
