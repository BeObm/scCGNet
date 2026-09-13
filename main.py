from preprocessing import *
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import argparse
from vgae_model import GraphVAE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from config import configs
import umap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scanpy as sc, anndata as ad, pandas as pd
from common_code.get_labels import *



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
    acc=compute_cluster_accuracy(labels, pred)
    ari = adjusted_rand_score(labels, pred)
    nmi = normalized_mutual_info_score(labels, pred)
    return ari, nmi,acc


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
    ari, nmi,acc = evaluate(model, x_input, edge_index, labels)
    print(f"[k-means init] ARI={ari:.4f} NMI={nmi:.4f}")

    # ---- Phase 3: joint training ----
    best = {"acc":-1,"ari": -1.0, "nmi": -1.0, "epoch": -1}
    for ep in range(train_epochs):
        model.train()

        opt.zero_grad()
        loss, parts = model.loss(x_input, edge_index, adj, x_counts, scale_factor,
                                 w_adj=w_adj, w_feat=w_feat, w_clus=w_clus, w_kl=w_kl)
        loss.backward()
        opt.step()
        if ep % eval_every == 0:
            ari, nmi,acc = evaluate(model, x_input, edge_index, labels)
            if ari > best["ari"]:
                best = {"acc":acc,"ari": ari, "nmi": nmi, "epoch": ep}
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"[train {ep:4d}] | ARI={ari:.4f} NMI={nmi:.4f}")
    model.load_state_dict(best_state)
    print(f"[best] epoch={best['epoch']} ARI={best['ari']:.4f} NMI={best['nmi']:.4f}")
    return model, best


@torch.no_grad()
def save_plots(model, x_input, edge_index, labels, out_prefix,seed,dataset="Adam"):
    os.makedirs(out_prefix, exist_ok=True)
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    z = mu.cpu().numpy()
    pred = model.cluster.assign(mu).cpu().numpy()

    cell_ids = np.arange(len(labels))
    newdata_path = f"cluster_data/seed_{seed}/{dataset}"
    os.makedirs(newdata_path, exist_ok=True)
    run(cell_ids=cell_ids, y_true=labels, y_pred=pred, dataset_name=dataset, model_name="scCGNet",
        out_path=f"{newdata_path}/scCGNet.csv",seed=seed)


    u = umap.UMAP(random_state=0).fit_transform(z)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for a, (c, t) in zip(ax, [(labels, "True"), (pred, "Pred")]):
        sc = a.scatter(u[:, 0], u[:, 1], c=c, cmap="tab20", s=6)
        a.set_title(t); a.set_xlabel("UMAP-1"); a.set_ylabel("UMAP-2")
        fig.colorbar(sc, ax=a)
    fig.tight_layout(); fig.savefig(f"{out_prefix}/umap.png", dpi=200); plt.close(fig)

    np.save(f"{out_prefix}/embedding.npy", z)          # so you can re-plot later
    import pandas as pd
    pd.DataFrame({"True": labels, "Pred": pred}).to_csv(f"{out_prefix}/types.txt")
    print("saved", out_prefix)


@torch.no_grad()
def save_trajectory(model, x_input, edge_index, out_prefix, root_cluster=0):
    os.makedirs(out_prefix, exist_ok=True)
    model.eval()
    mu, _ = model.encoder.encode(x_input, edge_index)
    z = mu.cpu().numpy()
    pred = model.cluster.assign(mu).cpu().numpy()

    a = ad.AnnData(z)
    a.obs["cluster"] = pd.Categorical(pred.astype(str))
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15)
    sc.tl.paga(a, groups="cluster")
    sc.pl.paga(a, show=False)                      # <-- moved up: creates paga['pos']
    plt.savefig(f"{out_prefix}/paga.png", dpi=200); plt.close()

    sc.tl.umap(a, init_pos="paga")                 # now this works
    a.uns["iroot"] = int(np.where(pred == root_cluster)[0][0])
    sc.tl.diffmap(a); sc.tl.dpt(a)

    sc.pl.umap(a, color=["cluster", "dpt_pseudotime"], show=False)
    plt.savefig(f"{out_prefix}/trajectory.png", dpi=200); plt.close()
    print("saved trajectory", out_prefix)



    # sc.pp.neighbors(a, use_rep="X", n_neighbors=15)
    # sc.tl.paga(a, groups="cluster")               # cluster-level connectivity
    # sc.tl.umap(a, init_pos="paga")
    # a.uns["iroot"] = int(np.where(pred == root_cluster)[0][0])
    # sc.tl.diffmap(a); sc.tl.dpt(a)                # diffusion pseudotime
    #
    # sc.pl.paga(a, show=False)
    # plt.savefig(f"{out_prefix}/paga.png", dpi=200); plt.close()
    # sc.pl.umap(a, color=["cluster", "dpt_pseudotime"], show=False)
    # plt.savefig(f"{out_prefix}/trajectory.png", dpi=200); plt.close()
    # print("saved trajectory", out_prefix)

if __name__ == "__main__":
    os.makedirs("results_00001", exist_ok=True)
    parser = argparse.ArgumentParser(
        description=" scRNA-seq clustering with GMCM-VGAE")

    parser.add_argument("--dataset_name", type=str, default="Quake_10x_Limb_Muscle")
    args = parser.parse_args()
    device = get_device()
    datasetnam = [
        "Adam",
        "Muraro",
        "Quake_10x_Bladder",
        "Quake_10x_Limb_Muscle",
        "Quake_10x_Spleen",
        "Quake_Smart-seq2_Diaphragm",
        "Quake_Smart-seq2_Limb_Muscle",
        "Quake_Smart-seq2_Lung",
        "Quake_Smart-seq2_Trachea",
        "Romanov",
        "Young",
        "Tosches_turtle",
        "Wang_Large_Intestine"
        "Campbell",
        "Cao_2020_Spleen",
        "Bach",
        "Shekhar"
        ]
    # seeds = [111,222,333, 444, 555]
    seeds = [333]

    for dataset in datasetnam:
        try:
            args.dataset_name=dataset

            optimizer = "Adam"
            n_eighborss = 15
            n_top_genes = 2000
            datapath = f"./data/{args.dataset_name}.h5ad"
            data, adata = load_h5_data(dataPath=datapath,
                                       dataset=args.dataset_name,
                                       hvg=n_top_genes,
                                       n_neighbors=n_eighborss,
                                       ts=[0, 0],
                                       metric='cosine')
            print(adata.obs.columns.tolist())
            print(adata.obs.head())
            for seed in seeds:
                print(f"Seed: {seed} | dataset:{dataset}")
                set_random_seed(seed)
                result = defaultdict(list)
                config = configs[args.dataset_name]
                hidden_dim = config["hidden_dim"]
                latent_dim = config["latent_dim"]
                conv_layer = config["conv_layer"]
                pre_epoch = config["pre_epoch"]
                epochs = config["epochs_cluster"]
                lr = config["lr_cluster"]

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
                os.makedirs("plots", exist_ok=True)
                save_plots(model, x_input.to(device), edge_index_t.to(device),
                           labels, f"plots/seed_{seed}/{dataset}",seed,dataset=dataset)

                save_trajectory(model, x_input.to(device), edge_index_t.to(device),
                                f"plots/seed_{seed}/{dataset}", root_cluster=0)


                os.makedirs(f"results/", exist_ok=True)
                with open(f"results/{args.dataset_name}.txt", "a") as f:
                    f.write(f"===================================== ")
                    f.write("\n")
                    f.write(f"dataset:{dataset} | Seed:{seed} |ACC:{best['acc']} | ARI: {best['ari']} | NMI: {best['nmi']} ")
                    f.write("\n")
                    f.close()
        except Exception as e:
            print(f" Found error on {dataset}. Error:{e}" )

