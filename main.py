
import os
import argparse
import time
import warnings
import torch
import pandas as pd
from torch_geometric.transforms import RandomLinkSplit
from preprocessing import get_device, load_data,clustering_metrics
from model1 import *
warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "15"

device=get_device()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train GMCM-VGAE for scRNA-seq clustering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ───
    p.add_argument("--dataset",      type=str,   default="Klein",
                   help="Dataset name (must match a folder under --data_path).")
    p.add_argument("--data_path",    type=str,   default=None,
                   help="Path to data folder. Defaults to ./data/<dataset>.")
    p.add_argument("--n_top_genes",  type=int,   default=2000,
                   help="Number of highly variable genes to select.")
    p.add_argument("--n_neighbors",  type=int,   default=5,
                   help="k for kNN graph construction.")
    p.add_argument("--n_pcs",        type=int,   default=15,
                   help="Number of PCs used for neighbor graph.")
    p.add_argument("--save_path",    type=str,   default="./results/",
                   help="Root directory for saving results and checkpoints.")

    # ── Model architecture ────────────────────────────────────────────────
    p.add_argument("--embedding_size",  type=int,   default=1024,
                   help="Dimensionality of the latent space z.")
    p.add_argument("--hidden_channels",     type=int,   default=518,
                   help="Hidden size of the GCN encoder layer.")
    p.add_argument("--gmcm_dim",        type=int,   default=64,
                   help="Projection dimension for the GMCM copula (must be >= 2).")
    p.add_argument("--activation",      type=str,   default="ReLU",
                   choices=["ReLU", "Sigmoid", "Tanh", "Linear"],
                   help="Encoder activation function.")
    p.add_argument("--min_clamp_mean",  type=float, default=1e-5)
    p.add_argument("--max_clamp_mean",  type=float, default=1e6)
    p.add_argument("--min_clamp_dis",   type=float, default=1e-4)
    p.add_argument("--max_clamp_dis",   type=float, default=1e4)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=800,
                   help="Maximum number of joint training epochs.")
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Learning rate.")
    p.add_argument("--wd",          type=float, default=1e-4,
                   help="Weight decay (L2 regularisation).")
    p.add_argument("--dropout", type=float, default=0.),
    p.add_argument("--momentum",    type=float, default=0.9,
                   help="Momentum (used only with SGD / RMSProp).")
    p.add_argument("--optimizer",   type=str,   default="Adam",
                   choices=["Adam", "SGD", "RMSProp"],
                   help="Optimiser.")
    p.add_argument("--tau_rank",    type=float, default=0.1,
                   help="Temperature for soft-rank copula transform.")
    p.add_argument("--seed",        type=int,   default=8,
                   help="Random seed for reproducibility.")

    return p.parse_args()



def main():
    args   = parse_args()
    device = get_device()
    print(f"Code running on device: {device} using {args.dataset} dataset")
    # Seed everything
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    
    data_path = f"./data/{args.dataset}"

    print(f"\n{'='*9}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Data path: {data_path}")
    print(f"{'='*9}\n")

    data, n_clusters = load_data(
        dataset=args.dataset,
        data_path=data_path,
        n_top_genes=args.n_top_genes,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
    )

    # print(
    #     f"\nGraph: {data.num_nodes} nodes | "
    #     f"{data.x.shape[1]} features | "
    #     f"{data.num_edges} edges | "
    #     f"{n_clusters} clusters\n"
    # )
    print(f"{'='*50}")
    print(f"  Config")
    print(f"{'='*55}")
    for k, v in vars(args).items():
        print(f"  {k:<20} : {v}")
    print(f"{'='*55}\n")
    model=""


    optimizer = torch.optim.Adam()

    for epoch in range(args.epochs):
        pass

    y_pred=gmcm.predict(z)

    acc, ari, nmi = clustering_metrics(data.y, y_pred)
    print(f"ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")




if __name__ == "__main__":
    main()
