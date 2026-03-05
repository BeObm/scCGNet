import pyreadr
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import torch
from torch_geometric.data import Data

def read_rds(norm_filepath, meta_filepath, excel_out="cluster_distribution.xlsx"):

    # read expression matrix
    norm_data = pyreadr.read_r(norm_filepath)
    dataset = list(norm_data.values())[0].T   # cells × genes

    # read metadata
    label_data = pyreadr.read_r(meta_filepath)
    label = list(label_data.values())[0]

    # check cell alignment
    if list(dataset.index) == list(label.index):
        print("Cell indices are aligned in the two files")
    elif set(dataset.index) == set(label.index):
        raise ValueError("Same cells but different order. Please reorder metadata.")
    else:
        raise ValueError("Cell indices do not match between files.")

    # detect cluster column
    cluster_col = None
    for col in label.columns:
        c = col.lower()
        if "cluster" in c or "id" in c or "label" in c:
            cluster_col = col
            break

    if cluster_col is None:
        raise ValueError("No cluster/label column found in metadata.")

    # extract data
    X = dataset.to_numpy()
    y = label[cluster_col]

    print(f"X shape: {X.shape} ({X.shape[0]} cells, {X.shape[1]} genes)")

    n_clusters= save_cluster_distribution(y, excel_out)

    return X, y,n_clusters


def save_cluster_distribution(y, excel_out="cluster_distribution.xlsx"):
    # cluster distribution
    clusters, counts = np.unique(y, return_counts=True)
    n_clusters = len(clusters)

    print(f"Number of clusters: {n_clusters}")

    dist_df = pd.DataFrame({
        "cluster": clusters,
        "count": counts
    })
    # save distribution to Excel
    dist_df.to_excel(excel_out, index=False)
    return n_clusters

def h5_xy_to_pyg(path, n_neighbors=15, n_pcs=50,excel_out="cluster_distribution.xlsx"):

    # read dataset
    with h5py.File(path, "r") as f:
        X = f["X"][:]
        y = f["Y"][:]

    # build AnnData
    adata = ad.AnnData(X)
    adata.obs["label"] = y

    # scanpy preprocessing
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # adjacency matrix
    A = adata.obsp["connectivities"]

    # convert to edge_index
    edge_index = np.vstack(A.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # node features (PCA embedding)
    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)

    # labels
    y = torch.tensor(adata.obs["label"].values, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    n_clusters=save_cluster_distribution(y, excel_out)
    print("Nodes:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Clusters:", len(np.unique(y.numpy())))

    return data, n_clusters

def h5_to_pyg(path, n_neighbors=15, n_pcs=50,excel_out="cluster_distribution.xlsx"):
    """
    Read Quake_Smart-seq2_Trachea.h5 and convert it to a PyTorch Geometric graph.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    n_neighbors : int
        Number of neighbors used to build the KNN graph.
    n_pcs : int
        Number of principal components used for the graph.

    Returns
    -------
    data : torch_geometric.data.Data
        PyG graph object
    adata : AnnData
        AnnData object containing the processed dataset
    """

    # --- read file ---
    with h5py.File(path, "r") as f:

        grp = f["exprs"]

        X = sp.csr_matrix(
            (grp["data"][:], grp["indices"][:], grp["indptr"][:]),
            shape=grp["shape"][:]
        )

        cells = f["obs_names"][:].astype(str)
        genes = f["var_names"][:].astype(str)

        y = f["obs"]["cluster"][:]

    # --- build AnnData ---
    adata = ad.AnnData(X)
    adata.obs_names = cells
    adata.var_names = genes
    adata.obs["cluster"] = y

    # --- Scanpy pipeline ---
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    # adjacency matrix
    A = adata.obsp["connectivities"]

    edge_index = np.vstack(A.nonzero())
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = torch.tensor(adata.obsm["X_pca"], dtype=torch.float)
    y = torch.tensor(adata.obs["cluster"].values)
    n_clusters=save_cluster_distribution(y, excel_out)
    data = Data(x=x, edge_index=edge_index, y=y)

    print("Cells:", data.num_nodes)
    print("Edges:", data.num_edges)
    print("Clusters:", len(np.unique(y.numpy())))

    return data, n_clusters

def read_tsv(norm_filepath, meta_filepath, excel_out="cluster_distribution.xlsx"):

    # read expression matrix
    dataset = pd.read_csv(norm_filepath, sep='\t')
    dataset = dataset.T
    # read metadata
    label = pd.read_csv(meta_filepath, sep=',')

    # print([i for i in list(label["cell_name"].values)])
    # print([i for i in dataset.index[1:]])
    # check cell alignment
    if list(dataset.index[1:]) == list(label["cell_name"].values):
        print("Cell indices are aligned in the two files")
    elif set(dataset.index) == set(label.index):
        raise ValueError("Same cells but different order. Please reorder metadata.")
    else:
        raise ValueError("Cell indices do not match between files.")

    # detect cluster column
    cluster_col = "cell_type"

    # extract data
    X = dataset.to_numpy()
    y = label[cluster_col]

    print(f"X shape: {X.shape} ({X.shape[0]} cells, {X.shape[1]} genes)")

    # cluster distribution
    clusters, counts = np.unique(y, return_counts=True)
    n_clusters = len(clusters)

    print(f"Number of clusters: {n_clusters}")

    dist_df = pd.DataFrame({
        "cluster": clusters,
        "count": counts
    })

    # print("Cluster distribution:")
    # print(dist_df)

    # save distribution to Excel
    dist_df.to_excel(excel_out, index=False)

    print(f"Cluster distribution saved to: {excel_out}")

    return X, y,n_clusters


def cell_matrix_to_graph(X, y, n_neighbors=15, n_pcs=50):

    adata = sc.AnnData(X)
    adata.obs["label"] = y

    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    A = adata.obsp["connectivities"]

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)

    data = Data(
        x=torch.tensor(X, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y)
    )

    return data