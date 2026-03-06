import pyreadr
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import anndata as ad
import scanpy as sc
import torch
from torch_geometric.data import Data

def read_rds(norm_filepath, meta_filepath, csv_out="cluster_distribution.csv"):

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


    n_clusters= save_cluster_distribution(y, csv_out)

    return X, y,n_clusters


def save_cluster_distribution(y, csv_out="cluster_distribution.csv"):
    # cluster distribution
    clusters, counts = np.unique(y, return_counts=True)
    n_clusters = len(clusters)

    dist_df = pd.DataFrame({
        "cluster": clusters,
        "count": counts
    })
    # save distribution to Excel
    dist_df.to_csv(csv_out, index=False)
    return n_clusters

def h5_xy_to_pyg(path, n_neighbors=15, n_pcs=50,csv_out="cluster_distribution.csv"):

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
    n_clusters=save_cluster_distribution(y, csv_out)


    return data, n_clusters

def h5_to_pyg(path, n_neighbors=15, n_pcs=50,csv_out="cluster_distribution.csv"):
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
    n_clusters=save_cluster_distribution(y, csv_out)
    data = Data(x=x, edge_index=edge_index, y=y)

    return data, n_clusters

def read_tsv(norm_filepath, meta_filepath, csv_out="cluster_distribution.csv"):

    # read expression matrix
    dataset = pd.read_csv(norm_filepath, sep='\t', index_col=0)

    dataset = dataset.T
    # read metadata
    label = pd.read_csv(meta_filepath, sep=',')


    # check cell alignment
    if list(dataset.index) == list(label["cell_name"].values):
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

    # cluster distribution
    clusters, counts = np.unique(y, return_counts=True)
    n_clusters = len(clusters)

    dist_df = pd.DataFrame({
        "cluster": clusters,
        "count": counts
    })

    # save distribution to Excel
    dist_df.to_csv(csv_out, index=False)


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



def check_normalization(X):

    diagnostics = {}
    if X.is_sparse:
        print("Matrix is sparse")
    else:
        print("Matrix is dense")

    if X.layout != torch.strided:
        print("Matrix with Sparse tensor")
    else:
        print("Matrix with Dense tensor")
    # integer fraction
    diagnostics["fraction_integer"] = torch.mean((X % 1 == 0).float()).item()

    diagnostics["min_value"] = torch.min(X).item()
    diagnostics["max_value"] = torch.max(X).item()

    # library size per cell
    libsize = torch.sum(X, dim=1)

    mean_lib = torch.mean(libsize).item()
    std_lib = torch.std(libsize).item()

    diagnostics["library_size_mean"] = mean_lib
    diagnostics["library_size_std"] = std_lib

    if mean_lib == 0:
        diagnostics["library_size_cv"] = None
        diagnostics["likely_type"] = "all_zero_matrix_or_invalid_input"
        return diagnostics

    diagnostics["library_size_cv"] = std_lib / mean_lib

    # heuristic classification
    if diagnostics["fraction_integer"] > 0.95 and diagnostics["max_value"] > 50:
        diagnostics["likely_type"] = "raw_counts"
    elif diagnostics["max_value"] < 20 and diagnostics["fraction_integer"] < 0.5:
        diagnostics["likely_type"] = "log_normalized"
    else:
        diagnostics["likely_type"] = "possibly_normalized"

    diagnostics_all={k: v for k, v in reversed(diagnostics.items())}
    return diagnostics_all


def check_sparse(X):

    results = {}

    # Method 1: layout
    results["layout_check"] = (X.layout != torch.strided)

    # Method 2: is_sparse attribute (COO only but still informative)
    results["is_sparse_attr"] = getattr(X, "is_sparse", False)


    # Method 4: try accessing indices
    try:
        _ = X._indices()
        results["indices_access"] = True
    except:
        results["indices_access"] = False

    # Method 5: try accessing nnz
    try:
        _ = X._nnz()
        results["nnz_access"] = True
    except:
        results["nnz_access"] = False

    # Collect decisions
    decisions = list(results.values())

    # Check agreement
    if len(set(decisions)) != 1:
        raise RuntimeError(
            f"Inconsistent sparse detection results: {results}"
        )

    is_sparse = decisions[0]

    # Additional informational diagnostic (not used for decision)
    zero_ratio = (X == 0).sum().item() / X.numel()

    return {"is_sparse": is_sparse}