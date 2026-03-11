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
    dataset = dataset.iloc[1:, :]
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








import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

















def build_pyg_graph(
    cell_gene_matrix=None,          # np.ndarray or pd.DataFrame: shape (n_cells, n_genes)
    cell_labels=None,               # array-like: length n_cells, categorical labels
    n_top_genes: int = 2000,   # HVGs to select
    n_neighbors: int = 15,     # kNN neighbors
    n_pcs: int = 50,
    file=None # PCs for neighbor graph
) -> Data:
    """
    Build a PyTorch Geometric graph from a cell x gene matrix.

    Pipeline:
        Raw counts → QC → Log1p → Top-k HVGs → PCA → kNN Graph → PyG Data

    Args:
        cell_gene_matrix: Raw count matrix (cells × genes).
        cell_labels:      Cell type / condition labels per cell.
        n_top_genes:      Number of highly variable genes to keep.
        n_neighbors:      k for kNN graph construction.
        n_pcs:            Number of PCs used for neighbor computation.

    Returns:
        torch_geometric.data.Data with:
            - x:          Node features (cells × HVGs), log-normalised
            - edge_index: COO edge list from kNN graph  (2 × E)
            - edge_attr:  Connectivities / weights       (E,)
            - y:          Integer-encoded cell labels    (N,)
    """

    # ------------------------------------------------------------------ #
    # 1. Build AnnData                                                     #
    # ------------------------------------------------------------------ #
    if file is not None:
        with h5py.File(file, "r") as f:
            X = f["X"][:]
            y = f["Y"][:]

        # build AnnData
        adata = ad.AnnData(X)
        adata.obs["label"] = y

    else:
        if isinstance(cell_gene_matrix, pd.DataFrame):
            adata = sc.AnnData(X=cell_gene_matrix.values.astype(np.float32))
            adata.var_names = cell_gene_matrix.columns.astype(str)
            adata.obs_names = cell_gene_matrix.index.astype(str)
        else:
            adata = sc.AnnData(X=cell_gene_matrix.astype(np.float32))

        adata.obs["label"] = np.array(cell_labels)
        print(f"[1] Input:  {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------ #
    # 2. QC filtering                                                      #
    # ------------------------------------------------------------------ #
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    print(f"[2] Post-QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------ #
    # 3. Normalise & log-transform                                         #
    # ------------------------------------------------------------------ #
    sc.pp.normalize_total(adata, target_sum=1e4)   # library-size normalisation
    sc.pp.log1p(adata)                              # log(x + 1)

    # ------------------------------------------------------------------ #
    # 4. Highly variable genes                                             #
    # ------------------------------------------------------------------ #
    actual_hvg = min(n_top_genes, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg)
    adata = adata[:, adata.var["highly_variable"]]
    print(f"[4] HVGs selected: {adata.n_vars}")

    # ------------------------------------------------------------------ #
    # 5. PCA + kNN graph                                                   #
    # ------------------------------------------------------------------ #
    # sc.pp.scale(adata, max_value=10,zero_center=True)  #Scale
    actual_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=actual_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_pcs)
    print(f"[5] kNN graph built (k={n_neighbors}, pcs={actual_pcs})")

    # ------------------------------------------------------------------ #
    # 6. Extract node features                                             #
    # ------------------------------------------------------------------ #
    X_mat = adata.X
    if hasattr(X_mat, "toarray"):          # sparse → dense
        X_mat = X_mat.toarray()      #.     mat = adata.X.toarray() if issparse(adata.X) else adata.X
    x = torch.tensor(X_mat, dtype=torch.float)

    # ------------------------------------------------------------------ #
    # 7. Extract edges from kNN connectivities (COO format)               #
    # ------------------------------------------------------------------ #
    conn = adata.obsp["connectivities"]   # sparse (N × N)
    cx = conn.tocoo()

    row = torch.tensor(cx.row, dtype=torch.long)
    col = torch.tensor(cx.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)          # (2, E)
    edge_attr  = torch.tensor(cx.data, dtype=torch.float) # (E,)

    # ------------------------------------------------------------------ #
    # 8. Encode labels                                                     #
    # ------------------------------------------------------------------ #
    le = LabelEncoder()
    y_int = le.fit_transform(adata.obs["label"].values)
    y = torch.tensor(y_int, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # 9. Assemble PyG Data object                                          #
    # ------------------------------------------------------------------ #
    n_clusters = len(le.classes_)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.label_encoder = le                # attach for decoding later
    data.num_classes   = len(le.classes_)

    print(f"\n── PyG Graph Summary ──────────────────────")
    print(f"  Nodes (cells) : {data.num_nodes}")
    print(f"  Node features : {data.num_node_features}  (HVGs)")
    print(f"  Edges         : {data.num_edges}")
    print(f"  Classes       : {data.num_classes}  → {list(le.classes_)}")
    print(f"───────────────────────────────────────────")

    return data,n_clusters
