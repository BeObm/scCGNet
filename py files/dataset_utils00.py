import pyreadr
import numpy as np
import pandas as pd
import h5py
import anndata as ad
import scanpy as sc
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse



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


def load_campell(data_path):
    label_data=pd.read_csv(f"{data_path}/GSE93374_cell_metadata.txt", sep="\t")
    data=pd.read_csv(f"{data_path}/GSE93374_Merged_all_020816_DGE.txt", sep="\t")

    label_name=[]
    labels=[]
    for i,col  in enumerate(data.columns):
        label_name.append(label_data.loc[label_data["1.ID"] == col,"1.ID"].iloc[0])
        labels.append(label_data.loc[label_data["1.ID"] == col,"2.group"].iloc[0])

    if set(data.columns)== set(label_name):
        for i,col  in enumerate(data.columns):
            if col != label_name[i]:
                raise ValueError("Mismatch in cell ID. Check file structure")
    elif set(data.columns) != set(label_name):
         raise ValueError("Mismatch in cell ID")

    X = data.T
    return X, labels

def build_pyg_graph1(
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



def extract_graph_components(data):
    N = data.num_nodes

    row     = data.edge_index[0].cpu().numpy()
    col     = data.edge_index[1].cpu().numpy()
    weights = (data.edge_attr.cpu().numpy()
               if data.edge_attr is not None
               else np.ones(len(row), dtype=np.float32))

    # Return sparse instead of dense — supports eliminate_zeros() and saves memory
    A = csr_matrix((weights, (row, col)), shape=(N, N))

    X = data.x.cpu().numpy()
    y = data.y.cpu().numpy()

    print(f"── Graph Components ────────────────────────")
    print(f"  A  (adjacency) : {A.shape}  nnz={A.nnz}")
    print(f"  X  (features)  : {X.shape}")
    print(f"  y  (labels)    : {y.shape}  classes={np.unique(y).tolist()}")
    print(f"────────────────────────────────────────────")

    return A, X, y

def read_csv_file(data_path):
    import pandas as pd

    # Load file
    df = pd.read_csv(data_path)

    # Extract cluster labels (target)
    y = df.iloc[1, 1:].astype(int).values

    # Extract gene expression matrix
    gene_expression = df.iloc[2:, 1:]

    # Gene names
    gene_names = df.iloc[2:, 0].values

    # Cell names
    cell_names = df.columns[1:]

    # Convert to numeric matrix
    X = gene_expression.astype(float).values

    # Transpose so shape = (cells, genes)
    X = X.T
    return X,y

def build_pyg_graph(
    cell_gene_matrix=None,
    cell_labels=None,
    n_top_genes: int = 2000,
    n_neighbors: int = 15,
    n_pcs: int = 50,
    file=None,
    normalize: bool = True,
) -> Data:

    # ------------------------------------------------------------------ #
    # 1. Build AnnData                                                     #
    # ------------------------------------------------------------------ #
    if file is not None:
        f = h5py.File(file, "r")
        X = f["X"][:]
        y = f["Y"][:]
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
    # 3. Normalise & log-transform (optional)                             #
    # ------------------------------------------------------------------ #
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"[3] Normalization applied: library-size (target=1e4) + log1p")
    else:
        print(f"[3] Normalization skipped: using matrix as-is")

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
    actual_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.pp.pca(adata, n_comps=actual_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_pcs)
    print(f"[5] kNN graph built (k={n_neighbors}, pcs={actual_pcs})")

    # ------------------------------------------------------------------ #
    # 6. Extract node features                                             #
    # ------------------------------------------------------------------ #
    X_mat = adata.X
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    x = torch.tensor(X_mat, dtype=torch.float)

    # ------------------------------------------------------------------ #
    # 7. Extract edges from kNN connectivities (COO format)               #
    # ------------------------------------------------------------------ #
    conn = adata.obsp["connectivities"]
    cx = conn.tocoo()

    row = torch.tensor(cx.row, dtype=torch.long)
    col = torch.tensor(cx.col, dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr  = torch.tensor(cx.data, dtype=torch.float)

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
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)  # ← FIXED: uncommented
    data.label_encoder = le
    data.num_classes   = len(le.classes_)

    # ------------------------------------------------------------------ #
    # 10. Build dense adjacency matrix                                     #
    # ------------------------------------------------------------------ #
    n_nodes = data.num_nodes
    adj = torch.zeros((n_nodes, n_nodes), dtype=torch.float)
    adj[row, col] = edge_attr  # weighted; use 1.0 for binary: adj[row, col] = 1.0

    print(f"\n── PyG Graph Summary ──────────────────────")
    print(f"  Nodes (cells)  : {data.num_nodes}")
    print(f"  Node features  : {data.num_node_features}  (HVGs)")
    print(f"  Edges          : {data.num_edges}")
    print(f"  Classes        : {data.num_classes}  → {list(le.classes_)}")
    print(f"  Normalized     : {normalize}")
    print(f"  Adjacency mat  : {adj.shape}")
    print(f"───────────────────────────────────────────")

    return adj, x, y, n_clusters  # ← FIXED: syntax error on original return


# def build_pyg_graph(
#     cell_gene_matrix=None,
#     cell_labels=None,
#     n_top_genes: int = 2000,
#     n_neighbors: int = 15,
#     n_pcs: int = 50,
#     file=None,
#     normalize: bool = True ,        # Toggle library-size normalization + log1p
# ) -> Data:
#     """
#     Build a PyTorch Geometric graph from a cell x gene matrix.
#
#     Pipeline:
#         Raw counts → QC → [Log1p] → Top-k HVGs → PCA → kNN Graph → PyG Data
#
#     Args:
#         cell_gene_matrix: Raw count matrix (cells × genes).
#         cell_labels:      Cell type / condition labels per cell.
#         n_top_genes:      Number of highly variable genes to keep.
#         n_neighbors:      k for kNN graph construction.
#         n_pcs:            Number of PCs used for neighbor computation.
#         file:             Path to an HDF5 file containing X and Y datasets.
#         normalize:        If True, applies library-size normalization (target
#                           sum = 1e4) followed by log1p transformation.
#                           Set to False if the matrix is already normalized
#                           or you want to work with raw counts.
#
#     Returns:
#         torch_geometric.data.Data with:
#             - x:          Node features (cells × HVGs), optionally log-normalised
#             - edge_index: COO edge list from kNN graph  (2 × E)
#             - edge_attr:  Connectivities / weights       (E,)
#             - y:          Integer-encoded cell labels    (N,)
#     """
#
#     # ------------------------------------------------------------------ #
#     # 1. Build AnnData                                                     #
#     # ------------------------------------------------------------------ #
#     if file is not None:
#         f= h5py.File(file, "r")
#         f.keys()
#         X = f["X"][:]
#         y = f["Y"][:]
#
#         adata = ad.AnnData(X)
#         adata.obs["label"] = y
#
#     else:
#         if isinstance(cell_gene_matrix, pd.DataFrame):
#             adata = sc.AnnData(X=cell_gene_matrix.values.astype(np.float32))
#             adata.var_names = cell_gene_matrix.columns.astype(str)
#             adata.obs_names = cell_gene_matrix.index.astype(str)
#         else:
#             adata = sc.AnnData(X=cell_gene_matrix.astype(np.float32))
#
#         adata.obs["label"] = np.array(cell_labels)
#         print(f"[1] Input:  {adata.n_obs} cells × {adata.n_vars} genes")
#
#     # ------------------------------------------------------------------ #
#     # 2. QC filtering                                                      #
#     # ------------------------------------------------------------------ #
#     sc.pp.filter_cells(adata, min_genes=200)
#     sc.pp.filter_genes(adata, min_cells=3)
#     print(f"[2] Post-QC: {adata.n_obs} cells × {adata.n_vars} genes")
#
#     # ------------------------------------------------------------------ #
#     # 3. Normalise & log-transform (optional)                             #
#     # ------------------------------------------------------------------ #
#     if normalize:
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         print(f"[3] Normalization applied: library-size (target=1e4) + log1p")
#     else:
#         print(f"[3] Normalization skipped: using matrix as-is")
#
#     # ------------------------------------------------------------------ #
#     # 4. Highly variable genes                                             #
#     # ------------------------------------------------------------------ #
#     actual_hvg = min(n_top_genes, adata.n_vars)
#     sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg)
#     adata = adata[:, adata.var["highly_variable"]]
#     print(f"[4] HVGs selected: {adata.n_vars}")
#
#     # ------------------------------------------------------------------ #
#     # 5. PCA + kNN graph                                                   #
#     # ------------------------------------------------------------------ #
#     actual_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
#     sc.pp.pca(adata, n_comps=actual_pcs)
#     sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_pcs)
#     print(f"[5] kNN graph built (k={n_neighbors}, pcs={actual_pcs})")
#
#     # ------------------------------------------------------------------ #
#     # 6. Extract node features                                             #
#     # ------------------------------------------------------------------ #
#     X_mat = adata.X
#     if hasattr(X_mat, "toarray"):
#         X_mat = X_mat.toarray()
#     x = torch.tensor(X_mat, dtype=torch.float)
#
#     # ------------------------------------------------------------------ #
#     # 7. Extract edges from kNN connectivities (COO format)               #
#     # ------------------------------------------------------------------ #
#     conn = adata.obsp["connectivities"]
#     cx = conn.tocoo()
#
#     row = torch.tensor(cx.row, dtype=torch.long)
#     col = torch.tensor(cx.col, dtype=torch.long)
#     edge_index = torch.stack([row, col], dim=0)
#     edge_attr  = torch.tensor(cx.data, dtype=torch.float)
#
#     # ------------------------------------------------------------------ #
#     # 8. Encode labels                                                     #
#     # ------------------------------------------------------------------ #
#     le = LabelEncoder()
#     y_int = le.fit_transform(adata.obs["label"].values)
#     y = torch.tensor(y_int, dtype=torch.long)
#
#     # ------------------------------------------------------------------ #
#     # 9. Assemble PyG Data object                                          #
#     # ------------------------------------------------------------------ #
#     n_clusters = len(le.classes_)
#     # data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
#     # data.label_encoder = le
#     # data.num_classes   = len(le.classes_)
#
#     print(f"\n── PyG Graph Summary ──────────────────────")
#     print(f"  Nodes (cells)  : {data.num_nodes}")
#     print(f"  Node features  : {data.num_node_features}  (HVGs)")
#     print(f"  Edges          : {data.num_edges}")
#     print(f"  Classes        : {data.num_classes}  → {list(le.classes_)}")
#     print(f"  Normalized     : {normalize}")
#     print(f"───────────────────────────────────────────")
#
#     return data,x,y n_clusters

def load_h5_data(dataPath, hvg=3000, n_neighbors=15, ts=None, metric='cosine'):
    adata = ad.read(dataPath)
    sc.pp.filter_cells(adata, min_genes=1)
    sc.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    adata.X = adata.X.astype(np.float32)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=hvg)
    adata.raw.var['highly_variable'] = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]

    rawData = adata.raw[:, adata.raw.var['highly_variable']].X
    adj, r_adj = adata_knn(adata, method='gauss', knn=True,
                           n_neighbors=n_neighbors, metric=metric)
    adj = adj.toarray()
    adj[adj > 0] = int(1)
    edge_index = np.where(adj > 0)
    edge_index = np.concatenate((np.expand_dims(edge_index[0], axis=0), np.expand_dims(edge_index[1], axis=0)), axis=0)
    adj_gene = kneighbors_graph(rawData.T, n_neighbors=5, mode='connectivity', include_self=True)
    if issparse(adj_gene):
        adj_gene = adj_gene.toarray()
    edge_index_g = np.where(adj_gene > 0)
    edge_index_g = np.concatenate((np.expand_dims(edge_index_g[0], axis=0), np.expand_dims(edge_index_g[1], axis=0)), axis=0)

    feature = rawData
    if 'cell_type1' in adata.obs.keys():
        celltype = adata.obs['cell_type1']
        label = celltype.values.codes
        feature = feature.toarray()
    else:
        celltype = adata.obs['celltype']
        label = celltype.values
    features = get_feat_input(feature, adj, ts)
    n_classes =len(np.unique(celltype))

    dataDict = {}
    dataDict['features'] = features
    dataDict['adj'] = adj
    dataDict['label'] = label
    dataDict['n_classes'] = n_classes
    dataDict['edge_index'] = edge_index
    dataDict['edge_index_g'] = edge_index_g

    return dataDict, adata



def eliminate_self_loops(A):
    """Remove self-loops from the sparse adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def filter_noise(feature, adj, times, renorm=True):
    if times == 0:
        return feature
    else:
        adj = sp.coo_matrix(adj)
        eye_ = sp.eye(adj.shape[0])
        adj_ = adj if not renorm else adj + eye_
        row_sum = np.array(adj_.sum(1))
        D_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
        adj_n = D_inv_sqrt.dot(adj_).dot(D_inv_sqrt).tocoo()
        m_ = eye_ - adj_n
        feat_ = feature
        for i in range(times):
            h_ = eye_ - m_
            feat_ = h_.dot(feat_)
        feat_ = sp.csr_matrix(feat_).toarray()
        return feat_




def get_feat_input(feat, adj, ts):
    adj_ = eliminate_self_loops(csr_matrix(adj))
    features = []
    if ts is None:
        ts = [0, 0]
    for t in ts:
        features.append(filter_noise(feat, adj_, t))
    return features

def adata_knn(adata, method, knn, n_neighbors=2, metric='cosine'):
    if adata.shape[0] >= 10000:
        sc.pp.pca(adata, n_comps=50)
        n_pcs = 50
    else:
        n_pcs = 0
    if method == 'umap':
        sc.pp.neighbors(adata, method=method, metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        sc.pp.neighbors(adata, method='gauss', metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']

    return adj, r_adj



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
