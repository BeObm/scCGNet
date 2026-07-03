import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from typing import Optional,Tuple
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import add_self_loops
import math



# Code below is adapted from https://github.com/nairouz/R-GAE/tree/master/GMM-VGAE here. We thank for the authors to make it publicly available

def parse_index_file(filename):
    """Parse the index file path

    Args:
        filename: The file name of the dataset

    Returns:
        _type_: List of files
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data1(dataset, data_path, modified):
    """Load the data

    Args:
        dataset: Dataset name
        data_path: Dataset file
        modified: if the data is modified for no combination

    Returns:
        _type_: return the adjacency, feature matrix and label
    """
    # load the data: x, tx, allx, graph
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(data_path + "/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(data_path + "/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if modified:
        features = allx
        labels = ally
    else:
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    num_edges = adj.sum()
    print("Dataset has %d nodes, %d edges, with a mean degree of %.2f" % (adj.shape[0], num_edges, adj.sum() / num_edges))
    # Convert labels - handle both one-hot and integer formats
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        # One-hot encoded
        labels_int = np.argmax(labels, 1)
    else:
        # Already integer
        labels_int = labels.flatten()

    return adj, features, labels_int


def load_data(dataset, data_path, n_top_genes, n_neighbors):
    if dataset in ["baron3", "baron4"]:
        adj, features, labels_int= load_data1(dataset, data_path,True)

        return adj, features, labels_int,14
    else:

        adj, x, y, n_clusters = build_pyg_graph(
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           file=f"{data_path}/{dataset}.h5ad",
                                            normalize=False,)
        return adj, x, y, n_clusters


def build_pyg_graph(
        n_top_genes: int = 1200,
        n_neighbors: int = 5,
        file: Optional[str] = None,
        normalize: bool = False,
        label_key = "cell_ontology_class",
        weighted_adj: bool = False,
) -> Tuple[sp.csr_array, sp.csc_matrix, np.ndarray, int]:
    """
    Build graph inputs from a cell x gene AnnData (.h5ad) file.

    Returns
    -------
    adj       : scipy.sparse.csr_array   -- binary (or weighted) kNN adjacency
    features  : scipy.sparse.csc_matrix  -- node feature matrix, HVG-subsetted
    labels    : numpy.ndarray            -- integer-encoded cell type labels
    nClusters : int                      -- number of distinct classes
    """

    # ------------------------------------------------------------------ #
    # 1. Load AnnData
    # ------------------------------------------------------------------ #

    if file is None:
        raise FileNotFoundError("No file path provided.")
    adata = sc.read_h5ad(file)  # loads fully into memory by default

    label_keys = ["cell_ontology_class", "celltype"]
    for lk in label_keys:
        if lk in adata.obs.columns:
            label_key = lk
            break

    # ------------------------------------------------------------------ #
    # 2. QC filtering
    # ------------------------------------------------------------------ #
    sc.pp.filter_cells(adata, min_genes=200)
    print(f"[2] Post-QC: {adata.n_obs} cells x {adata.n_vars} genes")

    if adata.n_obs <= n_neighbors:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be < number of remaining cells "
            f"({adata.n_obs}) after QC filtering."
        )
    if label_key not in adata.obs.columns:
        raise KeyError(
            f"label_key='{label_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )

    # ------------------------------------------------------------------ #
    # 3. Highly variable genes -- computed on RAW counts (seurat_v3 requires
    #    this) and, unlike the original, actually applied to subset adata.
    # ------------------------------------------------------------------ #
    actual_hvg = min(n_top_genes, adata.n_vars)
    print(f"[3] Requested n_top_genes={n_top_genes}, using {actual_hvg} "
          f"(n_vars={adata.n_vars})")
    sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[3] HVG subset applied: {adata.n_vars} genes retained")

    # ------------------------------------------------------------------ #
    # 4. Normalize & log-transform (optional) -- applied AFTER HVG selection
    #    so it doesn't bias the seurat_v3 variance model.
    # ------------------------------------------------------------------ #
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("[4] Normalization applied: library-size (target=1e4) + log1p")
    else:
        print("[4] Normalization skipped: using raw counts as-is")

    # ------------------------------------------------------------------ #
    # 5. kNN graph on the HVG-subsetted feature matrix
    # ------------------------------------------------------------------ #
    sc.pp.neighbors(
        adata, n_neighbors=n_neighbors, use_rep="X", method="gauss", metric="cosine"
    )
    print(f"[5] kNN graph built (k={n_neighbors}, {adata.n_vars}-dim features, "
          f"cosine metric)")

    # ------------------------------------------------------------------ #
    # 6. Node features -- kept sparse (csc), no densify round-trip
    # ------------------------------------------------------------------ #
    X_mat = adata.X
    if not sp.issparse(X_mat):
        X_mat = sp.csc_matrix(X_mat, dtype=np.float32)
    else:
        X_mat = X_mat.astype(np.float32).tocsc()
    N = X_mat.shape[0]
    print(f"[6] Node feature matrix shape: {X_mat.shape}, dtype: {X_mat.dtype}")

    # ------------------------------------------------------------------ #
    # 7. Adjacency from kNN connectivities
    # ------------------------------------------------------------------ #
    conn = adata.obsp["connectivities"]
    cx = conn.tocoo()
    print(f"[7] Number of edges: {cx.nnz}")

    if weighted_adj:
        adj_coo = cx
    else:
        adj_coo = sp.coo_matrix(
            (np.ones(cx.nnz, dtype=np.float32), (cx.row, cx.col)), shape=(N, N)
        )
    adj = sp.csr_array(adj_coo.tocsr())

    # ------------------------------------------------------------------ #
    # 8. Encode labels
    # ------------------------------------------------------------------ #
    le = LabelEncoder()
    labels = le.fit_transform(adata.obs[label_key].values).astype(np.int64)
    nClusters = len(le.classes_)

    return adj, X_mat, labels, nClusters

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple

    Args:
        sparse_mx: Sparse matrix

    Returns:
        _type_: Tuple values
    """
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape



def save_cluster_distribution(y, excel_out):
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


def GraphConstruction(data, features, num_clusters):
    cell_num = features.shape[0]
    average_num = cell_num // num_clusters
    neighbor_num = average_num // 10
    neighbor_num = min(neighbor_num, 15)
    neighbor_num = max(neighbor_num, 5)

    # Calculate Pearson distance matrix
    dis_matrix = squareform(pdist(features, metric='correlation'))
    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric='precomputed').fit(dis_matrix)
    _, indices = nbrs.kneighbors(dis_matrix)  # Get only indices

    # Create adjacency matrix
    n_samples = features.shape[0]
    adj_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in indices[i]:
            adj_matrix[i, j] = 1

            # Create a list of egdes
    edge_list = torch.empty((2, 0), dtype=torch.int64)
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                col = torch.tensor([i, j], dtype=torch.int64)
                edge_list = torch.cat((edge_list, col.unsqueeze(1)), dim=1)
    data.edge_index = edge_list

    # generate a graph
    G = nx.from_numpy_array(adj_matrix)
    print("Building a " + str(G))
    print("===================================================")

    return G


def preprocess_graph(adj):
    """Preprocess the graphs

    Args:
        adj: Adjacency matrix

    Returns:
        _type_: normalized adjacency matrix
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)




def get_device():
    """Get the best available device (CUDA > MPS > CPU)

    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device
