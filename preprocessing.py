import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse
import pyreadr
import h5py
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
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
    file=None,
    normalize: bool = False,
) -> Data:

    # ------------------------------------------------------------------ #
    # 1. Build AnnData                                                     #
    # ------------------------------------------------------------------ #
    if file is not None:
        data1 = sc.read_h5ad(file)
        adata = data1.to_memory()
    else:
        raise FileNotFoundError(" File not found")

    # ------------------------------------------------------------------ #
    # 2. QC filtering                                                      #
    # ------------------------------------------------------------------ #
    sc.pp.filter_cells(adata, min_genes=2000)
    sc.pp.filter_genes(adata, min_cells=800)
    print(f"[2] Post-QC: {adata.n_obs} cells × {adata.n_vars} genes")

    # ------------------------------------------------------------------ #
    # 3. Normalise & log-transform (optional)                             #
    # ------------------------------------------------------------------ #
    if normalize==True:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"[3] Normalization applied: library-size (target=1e4) + log1p")
    else:
        print(f"[3] Normalization skipped: using matrix as-is")

    # ------------------------------------------------------------------ #
    # 4. Highly variable genes                                             #
    # ------------------------------------------------------------------ #
    print(f"ntop genes: {n_top_genes}, n_vars is : {adata.n_vars}")
    actual_hvg = min(n_top_genes, adata.n_vars)
    print(f'actual_hvg: {actual_hvg}')
    sc.pp.highly_variable_genes(adata, n_top_genes=actual_hvg, flavor="seurat_v3")
    print(f"[4] HVGs selected: {adata.var['highly_variable'].sum()}")

    # ------------------------------------------------------------------ #
    # 5. kNN graph for VGAE (no PCA, raw counts)                         #
    # ------------------------------------------------------------------ #
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X", method="gauss",metric="cosine")

    print(f"[5] kNN graph built (k={n_neighbors}, raw features, cosine metric)")

    # ------------------------------------------------------------------ #
    # 6. Extract node features                                             #
    # ------------------------------------------------------------------ #
    X_mat = adata.X
    if hasattr(X_mat, "toarray"):
        X_mat = X_mat.toarray()
    N=X_mat.shape[0]
    x = torch.tensor(X_mat, dtype=torch.float)
    print("Shape:", x.shape)

    # ------------------------------------------------------------------ #
    # 7. Extract edges from kNN connectivities (COO format)               #
    # ------------------------------------------------------------------ #
    conn = adata.obsp["connectivities"]
    cx = conn.tocoo()  # converts dia_matrix → coo_matrix first

    row = torch.tensor(cx.row.copy(), dtype=torch.long)
    col = torch.tensor(cx.col.copy(), dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = torch.tensor(cx.data.copy(), dtype=torch.float)

    # ------------------------------------------------------------------ #
    # 8. Encode labels                                                     #
    # ------------------------------------------------------------------ #
    le = LabelEncoder()
    y_int = le.fit_transform(adata.obs["cell_ontology_class"].values)
    y = torch.tensor(y_int, dtype=torch.long)

    # ------------------------------------------------------------------ #
    # 9. Assemble PyG Data object                                          #
    # ------------------------------------------------------------------ #
    n_clusters = len(le.classes_)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.label_encoder = le
    data.num_classes   = len(le.classes_)

    # ------------------------------------------------------------------ #
    # 10. Build dense adjacency matrix                                     #
    # ------------------------------------------------------------------ #
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), size=(N, N))


    return adj, x, y, n_clusters  # ← FIXED: syntax error on original return


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
