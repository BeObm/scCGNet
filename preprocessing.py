import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected, remove_self_loops
from torch_geometric.transforms import RandomLinkSplit

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


def load_data(dataset, data_path, modified):
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


def preprocess_graph(adj):
    """
    Compute GCN-style normalized adjacency:
        Â = D^{-1/2} (A + I) D^{-1/2}
    and return it as (coords, values, shape) via sparse_to_tuple().

    Parameters
    ----------
    adj : array-like or scipy.sparse.spmatrix
        Square adjacency matrix.

    Returns
    -------
    tuple
        Output of sparse_to_tuple(Â),
    """
    adj = sp.coo_matrix(adj)
    n, m = adj.shape
    if n != m:
        raise ValueError(f"adj must be square, got {adj.shape}")

    adj_ = adj + sp.eye(n, dtype=adj.dtype, format="coo")

    rowsum = np.asarray(adj_.sum(axis=1)).ravel()  # degrees
    # Guard against non-positive degrees (signed/invalid graphs)
    if np.any(rowsum <= 0):
        raise ValueError("Non-positive degree encountered; cannot take d^(-1/2).")

    d_inv_sqrt = np.power(rowsum, -0.5)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format="coo")

    adj_normalized = (D_inv_sqrt @ adj_ @ D_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def get_device():
    """Get the best available device (CUDA > MPS > CPU)

    Returns:
        torch.device: The device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device



def to_pyg_data(adj, features, labels=None, make_undirected=False, add_self_loops=False):
    # --- adjacency -> edge_index / edge_weight ---
    adj = sp.coo_matrix(adj)  # safe for dense or sparse input

    if not add_self_loops:
        # remove diagonal (self-loops) if present
        adj = adj - sp.diags(adj.diagonal(), offsets=0, shape=adj.shape, format="coo")
        adj.eliminate_zeros()
    else:
        # ensure self-loops exist
        adj = (adj + sp.eye(adj.shape[0], format="coo")).tocoo()

    if make_undirected:
        # symmetrize by taking max weight in either direction
        adj_t = adj.transpose().tocoo()
        adj = adj.maximum(adj_t).tocoo()

    # COO gives row, col, data
    row = torch.from_numpy(adj.row.astype(np.int64))
    col = torch.from_numpy(adj.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)  # [2, num_edges]

    edge_weight = torch.from_numpy(adj.data.astype(np.float32))  # [num_edges]

    # --- features -> x ---
    if sp.issparse(features):
        features = features.tocsr()
        x = torch.from_numpy(features.toarray()).float()
    else:
        x = torch.from_numpy(np.asarray(features)).float()

    # --- labels -> y ---
    data_kwargs = {"x": x, "edge_index": edge_index, "edge_weight": edge_weight}
    if labels is not None:
        y = torch.from_numpy(np.asarray(labels)).long()
        data_kwargs["y"] = y

    data = Data(**data_kwargs)
    return data

def build_pyg_data(adj, features, labels=None, make_undirected=True, remove_diag=True):
    adj = sp.coo_matrix(adj)
    n, m = adj.shape
    if n != m:
        raise ValueError(f"adj must be square, got {adj.shape}")

    if remove_diag:
        adj = adj - sp.diags(adj.diagonal(), offsets=0, shape=adj.shape, format="coo")
        adj.eliminate_zeros()

    edge_index, _ = from_scipy_sparse_matrix(adj)

    if make_undirected:
        edge_index = to_undirected(edge_index)

    if sp.issparse(features):
        x = torch.from_numpy(features.toarray()).float()
    else:
        x = torch.from_numpy(np.asarray(features)).float()

    data = Data(x=x, edge_index=edge_index)

    if labels is not None:
        data.y = torch.from_numpy(np.asarray(labels)).long()

    return data

def get_pos_neg_edges(split_data):
    """
    Return (pos_edge_index, neg_edge_index) each with shape [2, E].
    Handles common PyG attribute variants.
    """
    # Newer versions:
    if hasattr(split_data, "pos_edge_label_index") and hasattr(split_data, "neg_edge_label_index"):
        return split_data.pos_edge_label_index, split_data.neg_edge_label_index

    # Older versions:
    if hasattr(split_data, "pos_edge_index") and hasattr(split_data, "neg_edge_index"):
        return split_data.pos_edge_index, split_data.neg_edge_index

    # Unified label format:
    if hasattr(split_data, "edge_label_index") and hasattr(split_data, "edge_label"):
        idx = split_data.edge_label_index
        y = split_data.edge_label
        pos = idx[:, y == 1]
        neg = idx[:, y == 0]
        return pos, neg