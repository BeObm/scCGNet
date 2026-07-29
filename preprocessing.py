from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import anndata as ad
import h5py
# import numpy as np
# import pandas as pd
# import scanpy as sc
# import scipy.sparse as sp
# from sklearn.preprocessing import LabelEncoder
# import torch
# import numpy as np
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
    elif dataset in ["Adam","Bach","Baron_human","Baron_mouse","Campbell","Cao_2020_Spleen","Muraro","Quake_10x_Limb_Muscle_raw",
                     "Quake_Smart-seq2_Diaphragm","Shekhar","Tosches_turtle","Wang_Large_Intestine","Young","zeisel_preprocessed","muraro_preprocessed"]:

        adj, x, y, n_clusters = build_pyg_graph(
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           file=f"{data_path}/{dataset}.h5ad",
                                            normalize=False)
        return adj, x, y, n_clusters
    elif dataset in ["10X_PBMC","human_kidney_counts","Mouse","mouse_ES","Quake_10x_Bladder","Quake_Smart-seq2_Limb_Muscle",
                     "Quake_Smart-seq2_Trachea","worm_neuron","Zeisel"]:
        adj, x, y, n_clusters = build_pyg_graph2(
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           file=f"{data_path}/{dataset}.h5",
                                           normalize=False)
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


__all__ = ["build_pyg_graph2", "NoLabelsError", "inspect_h5", "detect_layout"]


# --------------------------------------------------------------------- #
# Internal policy constants -- not exposed, to keep the signature aligned
# with build_pyg_graph.
# --------------------------------------------------------------------- #
MIN_CELLS = 3          # gene filter; raise to 10 if seurat_v3 still fails
AUTO_USE_RAW = True    # prefer raw.X when X is processed and raw is counts

LABEL_CANDIDATES = [
    "cell_ontology_class",
    "celltype",
    "cell_type",
    "cell_type1",
    "CellType",
    "level1class",       # Zeisel: coarse published annotation
    "level2class",       # Zeisel: fine published annotation
    "Group",
    "label",
    "labels",
    "y",
]

# Outputs of a clustering or classification algorithm, not ground truth.
DERIVED_PATTERNS = [
    "louvain", "leiden", "kmeans", "phenograph", "seurat_cluster",
    "self-project", "self_project", "_round", "_result", "_pred",
    "predicted", "cluster_id", "scanpy",
]


class NoLabelsError(Exception):
    """Raised when a file carries no usable cell-type annotation."""


# --------------------------------------------------------------------- #
# HDF5 helpers
# --------------------------------------------------------------------- #
def _decode(a) -> np.ndarray:
    """Convert HDF5 byte-string arrays to unicode; pass anything else through."""
    a = np.asarray(a)
    if a.dtype.kind in ("S", "O"):
        flat = [x.decode("utf-8", "replace") if isinstance(x, bytes) else x
                for x in a.ravel()]
        return np.array(flat, dtype=object).reshape(a.shape)
    return a


def _sparse_shape(node) -> Optional[tuple]:
    for key in ("shape", "h5sparse_shape"):
        if key in node.attrs:
            return tuple(int(v) for v in np.asarray(node.attrs[key]).ravel())
    if "shape" in node.keys():
        return tuple(int(v) for v in np.asarray(node["shape"][:]).ravel())
    return None


def _read_matrix(node):
    """
    Read a dense dataset or a CSR/CSC-style group into a scipy sparse matrix.

    CSR vs CSC is taken from the 'encoding-type' (AnnData) or
    'h5sparse_format' (h5sparse) attribute. If neither exists the format is
    genuinely unknown; CSR is assumed and a warning printed, because the
    wrong guess garbles the matrix silently rather than raising.
    """
    if isinstance(node, h5py.Group):
        keys = set(node.keys())
        if not {"data", "indices", "indptr"} <= keys:
            raise ValueError(f"Not a sparse group; keys: {sorted(keys)}")
        shape = _sparse_shape(node)
        if shape is None:
            raise ValueError("Sparse group carries no shape attribute/dataset.")
        enc = node.attrs.get("encoding-type",
                             node.attrs.get("h5sparse_format", None))
        if enc is None:
            print("    WARNING: sparse group has no encoding-type attribute; "
                  "assuming CSR.", file=sys.stderr)
            enc = "csr"
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
        cls = sp.csc_matrix if "csc" in enc.lower() else sp.csr_matrix
        return cls((node["data"][:], node["indices"][:], node["indptr"][:]),
                   shape=shape)

    arr = np.asarray(node[:])
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {arr.shape}")
    return sp.csr_matrix(arr)


def _read_frame(node, n_rows: int) -> pd.DataFrame:
    """Build obs/var from a compound dataset or a group of per-column datasets."""
    if isinstance(node, h5py.Dataset):
        arr = node[:]
        if arr.dtype.names:
            df = pd.DataFrame({n: _decode(arr[n]) for n in arr.dtype.names})
            for idx_col in ("_index", "index"):
                if idx_col in df.columns:
                    df = df.set_index(idx_col)
                    df.index.name = None
                    break
            return df
        return pd.DataFrame({"value": _decode(arr)})

    cols = {}
    index = None
    for key in node.keys():
        sub = node[key]
        if isinstance(sub, h5py.Group):
            if {"codes", "categories"} <= set(sub.keys()):
                cats = list(_decode(sub["categories"][:]))
                codes = np.asarray(sub["codes"][:], dtype=int)
                cols[key] = pd.Categorical.from_codes(codes, categories=cats)
            continue
        arr = np.asarray(sub[:])
        if arr.ndim != 1:
            continue
        if key in ("_index", "index"):
            index = _decode(arr)
        elif n_rows == 0 or arr.shape[0] == n_rows:
            cols[key] = _decode(arr)

    df = pd.DataFrame(cols)
    if index is not None and len(index) == len(df):
        df.index = pd.Index(index)
    return df


def _orient(M, n_obs: int, n_var: int):
    """Return M as cells x genes, transposing if stored genes x cells."""
    if M.shape == (n_obs, n_var):
        return M.tocsr()
    if M.shape == (n_var, n_obs):
        print(f"    Matrix stored genes x cells {M.shape}; transposing.")
        return M.T.tocsr()
    raise ValueError(
        f"Matrix shape {M.shape} matches neither (n_obs={n_obs}, "
        f"n_var={n_var}) nor its transpose."
    )


# --------------------------------------------------------------------- #
# Layout detection and inspection
# --------------------------------------------------------------------- #
def detect_layout(path) -> str:
    """Return 'anndata', 'exprs', 'xy' or '10x'. Raises if unrecognized."""
    path = Path(path)
    if path.suffix == ".h5ad":
        return "anndata"

    with h5py.File(path, "r") as f:
        root = set(f.keys())
        if "exprs" in root and {"obs_names", "var_names"} <= root:
            return "exprs"
        if {"X", "obs", "var"} <= root:
            return "anndata"
        if {"X", "Y"} <= root:
            return "xy"
        if "matrix" in root:
            return "10x"
        for k in root:
            node = f[k]
            if isinstance(node, h5py.Group) and \
                    {"data", "indices", "indptr"} <= set(node.keys()):
                return "10x"

    raise ValueError(f"Unrecognized .h5 layout in {path.name}. "
                     f"Root keys: {sorted(root)}")


def inspect_h5(path) -> None:
    """Print the structure of an .h5 file. Run this before anything else."""
    path = Path(path)

    def walk(name, node):
        if isinstance(node, h5py.Dataset):
            names = node.dtype.names
            extra = f" fields={list(names)}" if names else ""
            print(f"  {name:<40} dataset shape={node.shape} "
                  f"dtype={node.dtype}{extra}")
        else:
            attrs = {k: (v.decode() if isinstance(v, bytes) else v)
                     for k, v in dict(node.attrs).items()}
            print(f"  {name:<40} group  attrs={attrs}")

    print(f"\n=== {path.name} ===")
    with h5py.File(path, "r") as f:
        print(f"root keys: {sorted(f.keys())}")
        f.visititems(walk)
    try:
        print(f"detected layout: {detect_layout(path)}")
    except ValueError as exc:
        print(f"detected layout: FAILED -- {exc}")
    print()


def _has_raw(path) -> bool:
    with h5py.File(Path(path), "r") as f:
        return "raw.X" in f.keys() or "raw" in f.keys()


# --------------------------------------------------------------------- #
# Per-layout loaders
# --------------------------------------------------------------------- #
def _load_exprs(path: Path):
    """Matrix under /exprs, names under /obs_names and /var_names."""
    with h5py.File(path, "r") as f:
        obs_names = [str(x) for x in _decode(f["obs_names"][:]).ravel()]
        var_names = [str(x) for x in _decode(f["var_names"][:]).ravel()]
        M = _orient(_read_matrix(f["exprs"]), len(obs_names), len(var_names))
        obs = _read_frame(f["obs"], len(obs_names)) if "obs" in f else pd.DataFrame()
        var = _read_frame(f["var"], len(var_names)) if "var" in f else pd.DataFrame()

    obs = obs if len(obs) == len(obs_names) else pd.DataFrame(index=range(len(obs_names)))
    var = var if len(var) == len(var_names) else pd.DataFrame(index=range(len(var_names)))
    obs.index = pd.Index(obs_names)
    var.index = pd.Index(var_names)

    adata = ad.AnnData(X=M, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    return adata


def _load_xy(path: Path):
    """Bare /X matrix + /Y label vector. No gene or cell identifiers."""
    with h5py.File(path, "r") as f:
        y = _decode(f["Y"][:]).ravel()
        M = _read_matrix(f["X"])

    if M.shape[0] != len(y):
        if M.shape[1] == len(y):
            print(f"    X stored genes x cells {M.shape}; transposing.")
            M = M.T.tocsr()
        else:
            raise ValueError(f"X shape {M.shape} matches len(Y)={len(y)} "
                             f"on neither axis.")
    elif M.shape[0] == M.shape[1]:
        print("    WARNING: X is square; orientation ambiguous, assuming "
              "rows are cells.", file=sys.stderr)

    n_obs, n_var = M.shape
    adata = ad.AnnData(
        X=M.tocsr(),
        obs=pd.DataFrame({"celltype": y},
                         index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_var)]),
    )
    print("    NOTE: gene names are positional placeholders; not comparable "
          "across files.")
    return adata


def _load_legacy_anndata(path: Path, use_raw: bool):
    """Hand-read a legacy h5ad: X/obs/var at root, raw as 'raw.X'/'raw.var'."""
    with h5py.File(path, "r") as f:
        root = set(f.keys())
        x_key = "raw.X" if (use_raw and "raw.X" in root) else "X"
        v_key = "raw.var" if (x_key == "raw.X" and "raw.var" in root) else "var"
        obs = _read_frame(f["obs"], 0)
        var = _read_frame(f[v_key], 0)
        M = _orient(_read_matrix(f[x_key]), len(obs), len(var))

    adata = ad.AnnData(X=M, obs=obs, var=var)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    print(f"    legacy reader used '{x_key}' / '{v_key}'")
    return adata


def _is_counts(X, sample: int = 200_000) -> bool:
    """True if values are non-negative integers, i.e. raw counts."""
    v = X.data if sp.issparse(X) else np.asarray(X).ravel()
    if v.size == 0:
        return False
    if v.size > sample:
        v = v[np.random.default_rng(0).choice(v.size, sample, replace=False)]
    return bool(np.all(v >= 0) and np.allclose(v, np.round(v)))


def _load_h5(file) -> "ad.AnnData":
    """Load any supported .h5 / .h5ad, dispatching on layout."""
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    layout = detect_layout(path)
    print(f"[1] {path.name}: layout='{layout}'")

    if layout == "exprs":
        adata = _load_exprs(path)
    elif layout == "xy":
        adata = _load_xy(path)
    elif layout == "10x":
        adata = sc.read_10x_h5(path)
        adata.var_names_make_unique()
        print("    NOTE: 10x files carry no cell-type annotation.")
    elif layout == "anndata":
        try:
            adata = sc.read_h5ad(path)
        except Exception as exc:
            print(f"    read_h5ad failed ({type(exc).__name__}: {exc}); "
                  f"using the manual legacy reader.", file=sys.stderr)
            adata = _load_legacy_anndata(path, use_raw=False)

        # Prefer raw when X has been processed but raw holds counts --
        # seurat_v3 needs counts, and a raw layer usually means X is not.
        if AUTO_USE_RAW and getattr(adata, "raw", None) is not None:
            if not _is_counts(adata.X) and _is_counts(adata.raw.X):
                print(f"    X is processed; raw holds counts. Switching to "
                      f"raw: {adata.raw.shape[1]} genes (X had {adata.n_vars}).")
                obs = adata.obs.copy()
                adata = ad.AnnData(X=adata.raw.X, obs=obs,
                                   var=adata.raw.var.copy())
                adata.var_names_make_unique()
    else:  # unreachable; detect_layout raises first
        raise ValueError(f"Unhandled layout '{layout}'")

    print(f"[1] Loaded {adata.n_obs} cells x {adata.n_vars} genes")
    return adata


# --------------------------------------------------------------------- #
# Label resolution
# --------------------------------------------------------------------- #
def _is_derived(col: str) -> bool:
    low = str(col).lower()
    return any(pat in low for pat in DERIVED_PATTERNS)


def _suggest_label_columns(adata, max_classes: int = 200):
    """Rank obs columns that plausibly hold ground truth. Best first."""
    out = []
    for col in adata.obs.columns:
        vals = pd.Series(np.asarray(adata.obs[col]))
        n = vals.nunique(dropna=True)
        if n < 2 or n > max_classes or n == len(vals):
            continue
        if pd.api.types.is_numeric_dtype(vals) and n > 20:
            continue
        out.append((col, int(n), _is_derived(col)))
    out.sort(key=lambda t: (t[2], t[1]))
    return out


def _resolve_label_key(adata, path, label_key: Optional[str]) -> str:
    """
    Resolve the ground-truth column or raise NoLabelsError.

    An explicit label_key wins if it exists in obs. The default value from
    the signature is treated as a hint, not a demand, so the function works
    unchanged on files using other conventions.
    """
    if label_key and label_key in adata.obs.columns:
        if _is_derived(label_key):
            print(f"    WARNING: '{label_key}' looks like an algorithm output, "
                  f"not ground truth. Metrics against it measure agreement "
                  f"with that method, not accuracy.", file=sys.stderr)
        return label_key

    found = []
    for cand in LABEL_CANDIDATES:
        if cand not in adata.obs.columns or _is_derived(cand):
            continue
        vals = pd.Series(np.asarray(adata.obs[cand]))
        n = vals.nunique(dropna=True)
        if n < 2 or n == len(vals):
            continue
        found.append((cand, int(n)))

    if found:
        if len(found) > 1:
            listing = ", ".join(f"'{c}' ({n})" for c, n in found)
            print(f"    Multiple annotations available: {listing}. "
                  f"Using '{found[0][0]}'; pass label_key= to override.")
        return found[0][0]

    sugg = _suggest_label_columns(adata)
    real = [(c, n) for c, n, d in sugg if not d]
    derived = [c for c, _, d in sugg if d]
    msg = [f"{Path(path).name}: no known label column found."]
    if real:
        msg.append("Plausible ground-truth columns (name, #classes): "
                   + ", ".join(f"{c} ({n})" for c, n in real[:8])
                   + ". Pass label_key='<name>'.")
    if derived:
        msg.append("Ignored as algorithm output: "
                   + ", ".join(derived[:8]) + ".")
    if not real:
        msg.append("Nothing in obs looks like a ground-truth annotation, so "
                   "this file cannot support clustering metrics.")
    raise NoLabelsError(" ".join(msg))


# --------------------------------------------------------------------- #
# HVG selection
# --------------------------------------------------------------------- #
def _select_hvg(adata, n_top: int, flavor: str) -> str:
    """
    Flag highly variable genes, degrading gracefully if the flavor fails.

    seurat_v3 fits a degree-2 loess of log10(var) on log10(mean). Genes
    detected in only a few cells give near-tied log10(mean) values, the
    local design matrix becomes singular, and skmisc raises
    ValueError('reciprocal condition number ...'). Gene filtering usually
    prevents this; these fallbacks cover what it doesn't.

    Returns the flavor actually used.
    """
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=flavor)
        return flavor
    except Exception as exc:
        print(f"    flavor='{flavor}' failed ({type(exc).__name__}: {exc}); "
              f"falling back to dispersion-based 'seurat'.", file=sys.stderr)

    try:
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        sc.pp.highly_variable_genes(tmp, n_top_genes=n_top, flavor="seurat")
        adata.var["highly_variable"] = tmp.var["highly_variable"].values
        del tmp
        return "seurat (fallback)"
    except Exception as exc:
        print(f"    'seurat' also failed ({type(exc).__name__}: {exc}); "
              f"falling back to variance ranking.", file=sys.stderr)

    X = adata.X.tocsr() if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    lib = np.asarray(X.sum(axis=1)).ravel()
    lib[lib == 0] = 1.0
    Xn = sp.diags(1e4 / lib) @ X
    Xn.data = np.log1p(Xn.data)
    mean = np.asarray(Xn.mean(axis=0)).ravel()
    var = np.asarray(Xn.multiply(Xn).mean(axis=0)).ravel() - mean ** 2
    keep = np.zeros(adata.n_vars, dtype=bool)
    keep[np.argsort(var)[::-1][:n_top]] = True
    adata.var["highly_variable"] = keep
    return "variance ranking (fallback)"


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #
def build_pyg_graph2(
        n_top_genes: int = 1200,
        n_neighbors: int = 5,
        file: Optional[str] = None,
        normalize: bool = False,
        label_key="cell_ontology_class",
        weighted_adj: bool = False,
) -> Tuple[sp.csr_array, sp.csc_matrix, np.ndarray, int]:
    """
    Build graph inputs from a cell x gene .h5 file.

    Same signature and same return type as build_pyg_graph, which handles
    .h5ad. label_key is a hint: if the named column is absent, the label is
    auto-detected from LABEL_CANDIDATES, skipping algorithm-output columns.

    Returns
    -------
    adj       : scipy.sparse.csr_array   -- binary (or weighted) kNN adjacency
    features  : scipy.sparse.csc_matrix  -- node feature matrix, HVG-subsetted
    labels    : numpy.ndarray            -- integer-encoded cell type labels
    nClusters : int                      -- number of distinct classes

    Raises
    ------
    NoLabelsError  if the file carries no ground-truth annotation.
    """
    # ------------------------------------------------------------------ #
    # 1. Load and resolve labels (fails fast, before any computation)
    # ------------------------------------------------------------------ #
    if file is None:
        raise FileNotFoundError("No file path provided.")

    adata = _load_h5(file)
    label_key = _resolve_label_key(adata, file, label_key)
    n_classes = int(pd.Series(np.asarray(adata.obs[label_key])).nunique())
    print(f"[1] Label column '{label_key}': {n_classes} classes")

    # ------------------------------------------------------------------ #
    # 2. Cell filtering -- DELIBERATELY NOT PERFORMED. Every cell reaches
    #    the graph, so the class distribution matches the source file and
    #    metrics stay comparable to published numbers on the same data.
    # ------------------------------------------------------------------ #
    print(f"[2] Cell filtering skipped: all {adata.n_obs} cells retained")

    if adata.n_obs <= n_neighbors:
        raise ValueError(
            f"n_neighbors={n_neighbors} must be < number of cells "
            f"({adata.n_obs})."
        )

    # ------------------------------------------------------------------ #
    # 2b. Gene filtering -- required. Genes detected in zero or one cell
    #     are what make the seurat_v3 loess singular.
    # ------------------------------------------------------------------ #
    n_genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
    print(f"[2b] Gene filter (min_cells={MIN_CELLS}): {n_genes_before} -> "
          f"{adata.n_vars} genes ({n_genes_before - adata.n_vars} dropped)")

    if adata.n_vars == 0:
        raise ValueError(
            f"min_cells={MIN_CELLS} removed every gene; the matrix is too "
            f"sparse. Lower h5_graph.MIN_CELLS."
        )

    # ------------------------------------------------------------------ #
    # 3. Highly variable genes -- flavor chosen from the data. seurat_v3
    #    requires raw counts; seurat expects transformed values.
    # ------------------------------------------------------------------ #
    is_counts = _is_counts(adata.X)
    flavor = "seurat_v3" if is_counts else "seurat"
    if not is_counts:
        print("[3] Values are non-integer -- matrix already "
              "normalized/transformed; using flavor='seurat'.")

    actual_hvg = min(n_top_genes, adata.n_vars)
    print(f"[3] Requested n_top_genes={n_top_genes}, using {actual_hvg} "
          f"(n_vars={adata.n_vars}), flavor='{flavor}'")
    used = _select_hvg(adata, actual_hvg, flavor)
    if used != flavor:
        print(f"[3] HVG selection actually used: {used}")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[3] HVG subset applied: {adata.n_vars} genes retained")

    # ------------------------------------------------------------------ #
    # 4. Normalize & log-transform (optional), AFTER HVG selection so it
    #    doesn't bias the variance model. Skipped when the input is already
    #    transformed, to avoid double-transforming.
    # ------------------------------------------------------------------ #
    if normalize and is_counts:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print("[4] Normalization applied: library-size (target=1e4) + log1p")
    elif normalize:
        print("[4] Normalization requested but input is not raw counts -- "
              "skipped to avoid double-transforming.")
    else:
        print("[4] Normalization skipped: using values as-is")

    # ------------------------------------------------------------------ #
    # 5. kNN graph on the HVG-subsetted feature matrix
    # ------------------------------------------------------------------ #
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X",
                    method="gauss", metric="cosine")
    print(f"[5] kNN graph built (k={n_neighbors}, {adata.n_vars}-dim "
          f"features, cosine metric)")

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
    cx = adata.obsp["connectivities"].tocoo()
    print(f"[7] Number of edges: {cx.nnz}")

    if weighted_adj:
        adj_coo = cx
    else:
        adj_coo = sp.coo_matrix(
            (np.ones(cx.nnz, dtype=np.float32), (cx.row, cx.col)),
            shape=(N, N),
        )
    adj = sp.csr_array(adj_coo.tocsr())

    # ------------------------------------------------------------------ #
    # 8. Encode labels
    # ------------------------------------------------------------------ #
    le = LabelEncoder()
    labels = le.fit_transform(np.asarray(adata.obs[label_key].values)
                              ).astype(np.int64)
    nClusters = len(le.classes_)
    print(f"[8] {nClusters} classes: {list(le.classes_)[:10]}"
          f"{' ...' if nClusters > 10 else ''}")

    return adj, X_mat, labels, nClusters


# --------------------------------------------------------------------- #
# Dispatcher -- optional convenience if you'd rather not branch by hand
# --------------------------------------------------------------------- #
def build_graph_any(file: str, **kwargs):
    """Route to build_pyg_graph (.h5ad) or build_pyg_graph2 (.h5) by suffix."""
    suffix = Path(file).suffix
    if suffix == ".h5ad":
        from __main__ import build_pyg_graph   # your existing function
        return build_pyg_graph(file=file, **kwargs)
    if suffix == ".h5":
        return build_pyg_graph2(file=file, **kwargs)
    raise ValueError(f"Unsupported extension '{suffix}' (expect .h5 / .h5ad)")