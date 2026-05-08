import torch
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse
import h5py
import anndata as ad
import scanpy as sc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder


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


def load_data(dataset, data_path, n_top_genes, n_neighbors, n_pcs):
    if dataset in ["baron3", "baron4"]:
        adj, features, labels_int= load_data1(dataset, data_path,True)

        return adj, features, labels_int,14
    if dataset in ["Klein", "Chung", "YAN"]:
        X, y, n_clusters = read_tsv(f"{data_path}/data.tsv",
                                    f"{data_path}/label.ann",
                                    f"{data_path}/cluster_distribution.xlsx")
        adj, x, y, n_clusters = build_pyg_graph(cell_gene_matrix=X, cell_labels=y,
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           n_pcs=n_pcs)
        return adj, x, y, n_clusters
    elif dataset in ["facs_lung", "droplet_lung"]:
        X, y, n_clusters = read_rds(f"{data_path}/{dataset}_norm.rds",
                                    f"{data_path}/{dataset}_meta.rds",
                                    f"{data_path}/cluster_distribution.xlsx")
        adj, x, y, n_clusters = build_pyg_graph(cell_gene_matrix=X, cell_labels=y,
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           n_pcs=n_pcs,
                                           file=None)
        return adj, x, y, n_clusters

    elif dataset in ["10X_PMBC", 'lps_int2', "human_kidney", "Muraro", "Mouse", "mouse_ES", "worm_neuron",
                     "Quake_10x_Bladder", "Quake_Smart-seq2_Limb_Muscle", "Quake_Smart-seq2_Trachea",
                     "Quake_10x_Limb_Muscle", "Quake_10x_Spleen", "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Lung",
                     "Romanov"]:  # These dataset have raw count data
        adj, x, y, n_clusters = build_pyg_graph(cell_gene_matrix=None, cell_labels=None,
                                           n_top_genes=n_top_genes,
                                           n_neighbors=n_neighbors,
                                           n_pcs=n_pcs,
                                           file=f"{data_path}/{dataset}.h5")
        return adj, x, y, n_clusters
    elif dataset in ["Campell"]:
        x, y = load_campell(data_path)
        adj, x, y, n_clusters = build_pyg_graph(cell_gene_matrix=x, cell_labels=y)
        return adj, x, y, n_clusters
    elif dataset in ["baron_mouse", "biase", "darmanis", "deng", "goolam", "romanov", "zeisel"]:
        x, y = read_csv_file(f"{data_path}/{dataset}.csv")
        adj, x, y, n_clusters = build_pyg_graph(cell_gene_matrix=x, cell_labels=y)
        return adj, x, y, n_clusters
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))



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
    cx = conn.tocoo()  # converts dia_matrix → coo_matrix first

    row = torch.tensor(cx.row.copy(), dtype=torch.long)
    col = torch.tensor(cx.col.copy(), dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = torch.tensor(cx.data.copy(), dtype=torch.float)

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

    # Convert sparse connectivity matrix directly to dense numpy, then tensor
    adj = adata.obsp["connectivities"].toarray()  # keep as numpy here

    # Remove self-loops (this is the line that was failing)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape).toarray()

    # Now convert to tensor
    adj = torch.tensor(adj, dtype=torch.float)
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

    return torch.device("cpu")
