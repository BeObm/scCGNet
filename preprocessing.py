
import pickle as pkl
import networkx as nx
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected
import torch
import scipy.sparse as sp
from torch.utils.data import Dataset
import os
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from dataset_utils import *
import numpy as np
from munkres import Munkres
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


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


def load_data(dataset, data_path,n_top_genes,n_neighbors,n_pcs):
    if dataset in ["baron3","baron4"]:
        adj, features, labels_int = load_data1(dataset, data_path)
        data, n_clusters = build_pyg_data(adj, features, labels_int)
        return data, n_clusters
    if dataset in ["Klein", "Chung","YAN"]:
        X, y, n_clusters = read_tsv(f"{data_path}/data.tsv",
                                    f"{data_path}/label.ann",
                                    f"{data_path}/cluster_distribution.xlsx")
        data, n_clusters = build_pyg_graph(cell_gene_matrix=X, cell_labels= y,
                               n_top_genes=n_top_genes,
                               n_neighbors=n_neighbors,
                               n_pcs=n_pcs)
        return data, n_clusters
    elif dataset in ["facs_lung", "droplet_lung"]:
        X,y,n_clusters = read_rds(f"{data_path}/{dataset}_norm.rds",
                      f"{data_path}/{dataset}_meta.rds",
                                  f"{data_path}/cluster_distribution.xlsx")
        data,n_clusters = build_pyg_graph(cell_gene_matrix=X, cell_labels= y,
                               n_top_genes=n_top_genes,
                               n_neighbors=n_neighbors,
                               n_pcs=n_pcs,
                               file=None)
        return data,n_clusters

    elif dataset in ["10X_PMBC", 'lps_int2',"human_kidney","Muraro","Mouse","mouse_ES","worm_neuron","Quake_10x_Bladder", "Quake_Smart-seq2_Limb_Muscle","Quake_Smart-seq2_Trachea","Quake_10x_Limb_Muscle","Quake_10x_Spleen","Quake_Smart-seq2_Diaphragm","Quake_Smart-seq2_Lung","Romanov"]:   #  These dataset have raw count data
        data,n_clusters = build_pyg_graph(cell_gene_matrix=None, cell_labels=None,
                               n_top_genes=n_top_genes,
                               n_neighbors=n_neighbors,
                               n_pcs=n_pcs,
                               file=f"{data_path}/{dataset}.h5")
        return data, n_clusters
    elif dataset in ["Campell"]:
        x,y=load_campell(data_path)
        data, n_clusters = build_pyg_graph(cell_gene_matrix=x, cell_labels=y)
        return data, n_clusters
    elif dataset in ["baron_mouse","biase","darmanis","deng","goolam","romanov","zeisel"]:
        x,y = read_csv_file(f"{data_path}/{dataset}.csv")
        data, n_clusters = build_pyg_graph(cell_gene_matrix=x, cell_labels=y)
        return data, n_clusters
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))

def load_data1(dataset, data_path, modified=True):
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
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return torch.device("cpu") # Just


def clustering_metrics(y_true, y_pred):
    """
    Compute ACC, ARI, NMI for clustering.

    Args:
        y_true: array-like, shape (N,)
        y_pred: array-like, shape (N,)

    Returns:
        acc, ari, nmi
    """

    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    assert y_true.shape == y_pred.shape

    # --- ACC (with Hungarian matching) ---
    def compute_acc(y_true, y_pred):
        y_true = y_true - np.min(y_true)
        l1 = list(set(y_true))
        numclass1 = len(l1)
        l2 = list(set(y_pred))
        numclass2 = len(l2)
        ind = 0
        if numclass1 != numclass2:
            for i in l1:
                if i in l2:
                    pass
                else:
                    y_pred[ind] = i
                    ind += 1
        l2 = list(set(y_pred))
        numclass2 = len(l2)

        if numclass1 != numclass2:
            print('error')
            return
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
                cost[i][j] = len(mps_d)
        m = Munkres()
        cost = cost.__neg__().tolist()
        indexes = m.compute(cost)

        new_predict = np.zeros(len(y_pred))
        for i, c in enumerate(l1):
            c2 = l2[indexes[i][1]]

            ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(y_true, new_predict)
        return acc

    acc = compute_acc(y_true, y_pred)

    # --- ARI ---
    ari = adjusted_rand_score(y_true, y_pred)

    # --- NMI ---
    nmi = normalized_mutual_info_score(y_true, y_pred)

    return acc, ari, nmi

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
    num_classes=0
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
        num_classes = int(data.y.unique().numel())


    return data,num_classes

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


class load_data_origin_data1(Dataset):
    def __init__(self, dataset, dataset1, load_type="csv", take_log=False, scaling=False):
        def load_txt():
            self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
            self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

        def load_h5():
            data_mat = h5py.File(dataset)
            self.x = np.array(data_mat['X'])
            self.y = np.array(data_mat['Y'])

        def load_csv():
            pre_process_paras = {'take_log': take_log, 'scaling': scaling}
            self.pre_process_paras = pre_process_paras
            print(pre_process_paras)
            dataset_list = pre_processing_single1(dataset, dataset1, pre_process_paras, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            # self.y = dataset_list[0]['cell_labels'].astype(np.str)
            self.y = dataset_list[0]['cluster_labels'].astype(np.int32)

        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), \
            torch.from_numpy(np.array(self.y[idx])), \
            torch.from_numpy(np.array(idx))


# Copyright 2017 Goekcen Eraslan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


def read_dataset(adata, transpose=False, test_split=False, copy=False):
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error

    # if adata.X.size < 50e6: # check if adata.X is integer only if array is small
    #     if sp.sparse.issparse(adata.X):
    #         assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
    #     else:
    #         assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
    else:
        adata.obs['DCA_split'] = 'train'

    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, adata.n_obs))

    return adata


def process_normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)
        # sc.pp.filter_cells(adata, min_counts=1)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata


def normalization_for_NE(features_):
    features = features_.copy()
    for i in range(len(features)):
        features[i] = features[i] / sum(features[i]) * 1000000
    features = np.log2(features + 1)
    return features


def NE_dn(w, N, eps):
    w = w * N
    D = np.sum(np.abs(w), axis=1) + eps
    D = 1 / D
    D = np.diag(D)
    wn = np.dot(D, w)
    return wn


def dominateset(aff_matrix, NR_OF_KNN):
    thres = np.sort(aff_matrix)[:, -NR_OF_KNN]
    aff_matrix.T[aff_matrix.T < thres] = 0
    aff_matrix = (aff_matrix + aff_matrix.T) / 2
    return aff_matrix


def TransitionFields(W, N, eps):
    W = W * N
    W = NE_dn(W, N, eps)
    w = np.sqrt(np.sum(np.abs(W), axis=0) + eps)
    W = W / np.expand_dims(w, 0).repeat(N, 0)
    W = np.dot(W, W.T)
    return W


def getNeMatrix(W_in):
    N = len(W_in)

    K = min(20, N // 10)
    alpha = 0.9
    order = 3
    eps = 1e-20

    W0 = W_in * (1 - np.eye(N))
    W = NE_dn(W0, N, eps)
    W = (W + W.T) / 2

    DD = np.sum(np.abs(W0), axis=0)

    P = (dominateset(np.abs(W), min(K, N - 1))) * np.sign(W)
    P = P + np.eye(N) + np.diag(np.sum(np.abs(P.T), axis=0))

    P = TransitionFields(P, N, eps)

    D, U = np.linalg.eig(P)
    d = D - eps
    d = (1 - alpha) * d / (1 - alpha * d ** order)
    D = np.diag(d)
    W = np.dot(np.dot(U, D), U.T)
    W = (W * (1 - np.eye(N))) / (1 - np.diag(W))
    W = W.T

    D = np.diag(DD)
    W = np.dot(D, W)
    W[W < 0] = 0
    W = (W + W.T) / 2

    return W


def getGraph(dataset_str, features, L, K, method):
    print(method)

    if method == 'pearson':
        co_matrix = np.corrcoef(features)
    elif method == 'spearman':
        co_matrix, _ = spearmanr(features.T)
    elif method == 'NE':
        co_matrix = np.corrcoef(features)
        os.makedirs('result', exist_ok=True)
        NE_path = 'result/NE_' + dataset_str + '.csv'
        # os.remove(NE_path)
        if os.path.exists(NE_path):
            NE_matrix = pd.read_csv(NE_path).values
        else:
            features = normalization_for_NE(features)
            in_matrix = np.corrcoef(features)
            NE_matrix = getNeMatrix(in_matrix)
            pd.DataFrame(NE_matrix).to_csv(NE_path, index=False)

        N = len(co_matrix)
        sim_sh = 1.
        for i in range(len(NE_matrix)):
            NE_matrix[i][i] = sim_sh * max(NE_matrix[i])

        data = NE_matrix.reshape(-1)
        data = np.sort(data)
        data = data[:-int(len(data) * 0.02)]

        min_sh = data[0]
        max_sh = data[-1]

        delta = (max_sh - min_sh) / 100

        temp_cnt = []
        for i in range(20):
            s_sh = min_sh + delta * i
            e_sh = s_sh + delta
            temp_data = data[data > s_sh]
            temp_data = temp_data[temp_data < e_sh]
            temp_cnt.append([(s_sh + e_sh) / 2, len(temp_data)])

        candi_sh = -1
        for i in range(len(temp_cnt)):
            pear_sh, pear_cnt = temp_cnt[i]
            if 0 < i < len(temp_cnt) - 1:
                if pear_cnt < temp_cnt[i + 1][1] and pear_cnt < temp_cnt[i - 1][1]:
                    candi_sh = pear_sh
                    break
        if candi_sh < 0:
            for i in range(1, len(temp_cnt)):
                pear_sh, pear_cnt = temp_cnt[i]
                if pear_cnt * 2 < temp_cnt[i - 1][1]:
                    candi_sh = pear_sh
        if candi_sh == -1:
            candi_sh = 0.3

        propor = len(NE_matrix[NE_matrix <= candi_sh]) / (len(NE_matrix) ** 2)
        propor = 1 - propor
        thres = np.sort(NE_matrix)[:, -int(len(NE_matrix) * propor)]
        co_matrix.T[NE_matrix.T <= thres] = 0

    else:
        return

    N = len(co_matrix)

    up_K = np.sort(co_matrix)[:, -K]

    mat_K = np.zeros(co_matrix.shape)
    mat_K.T[co_matrix.T >= up_K] = 1

    thres_L = np.sort(co_matrix.flatten())[-int(((N * N) // (1 // (L + 1e-8))))]
    mat_K.T[co_matrix.T < thres_L] = 0

    return mat_K


# !/usr/bin/env python

from sklearn import preprocessing


def read_csv1(filename1, filename2, take_log):
    dataset = {}
    data = pd.read_csv(filename1, index_col=0, sep='\t')
    print(data.shape)
    print('Data loaded')
    print('Before filtering...')
    print(' Number of genes is {}'.format(len(data.index.values)))
    print(' Number of cells is {}'.format(len(data.columns.values)))

    cluster_labels = pd.read_csv(filename2, sep=',').values
    # data = Selecting_highly_variable_genes(data, 2000)
    data = pd.DataFrame(data)
    dataset['cell_labels'] = data.columns.values
    dataset['cluster_labels'] = cluster_labels[:, -1]
    gene_sym = data.index.values
    gene_exp = data.values

    if take_log:
        gene_exp = np.log2(gene_exp + 1)

    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym

    return dataset


def read_txt(filename, take_log):
    dataset = {}
    df = pd.read_table(filename, header=None)
    dat = df[df.columns[1:]].values
    dataset['cell_labels'] = dat[8, 1:]
    gene_sym = df[df.columns[0]].tolist()[11:]
    gene_exp = dat[11:, 1:].astype(np.float32)
    if take_log:
        gene_exp = np.log2(gene_exp + 1)
    dataset['gene_exp'] = gene_exp
    dataset['gene_sym'] = gene_sym
    dataset['cell_labels'] = convert_strclass_to_numclass(dataset['cell_labels'])

    save_csv(gene_exp, gene_sym, dataset['cell_labels'])

    return dataset


def pre_processing_single1(filename1, filename2, pre_process_paras, type='csv'):
    """ pre-processing of multiple droplet_lung
    Args:
        dataset_file_list: list of filenames of droplet_lung
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of droplet_lung
    """
    # parameters
    take_log = pre_process_paras['take_log']
    scaling = pre_process_paras['scaling']
    dataset_list = []
    data_file1 = filename1
    data_file2 = filename2
    if type == 'csv':
        dataset = read_csv1(data_file1, data_file2, take_log)
    elif type == 'txt':
        dataset = read_txt(data_file1, take_log)
    dataset['gene_exp'] = dataset['gene_exp']

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)

    dataset_list.append(dataset)
    return dataset_list


def load_data2(dataset_name,data_path):
    k = 10
    load_type = 'csv'
    dropout_rate = 0.4
    method = 'NE'
    if k == 1:
        dropout_rate = 0.
    else:
        dropout_rate = dropout_rate

    file_path1 = f"{data_path}/data.tsv"
    file_path2 = f"{data_path}/label.ann"
    dataset = load_data_origin_data1(file_path1, file_path2, load_type, scaling=False)
    print(f" This is the dataset: Type:{type(dataset)}| {dataset}")
    print(dataset_name)
    print(dataset.x.shape)
    print(dataset.y.shape)
    np.seterr(divide='ignore', invalid='ignore')

    k = int(len(dataset.y) / 100)
    if k < 5:
        k = 5
    if k > 20:
        k = 20

    A = getGraph(dataset_name, dataset.x, 0, k, method)
    A = torch.tensor(A)

    return  A, dataset.x,dataset.y
