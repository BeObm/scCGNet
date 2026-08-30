from __future__ import annotations

import sys
from pathlib import Path
import scanpy
from scipy.sparse import issparse
import anndata as ad
import h5py
from utils import *
from sklearn.neighbors import kneighbors_graph
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from typing import Optional,Tuple
import scanpy as sc
import numpy as np
import pandas as pd
import torch



def parse_index_file(filename):
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

def load_h5_data(dataPath, dataset, hvg, n_neighbors=15, ts=None, metric='cosine'):
    adata = ad.read_h5ad(dataPath)


    if "n_count" not in adata.obs.columns:
        adata.obs['n_counts'] = np.asarray(
            adata.X.sum(axis=1)
        ).ravel()
    print("obs columns:")



    print(adata.obs.columns.tolist())
    scanpy.pp.filter_cells(adata, min_genes=1)
    scanpy.pp.filter_genes(adata, min_cells=1)
    adata.raw = adata
    adata.X = adata.X.astype(np.float32)
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    scanpy.pp.normalize_total(adata)
    scanpy.pp.log1p(adata)
    scanpy.pp.highly_variable_genes(adata, n_top_genes=hvg)
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



def adata_knn(adata, method, knn, n_neighbors=2, metric='cosine'):
    if adata.shape[0] >= 10000:
        scanpy.pp.pca(adata, n_comps=50)
        n_pcs = 50
    else:
        n_pcs = 0
    if method == 'umap':
        scanpy.pp.neighbors(adata, method=method, metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']
    elif method == 'gauss':
        scanpy.pp.neighbors(adata, method='gauss', metric=metric,
                            knn=knn, n_pcs=n_pcs, n_neighbors=n_neighbors)
        r_adj = adata.obsp['distances']
        adj = adata.obsp['connectivities']

    return adj, r_adj
