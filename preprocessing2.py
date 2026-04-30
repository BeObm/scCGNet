from __future__ import print_function, division
import argparse
from keras import backend as K
from sklearn.cluster import KMeans
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
import os
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import h5py
import torch
from torch.utils.data import Dataset
import scanpy as sc
from sklearn.preprocessing import scale, minmax_scale
import time
import numpy as np
import scipy.sparse as sp


seed = 666
torch.set_num_threads(seed)
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import random

random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def print_time(f):
    """Decorator of viewing function runtime.
    eg:
        ```py
        from print_time import print_time as pt
        @pt
        def work(...):
            print('work is running')
        word()
        # work is running
        # --> RUN TIME: <work> : 2.8371810913085938e-05
        ```
    """

    def fi(*args, **kwargs):
        s = time.time()
        res = f(*args, **kwargs)
        print('--> RUN TIME: <%s> : %s' % (f.__name__, time.time() - s))
        return res

    return fi




def load_graph(dataset, k=None, n=10, label=None):
    import os
    graph_path = os.getcwd()
    if k:
        path = graph_path + '/{}{}_graph.txt'.format(dataset, k)
    else:
        path =graph_path +  '/{}_graph.txt'.format(dataset)


    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    import os
    print("delete file: ", path)
    os.remove(path)

    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def anta_normalize(x, y):
    # preprocessing scRNA-seq read counts matrix
    y = y.astype(np.int32)
    adata = sc.AnnData(x)
    adata.obs['Group'] = y

    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         copy=True)

    adata = process_normalize(adata,
                              size_factors=True,
                              normalize_input=True,
                              logtrans_input=True)

    print(adata.X.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    x = adata.X.astype(np.float32)
    y = y.astype(np.int32)
    raw_data = adata.raw.X
    return x, y, adata.obs.size_factors, raw_data


class load_data_origin_data(Dataset):
    def __init__(self, dataset, load_type="csv", take_log=False, scaling=False):
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
            dataset_list = pre_processing_single(dataset, pre_process_paras, type='csv')
            self.x = dataset_list[0]['gene_exp'].transpose().astype(np.float32)
            self.y = dataset_list[0]['cell_labels'].astype(np.int32)
            self.cluster_label = dataset_list[0]['cluster_labels'].astype(np.int32)

        if load_type == "csv":
            load_csv()
        elif load_type == "h5":
            load_h5()
        elif load_type == "txt":
            load_txt()



    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


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
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))

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

    save_csv(gene_exp, gene_sym,  dataset['cell_labels'])

    return dataset

def pre_processing_single1(filename1, filename2, pre_process_paras, type='csv'):
    """ pre-processing of multiple datasets
    Args:
        dataset_file_list: list of filenames of datasets
        pre_process_paras: dict, parameters for pre-processing
    Returns:
        dataset_list: list of datasets
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
    dataset['gene_exp'] = dataset['gene_exp'].astype(np.float32)

    if scaling:  # scale to [0,1]
        minmax_scale(dataset['gene_exp'], feature_range=(0, 1), axis=1, copy=False)

    dataset_list.append(dataset)
    return dataset_list


def Selecting_highly_variable_genes(X, highly_genes):
    adata = sc.AnnData(X)
    adata.var_names_make_unique()
    # sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    # sc.pp.scale(adata, max_value=3)
    data = adata.X

    return data

def construct_graph_kmean(file_name, features, pred_label, label, load_type='csv', topk=None, method='ncos'):
    import os
    graph_path = os.getcwd()
    if topk:
        fname =graph_path + '/{}{}_graph.txt'.format(file_name, topk)
    else:
        fname =graph_path + '/{}_graph.txt'.format(file_name)
    num = len(label)
    dist = None

    if method == 'spearmanr':
        dist = spearmanr(features, axis=1)[0]
    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)
    elif method == 'ncos':
        dist = cosine_similarity(features, features)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    y_pred = pred_label
    f = open(fname, 'w')

    counter = 0
    total = 0

    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                total += 1
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))

    f.close()

    print('error rate: {}'.format(counter / (total*1.0)))
    print('error rate: {}'.format(counter / (num*topk)))

    return round(counter / (total * 1.0), 4)



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)



# Loss functions
def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def maie_class_loss(y_true, y_pred):
    loss_E = mae(y_true, y_pred)
    return loss_E


class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)

        return weight_output


class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)

        return weight_output


class GraphGAC(nn.Module):

    def __init__(self,n_enc_1, n_enc_2, n_enc_3,dropout_rate, n_heads,
                n_input, n_z, n_clusters, v=1):
        super(GraphGAC, self).__init__()

        self.gac = GAE(
            n_feat=n_input, F1=n_enc_1, F2=n_enc_2, F3=n_enc_3, n_z=n_z, dropout=dropout_rate, n_heads=n_heads
        )

        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(256, n_clusters)

        self.mlp = MLP_L(256)

        # attention on [Z_i || H_i]
        self.mlp1 = MLP_1(2 * n_enc_1)
        self.mlp2 = MLP_2(2 * n_enc_2)
        self.mlp3 = MLP_3(2 * n_enc_3)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def pretrain_ae(self, dataset, A):
        # train_loader = DataLoader(dataset, batch_size=args.pre_batch_size, shuffle=True)

        optimizer = Adam(self.gac.parameters(), lr=args.pre_lr)
        for epoch in range(args.pre_epoch):
            X = torch.Tensor(dataset.x).to(device)
            x_bar, h1, h2, h3, z = self.gac(X, A)  # X~，，，，H

            loss = F.mse_loss(x_bar, X)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def forward(self, x, A, adj):
        # DNN Module
        # X = np.array(x)
        x_bar, h1, h2, h3, z = self.gac(x, A)

        x_array = list(np.shape(x))
        n_x = x_array[0]

        # # AGCN-H
        z1 = self.agcn_0(x, adj)
        # z2
        m1 = self.mlp1(torch.cat((h1, z1), 1))
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:, 0], [n_x, 1])
        m12 = torch.reshape(m1[:, 1], [n_x, 1])
        m11_broadcast = m11.repeat(1, 128)
        m12_broadcast = m12.repeat(1, 128)
        z2 = self.agcn_1(m11_broadcast.mul(z1) + m12_broadcast.mul(h1), adj)
        # z3
        m2 = self.mlp2(torch.cat((h2, z2), 1))
        m2 = F.normalize(m2, p=2)
        m21 = torch.reshape(m2[:, 0], [n_x, 1])
        m22 = torch.reshape(m2[:, 1], [n_x, 1])
        m21_broadcast = m21.repeat(1, 64)
        m22_broadcast = m22.repeat(1, 64)
        z3 = self.agcn_2(m21_broadcast.mul(z2) + m22_broadcast.mul(h2), adj)
        # z4
        m3 = self.mlp3(torch.cat((h3, z3), 1))  # self.mlp3(h2)
        m3 = F.normalize(m3, p=2)
        m31 = torch.reshape(m3[:, 0], [n_x, 1])
        m32 = torch.reshape(m3[:, 1], [n_x, 1])
        m31_broadcast = m31.repeat(1, 32)
        m32_broadcast = m32.repeat(1, 32)
        z4 = self.agcn_3(m31_broadcast.mul(z3) + m32_broadcast.mul(h3), adj)

        # # AGCN-S
        u = self.mlp(torch.cat((z1, z2, z3, z4, z), 1))
        u = F.normalize(u, p=2)
        u0 = torch.reshape(u[:, 0], [n_x, 1])
        u1 = torch.reshape(u[:, 1], [n_x, 1])
        u2 = torch.reshape(u[:, 2], [n_x, 1])
        u3 = torch.reshape(u[:, 3], [n_x, 1])
        u4 = torch.reshape(u[:, 4], [n_x, 1])

        tile_u0 = u0.repeat(1, 128)
        tile_u1 = u1.repeat(1, 64)
        tile_u2 = u2.repeat(1, 32)
        tile_u3 = u3.repeat(1, 16)
        tile_u4 = u4.repeat(1, 16)

        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1)
        net_output = self.agcn_z(net_output, adj, active=False)
        predict = F.softmax(net_output, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class LoadDataset(Dataset):
    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))


def train_graphscc(dataset, A):
    model = GraphGAC(n_enc_1=args.n_enc_1,
                     n_enc_2=args.n_enc_2,
                     n_enc_3=args.n_enc_3,
                     dropout_rate=dropout_rate,
                     n_heads=args.n_attn_heads,
                     n_input=args.n_input,
                     n_z=args.n_z,
                     n_clusters=args.n_clusters,
                     v=1.0).to(device)
    print(model)

    model.pretrain_ae(dataset, A)

    optimizer = Adam(model.parameters(), lr=args.lr)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    A1 = A.float()

    with torch.no_grad():
        xbar, _, _, z = model(data, A, A1)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=666)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    y_pred_last = y_pred

    pae_acc, pae_nmi, pae_ari = eva(y, y_pred, 'pae', pp=False)
    print(':pae_acc {:.4f}'.format(pae_acc), ', pae_nmi {:.4f}'.format(pae_nmi), ', pae_ari {:.4f}'.format(pae_ari))

    features = z.data.cpu().numpy()
    # 利用KNN构造细胞图
    error_rate = construct_graph_kmean(args.name, features.copy(), y, y,
                                       load_type='csv', topk=args.k, method='ncos')
    adj = load_graph(args.name, k=args.k, n=dataset.x.shape[0])
    adj = adj.to(device)

    patient = 0
    series = False
    sil_logs = []
    final_pred = None
    max_sil = 0
    for epoch in range(args.train_epoch):
        if epoch % 1 == 0:
            # update_interval
            xbar, tmp_q, pred, z = model(data, A, adj)

            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            res1 = tmp_q.cpu().numpy().argmax(1)  # Q
            res2 = pred.data.cpu().numpy().argmax(1)  # Z
            res3 = p.data.cpu().numpy().argmax(1)  # P
            Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
            Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
            P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
            # G_acc, G_nmi, G_ari = eva(y, pred_label, str(epoch) + 'G', pp=False)
            print(epoch, ':Q_acc {:.5f}'.format(Q_acc), ', Q_nmi {:.5f}'.format(Q_nmi), ', Q_ari {:.5f}'.format(Q_ari))
            print(epoch, ':Z_acc {:.5f}'.format(Z_acc), ', Z_nmi {:.5f}'.format(Z_nmi), ', Z_ari {:.5f}'.format(Z_ari))
            print(epoch, ':P_acc {:.5f}'.format(P_acc), ', P_nmi {:.5f}'.format(P_nmi), ', p_ari {:.5f}'.format(p_ari))
            # print(epoch, ':G_acc {:.5f}'.format(G_acc), ', G_nmi {:.5f}'.format(G_nmi), ', G_ari {:.5f}'.format(G_ari))
            delta_label = np.sum(res2 != y_pred_last).astype(np.float32) / res2.shape[0]
            y_pred_last = res2
            if epoch > 0 and delta_label < 0.0001:
                if series:
                    patient += 1
                else:
                    patient = 0
                series = True
                if patient == 100:
                    print('Reached tolerance threshold. Stopping training.')
                    print("Z_acc: {}".format(Z_acc), "Z_nmi: {}".format(Z_nmi),
                          "Z_ari: {}".format(Z_ari))
                    break
            else:
                series = False

        x_bar, q, pred, _ = model(data, A, adj)
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)

        loss = args.kl_loss * kl_loss + args.ce_loss * ce_loss + re_loss
        print(epoch, ':kl_loss {:.5f}'.format(kl_loss), ', ce_loss {:.5f}'.format(ce_loss),
              ', re_loss {:.5f}'.format(re_loss), ', total_loss {:.5f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Q_acc, Q_nmi, Q_ari = eva(y, res1, str(epoch) + 'Q', pp=False)
    Z_acc, Z_nmi, Z_ari = eva(y, res2, str(epoch) + 'Z', pp=False)
    P_acc, P_nmi, p_ari = eva(y, res3, str(epoch) + 'P', pp=False)
    pd.DataFrame(res2).to_csv('result/pred_' + args.name + '.csv', index=False)
    print(epoch, ':Q_acc {:.4f}'.format(Q_acc), ', Q_nmi {:.4f}'.format(Q_nmi), ', Q_ari {:.4f}'.format(Q_ari))
    print(epoch, ':Z_acc {:.4f}'.format(Z_acc), ', Z_nmi {:.4f}'.format(Z_nmi), ', Z_ari {:.4f}'.format(Z_ari))
    print(epoch, ':P_acc {:.4f}'.format(P_acc), ', P_nmi {:.4f}'.format(P_nmi), ', p_ari {:.4f}'.format(p_ari))
    print('predict_y ', res2)
    print(args)
    return Z_acc, Z_nmi, Z_ari

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



#TODO: Fix this
class AnnSequence:
    def __init__(self, matrix, batch_size, sf=None):
        self.matrix = matrix
        if sf is None:
            self.size_factors = np.ones((self.matrix.shape[0], 1),
                                        dtype=np.float32)
        else:
            self.size_factors = sf
        self.batch_size = batch_size

    def __len__(self):
        return len(self.matrix) // self.batch_size

    def __getitem__(self, idx):
        batch = self.matrix[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_sf = self.size_factors[idx*self.batch_size:(idx+1)*self.batch_size]

        # return an (X, Y) pair
        return {'count': batch, 'size_factors': batch_sf}, batch


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
        #sc.pp.filter_cells(adata, min_counts=1)

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

def read_genelist(filename):
    genelist = list(set(open(filename, 'rt').read().strip().split('\n')))
    assert len(genelist) > 0, 'No genes detected in genelist file'
    print('### Autoencoder: Subset of {} genes will be denoised.'.format(len(genelist)))

    return genelist

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
def read_pickle(inputfile):
    return pickle.load(open(inputfile, "rb"))


"""
Construct a graph based on the cell features
"""
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

        NE_path = 'result/NE_' + dataset_str + '.csv'
        try:
            os.remove(NE_path)
        except:
            pass
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





def build_dataset():
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Klein')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pre_lr', type=float, default=1e-2)
    parser.add_argument('--n_clusters', default=5, type=int)
    parser.add_argument('--load_type', type=str, default='csv')
    parser.add_argument('--kl_loss', type=float, default=0.1)
    parser.add_argument('--ce_loss', type=float, default=0.01)
    parser.add_argument('--similar_method', type=str, default='ncos')
    parser.add_argument('--pre_batch_size', type=int, default=32)
    parser.add_argument('--pre_epoch', type=int, default=400)
    parser.add_argument('--train_epoch', type=int, default=8000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_enc_1', default=128, type=int, help='number of neurons in the 1-st layer of encoder')
    parser.add_argument('--n_enc_2', default=64, type=int, help='number of neurons in the 2-nd layer of encoder')
    parser.add_argument('--n_enc_3', default=32, type=int, help='number of neurons in the 1-st layer of encoder')
    parser.add_argument('--n_z', default=16, type=int, help='number of neurons in the 2-nd layer of encoder')
    parser.add_argument('--dropout_rate', default=0.4, type=float, help='dropout rate of neurons in autoencoder')
    parser.add_argument('--l2_reg', default=0, type=float, help='coefficient for L2 regularizition')
    parser.add_argument('--n_attn_heads', default=4, type=int, help='number of heads for attention')
    parser.add_argument('--method', default='NE', type=str, help='number of heads for attention')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    # torch.cuda.set_device(args.device)
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    n_clusters = args.n_clusters
    if args.k == 1:
        dropout_rate = 0.  # To avoid absurd results
    else:
        dropout_rate = args.dropout_rate

    file_path1 = "data/" + args.name + "/data.tsv"
    # file_path1 = "data/" + args.name + f"/{args.name}.h5"
    file_path2 = "data/" + args.name + "/label.ann"
    dataset = load_data_origin_data1(file_path1, file_path2, args.load_type, scaling=True)

    GAT_autoencoder_path = 'logs/GATae_' + args.name + '.h5'

    print(args.name)
    print(dataset.x.shape)
    print(dataset.y.shape)
    np.seterr(divide='ignore', invalid='ignore')

    args.k = int(len(dataset.y) / 100)
    if args.k < 5:
        args.k = 5
    if args.k > 20:
        args.k = 20

    args.n_clusters = len(np.unique(dataset.y))
    args.n_input = dataset.x.shape[1]
    A = getGraph(args.name, dataset.x, 0, args.k, args.method)
    A = torch.tensor(A).to(device)
    return dataset,A