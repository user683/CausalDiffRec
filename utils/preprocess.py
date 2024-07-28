import numpy as np
import scipy.sparse as sp
import torch


def mask_test_edges_dgl(graph):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)

    train_edge_idx = all_edge_idx[:]

    # NOTE: these edge lists only contain single direction of edge!
    return train_edge_idx


def get_test_valid_dgl(valid_graph, valid_false_graph, test_graph, test_false_graph):
    return creat_graph(valid_graph), creat_graph(valid_false_graph), creat_graph(test_graph), creat_graph(
        test_false_graph)


def creat_graph(graph):
    src, dst = graph.edges()
    edges_all = torch.stack([src, dst], dim=0)
    edges_all = edges_all.t().cpu().numpy()
    return edges_all


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized, sparse_to_tuple(adj_normalized)
