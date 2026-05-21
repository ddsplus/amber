import numpy as np
import torch
import scipy.sparse as sp


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1] # Number of columns of matrix = number of hyperedge
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)
    # Guard zero-degree nodes/edges to avoid inf -> NaN propagation.
    inv_de = np.power(DE, -1.0, where=(DE != 0), out=np.zeros_like(DE, dtype=float))
    inv_dv2 = np.power(DV, -0.5, where=(DV != 0), out=np.zeros_like(DV, dtype=float))
    invDE = np.mat(np.diag(inv_de))
    DV2 = np.mat(np.diag(inv_dv2))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        G = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(G))
        # G = torch.Tensor(G)
        return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
