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
    H_sp = sp.csr_matrix(np.asarray(H, dtype=np.float32))
    n_edge = H_sp.shape[1]

    # Node/edge degree.
    DV = np.asarray(H_sp.sum(axis=1)).reshape(-1)
    DE = np.asarray(H_sp.sum(axis=0)).reshape(-1)

    # Guard zero-degree nodes/edges to avoid inf -> NaN propagation.
    inv_de = np.zeros_like(DE, dtype=np.float32)
    nonzero_de = DE != 0
    inv_de[nonzero_de] = 1.0 / DE[nonzero_de]

    inv_dv2 = np.zeros_like(DV, dtype=np.float32)
    nonzero_dv = DV != 0
    inv_dv2[nonzero_dv] = np.power(DV[nonzero_dv], -0.5)

    invDE = sp.diags(inv_de, format='csr')
    DV2 = sp.diags(inv_dv2, format='csr')
    W = sp.eye(n_edge, dtype=np.float32, format='csr')

    if variable_weight:
        DV2_H = DV2 @ H_sp
        invDE_HT_DV2 = invDE @ H_sp.transpose() @ DV2
        return DV2_H, W, invDE_HT_DV2

    G = DV2 @ H_sp @ W @ invDE @ H_sp.transpose() @ DV2
    G = sparse_mx_to_torch_sparse_tensor(G.tocoo())
    return G


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
