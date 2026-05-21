import numpy as np
import scipy.sparse as sp
from KnowledgeTracing.Constant import Constants as C
import tqdm
import itertools
import torch
import os


def _detect_max_qid(path):
    max_qid = 0
    with open(path, 'r', encoding='UTF-8-sig') as train:
        for _, ques, _, _ in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='Scan max qid:            ',
                                       mininterval=2):
            if ques is None:
                break
            q_vals = np.array(ques.strip().strip(',').split(',')).astype(int)
            if q_vals.size > 0:
                cur_max = int(q_vals.max())
                if cur_max > max_qid:
                    max_qid = cur_max
    return max_qid


def get_adj():
    path = '../../Dataset/' + C.DATASET + '/' + C.DATASET + '_pid_train.csv'
    detected_q = _detect_max_qid(path)
    q = max(C.NUM_OF_QUESTIONS, detected_q)
    if q != C.NUM_OF_QUESTIONS:
        print(f'[WARN] get_adj detected larger qid: constants={C.NUM_OF_QUESTIONS}, data={detected_q}.')
        print(f'       Runtime q is expanded to {q} for adjacency building.')
        C.NUM_OF_QUESTIONS = q

    cache_path = '../../Dataset/' + C.DATASET + '/adj_' + C.DATASET + '_q' + str(q) + '.pt'
    if os.path.exists(cache_path):
        cached = torch.load(cache_path, map_location='cpu')
        return cached['adj_out'].coalesce(), cached['adj_in'].coalesce()

    src_all = []
    dst_all = []
    total_nodes = 2 * q

    with open(path, 'r', encoding='UTF-8-sig') as train:
        for len, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='Generate adjacency matrix:    ',
                                           mininterval=2):
            if len is None or ques is None or ans is None:
                break
            len = int(len.strip().strip(','))
            ques = np.array(ques.strip().strip(',').split(',')).astype(int)
            ans = np.array(ans.strip().strip(',').split(',')).astype(int)
            if len > 1:
                seq = ques.copy()
                wrong_mask = (ans[:len] == 0)
                seq[:len][wrong_mask] += q
                src = seq[:len - 1] - 1
                dst = seq[1:len] - 1
                valid = (src >= 0) & (src < total_nodes) & (dst >= 0) & (dst < total_nodes)
                if np.any(valid):
                    src_all.append(src[valid])
                    dst_all.append(dst[valid])

    if src_all:
        src_cat = np.concatenate(src_all, axis=0)
        dst_cat = np.concatenate(dst_all, axis=0)
        data = np.ones_like(src_cat, dtype=np.float32)
        resout = sp.coo_matrix((data, (src_cat, dst_cat)), shape=(total_nodes, total_nodes), dtype=np.float32).tocsr()
        resout.sum_duplicates()
    else:
        resout = sp.csr_matrix((total_nodes, total_nodes), dtype=np.float32)

    resin = resout.transpose().tocsr()
    resout = normalize(resout + sp.eye(resout.shape[0], format='csr', dtype=np.float32))
    resin = normalize(resin + sp.eye(resin.shape[0], format='csr', dtype=np.float32))

    resout = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(resout))
    resin = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(resin))

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({'adj_out': resout.coalesce(), 'adj_in': resin.coalesce()}, cache_path)
    return resout, resin


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)

    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
