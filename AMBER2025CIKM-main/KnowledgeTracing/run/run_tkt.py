import sys
from pathlib import Path
import os

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models import hypergraph_utils as hgut
from KnowledgeTracing.model.Modelnew import DKT
from KnowledgeTracing.data.dataloader import getLoader
from KnowledgeTracing.Constant import Constants as C
from torch import optim as optima
from KnowledgeTracing.evaluation import eval
import torch
import logging
from datetime import datetime
import numpy as np
import warnings
import random
import pandas as pd
import time

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    torch.cuda.set_device(0)

# Limit CPU thread contention to avoid full-core saturation.
try:
    _threads = int(os.environ.get('KT_TORCH_THREADS', '8'))
    _interop = int(os.environ.get('KT_TORCH_INTEROP_THREADS', '2'))
    torch.set_num_threads(max(1, _threads))
    torch.set_num_interop_threads(max(1, _interop))
except Exception:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _detect_max_qid(pid_train_path: Path) -> int:
    max_qid = 0
    with pid_train_path.open('r', encoding='UTF-8-sig') as f:
        while True:
            len_line = f.readline()
            if not len_line:
                break
            qline = f.readline().strip().strip(',')
            _ = f.readline()
            _ = f.readline()
            if not qline:
                continue
            vals = [int(x) for x in qline.split(',') if x]
            if vals:
                max_qid = max(max_qid, max(vals))
    return max_qid


def _sync_runtime_constants_with_data():
    pid_path = Path(f'../../Dataset/{C.DATASET}/{C.DATASET}_pid_train.csv')
    if not pid_path.exists():
        print(f'[WARN] Missing pid_train file for dataset sync: {pid_path}')
        return

    detected_q = _detect_max_qid(pid_path)
    if detected_q <= 0:
        print(f'[WARN] Could not detect valid question id from: {pid_path}')
        return

    if detected_q != C.NUM_OF_QUESTIONS:
        print(f'[WARN] NUM_OF_QUESTIONS mismatch: Constants={C.NUM_OF_QUESTIONS}, data max_qid={detected_q}')
        print('       Auto-syncing runtime NUM_OF_QUESTIONS to data max_qid for this run.')
        C.NUM_OF_QUESTIONS = detected_q

    # Keep H aligned to dataset so different datasets cannot accidentally share
    # incompatible hypergraph/weight cache files.
    if C.H != C.DATASET:
        print(f'[INFO] Runtime H tag adjusted: {C.H} -> {C.DATASET}')
        C.H = C.DATASET


_sync_runtime_constants_with_data()

print('GPU state: ', torch.cuda.is_available())
print('Dataset: ' + C.DATASET + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + '\n')

logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
date = datetime.now()
os.makedirs('log', exist_ok=True)
handler = logging.FileHandler(f'log/{date.year}_{date.month}_{date.day}_result.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('This is a new training log')
logger.info('\nDataset: ' + str(C.DATASET) + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + ', Batch_size: ' + str(C.BATCH_SIZE))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


set_seed(216)

trainLoader, testLoader = getLoader(C.DATASET)
print(f'Effective batch-size is controlled in dataloader for stability; configured BATCH_SIZE={C.BATCH_SIZE}')
loss_func = eval.lossFunc(C.HIDDEN, C.MAX_STEP, device)


def _build_h_from_pid_train(pid_train_path: Path, num_q: int) -> pd.DataFrame:
    users_rows = []
    with pid_train_path.open('r', encoding='UTF-8-sig') as f:
        while True:
            len_line = f.readline()
            if not len_line:
                break
            q_line = f.readline()
            _ = f.readline()  # placeholder line
            a_line = f.readline()
            if not q_line or not a_line:
                break

            try:
                seq_len = int(len_line.strip().strip(','))
            except ValueError:
                seq_len = 0

            q = np.array([int(x) for x in q_line.strip().strip(',').split(',') if x], dtype=np.int64)
            a = np.array([int(x) for x in a_line.strip().strip(',').split(',') if x], dtype=np.int64)
            if seq_len > 0:
                q = q[:seq_len]
                a = a[:seq_len]

            if q.size == 0 or a.size == 0:
                users_rows.append(np.array([], dtype=np.int64))
                continue

            valid = (q > 0) & (q <= num_q) & ((a == 0) | (a == 1))
            q = q[valid] - 1
            a = a[valid]

            rows = np.where(a > 0, q, num_q + q)
            users_rows.append(np.unique(rows))

    n_users = len(users_rows)
    H = np.zeros((2 * num_q, n_users), dtype=np.int8)
    for uidx, rows in enumerate(users_rows):
        if rows.size > 0:
            H[rows, uidx] = 1
    return pd.DataFrame(H)


def _load_or_rebuild_h() -> pd.DataFrame:
    h_root = Path('../../Dataset/H')
    h_path = h_root / f'{C.DATASET}.csv'
    legacy_h_path = h_root / f'{C.H}.csv'
    expected_rows = 2 * C.NUM_OF_QUESTIONS

    for candidate in [h_path, legacy_h_path]:
        if candidate.exists():
            h_df = pd.read_csv(candidate, header=None)
            if h_df.shape[0] == expected_rows:
                if candidate != h_path:
                    print(f'[INFO] Using legacy H file: {candidate}')
                return h_df
            print(f'[WARN] H shape mismatch at {candidate}: rows={h_df.shape[0]}, expected={expected_rows}.')

    print(f'[WARN] No compatible H file found under {h_root}. Rebuilding from pid_train.')

    pid_path = Path(f'../../Dataset/{C.DATASET}/{C.DATASET}_pid_train.csv')
    rebuilt = _build_h_from_pid_train(pid_path, C.NUM_OF_QUESTIONS)
    h_path.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(h_path, index=False, header=False)
    print(f'Rebuilt H from pid_train: shape={rebuilt.shape}, path={h_path}')
    return rebuilt


def KTtrain():
    t0 = time.time()
    g_cache = f'../../Dataset/{C.DATASET}/G_{C.DATASET}_q{C.NUM_OF_QUESTIONS}.pt'
    expected_rows = 2 * C.NUM_OF_QUESTIONS
    use_cache = False
    if os.path.exists(g_cache):
        cached_g = torch.load(g_cache, map_location='cpu').coalesce()
        if cached_g.shape[0] == expected_rows:
            G = cached_g.to(device)
            use_cache = True
            print(f'Loaded cached G in {time.time() - t0:.2f}s')
        else:
            print(f'[WARN] Cached G shape mismatch: {tuple(cached_g.shape)} vs expected ({expected_rows}, {expected_rows}). Rebuilding.')

    if not use_cache:
        h_mat = _load_or_rebuild_h()
        adj = hgut.generate_G_from_H(h_mat)
        G = adj.coalesce().to(device)
        os.makedirs(f'../../Dataset/{C.DATASET}', exist_ok=True)
        torch.save(adj.coalesce().cpu(), g_cache)
        print(f'Built and cached G in {time.time() - t0:.2f}s')

    t1 = time.time()
    adj_out, adj_in = get_adj()
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)
    print(f'Loaded/Built transition adj in {time.time() - t1:.2f}s')
    model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).to(device)

    optimizer = optima.Adam(model.parameters(), lr=C.LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-3)

    best_auc = 0.0
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(C.EPOCH):
        print('epoch: ' + str(epoch + 1) + '            lr = ', optimizer.param_groups[0]["lr"])
        model, optimizer = eval.train_epoch(model, trainLoader, optimizer, scheduler, loss_func)

        with torch.no_grad():
            auc, acc = eval.test_epoch(model, testLoader, loss_func, device)
            print(f'Test AUC: {auc:.4f}, Test ACC: {acc:.4f}')
            logger.info(f'Epoch {epoch + 1}: Test AUC={auc:.4f}, Test ACC={acc:.4f}')

            if best_auc < auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch + 1
                os.makedirs('../model', exist_ok=True)
                torch.save(model.state_dict(), '../model/save' + C.H + 'modelS_weights_teacher.pth')

            print('Best auc at present: %f  acc:  %f  Best epoch: %d' % (best_auc, best_acc, best_epoch))


if __name__ == '__main__':
    KTtrain()
