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
    _threads = int(os.environ.get('KT_TORCH_THREADS', '1'))
    _interop = int(os.environ.get('KT_TORCH_INTEROP_THREADS', '1'))
    torch.set_num_threads(max(1, _threads))
    torch.set_num_interop_threads(max(1, _interop))
except Exception:
    pass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def _sanity_check_question_count():
    train_path = f'../../Dataset/{C.DATASET}/{C.DATASET}_pid_train.csv'
    max_qid = 0
    with open(train_path, 'r', encoding='UTF-8-sig') as f:
        while True:
            l1 = f.readline()
            if not l1:
                break
            qline = f.readline().strip().strip(',')
            _ = f.readline()
            _ = f.readline()
            if not qline:
                continue
            vals = [int(x) for x in qline.split(',') if x]
            if vals:
                max_qid = max(max_qid, max(vals))
    if max_qid != C.NUM_OF_QUESTIONS:
        print(f'[WARN] NUM_OF_QUESTIONS mismatch: Constants={C.NUM_OF_QUESTIONS}, data max_qid={max_qid}')
        print('       This can cause major slowdown or shape issues. Please sync Constants.py with data stats.')

_sanity_check_question_count()


def KTtrain():
    t0 = time.time()
    g_cache = f'../../Dataset/{C.DATASET}/G_{C.H}_q{C.NUM_OF_QUESTIONS}.pt'
    if os.path.exists(g_cache):
        G = torch.load(g_cache, map_location='cpu').coalesce().to(device)
        print(f'Loaded cached G in {time.time() - t0:.2f}s')
    else:
        h_mat = pd.read_csv(r'../../Dataset/H/' + C.H + '.csv', header=None)
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
