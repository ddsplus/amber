from min_norm_solvers import MinNormSolver
import sys
project_root = "/root/KT/AMBER"  # replace to your project root
sys.path.append(project_root)
from KnowledgeTracing.DirectedGCN.load_data import get_adj
from KnowledgeTracing.hgnn_models import hypergraph_utils as hgut
from KnowledgeTracing.model.Modelnew import DKT
from KnowledgeTracing.data.dataloader import getLoader
from KnowledgeTracing.Constant import Constants as C
from torch import optim as optima
from KnowledgeTracing.run.test import test_epoch
import torch
import logging
from datetime import datetime
import numpy as np
import warnings
import os
import random
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from KnowledgeTracing.run.comloss import CombinedLossI, CombinedLossID
from torch.optim.lr_scheduler import LambdaLR

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    torch.cuda.set_device(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('GPU state: ', torch.cuda.is_available())
print('Dataset: ' + C.DATASET + ', Ques number: ' + str(C.NUM_OF_QUESTIONS) + '\n')

logger = logging.getLogger('main')
logger.setLevel(level=logging.DEBUG)
date = datetime.now()
os.makedirs('log', exist_ok=True)
handler = logging.FileHandler(f'log/{date.year}_{date.month}_{date.day}_result2s.log')
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
    torch.backends.cudnn.benchmark = False


set_seed(216)

trainLoader, testLoader = getLoader(C.DATASET)

n = 3


def KTtrain():
    torch.autograd.set_detect_anomaly(True)

    adj = hgut.generate_G_from_H(pd.read_csv(r'../../Dataset/H/' + C.H + '.csv', header=None))
    G = adj.to(device)
    adj_out, adj_in = get_adj()
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    student_model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).to(device)
    teacher_model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).to(device)

    teacher_path = '../model/save' + C.H + 'modelS_weights_teacher.pth'
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))

    comblossd = CombinedLossID(device=device)
    comblossc = CombinedLossI(device=device)

    student_optimizer = optima.Adam(student_model.parameters(), lr=C.LR)
    teacher_optimizer = optima.Adam(teacher_model.parameters(), lr=C.teacher_learning_rate)

    def linear_decay(step):
        return max(0.0, 1 - step / num_training_steps)

    num_training_steps = C.EPOCH * len(trainLoader) * 10
    student_scheduler = LambdaLR(student_optimizer, lr_lambda=linear_decay)
    teacher_scheduler = LambdaLR(teacher_optimizer, lr_lambda=linear_decay)

    best_auc = 0.0
    best_epoch = 0
    best_acc = 0.0

    for epoch in range(C.EPOCH):
        skip_steps = epoch > n
        batches_buffer = []

        train_progress = tqdm(trainLoader, desc=f"Training Epoch {epoch + 1}/{C.EPOCH}", leave=False)
        for d_step, d_batch in enumerate(train_progress):
            if not skip_steps:
                batches_buffer.append((d_step, d_batch))

                fast_weights = OrderedDict((name, param.clone()) for (name, param) in student_model.named_parameters())

                student_model.train()
                teacher_model.eval()

                assume_loss_total = 0.0
                for step, (_, buffer_d_batch) in enumerate(batches_buffer):
                    logit_c, logit_t, ensemble_logit, out_h_s, out_d_s = student_model(buffer_d_batch)

                    with torch.no_grad():
                        logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(buffer_d_batch)

                    loss = comblossc(
                        logit_c, logit_t, ensemble_logit,
                        logit_c_teacher, logit_t_teacher, ensemble_logit_teacher,
                        out_h_s, out_d_s, out_h_t, out_d_t, d_batch
                    )

                    grads = torch.autograd.grad(
                        loss,
                        student_model.parameters() if step == 0 else fast_weights.values(),
                        create_graph=True,
                        allow_unused=True,
                        retain_graph=True
                    )
                    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, fast_weights.values())]

                    fast_weights = OrderedDict(
                        (name, param - C.assume_s_step_size * grad)
                        for ((name, param), grad) in zip(fast_weights.items(), grads)
                    )
                    assume_loss_total += loss.item()

                print(f"\nStep 1: Assume Loss: {assume_loss_total:.4f}")

                student_model.eval()
                teacher_model.train()

                with torch.no_grad():
                    for name, param in student_model.named_parameters():
                        param.copy_(fast_weights[name])

                with torch.no_grad():
                    logit_c_student, logit_t_student, ensemble_logit_student, out_h_s, out_d_s = student_model(d_batch)
                logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(d_batch)

                quiz_loss = comblossc(
                    logit_c_teacher, logit_t_teacher, ensemble_logit_teacher,
                    logit_c_student, logit_t_student, ensemble_logit_student,
                    out_h_s, out_d_s, out_h_t, out_d_t,
                    d_batch
                )

                teacher_grads = torch.autograd.grad(quiz_loss, teacher_model.parameters(), allow_unused=True)
                teacher_grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(teacher_grads, teacher_model.parameters())]

                for param, grad in zip(teacher_model.parameters(), teacher_grads):
                    param.grad = grad

                torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
                teacher_optimizer.step()
                teacher_optimizer.zero_grad()

                print(f"\nStep 2: Teacher update loss: {quiz_loss.item():.4f}")
                del teacher_grads
                del grads
                del fast_weights

            student_model.train()
            teacher_model.eval()

            logit_c_student, logit_t_student, ensemble_logit_student, out_h_s, out_d_s = student_model(d_batch)
            with torch.no_grad():
                logit_c_teacher, logit_t_teacher, ensemble_logit_teacher, out_h_t, out_d_t = teacher_model(d_batch)

            loss_supervised, loss_kd, loss_embed = comblossd(
                logit_c_student, logit_t_student, ensemble_logit_student,
                logit_c_teacher, logit_t_teacher, ensemble_logit_teacher,
                out_h_s, out_d_s, out_h_t, out_d_t, d_batch
            )

            loss_supervised.backward(retain_graph=True)
            grads_supervised = [param.grad.clone() for param in student_model.parameters() if param.grad is not None]

            student_optimizer.zero_grad()
            loss_kd.backward(retain_graph=True)
            grads_kd = [param.grad.clone() for param in student_model.parameters() if param.grad is not None]

            vecs = [grads_supervised, grads_kd]
            sol, _ = MinNormSolver.find_min_norm_element(vecs)
            supervised_weight, distillation_weight = sol

            total_loss = (
                supervised_weight * loss_supervised +
                distillation_weight * loss_kd +
                1.0 * loss_embed
            )

            student_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            student_optimizer.step()

            batches_buffer = []
            student_scheduler.step()
            teacher_scheduler.step()

            print(f"\nStep 3: Total loss (student update): {total_loss.item():.4f}")

        with torch.no_grad():
            auc, acc = test_epoch(student_model, testLoader, device)
            print(f"Epoch {epoch + 1}: Test AUC: {auc:.4f}, Test ACC: {acc:.4f}")

            if auc > best_auc:
                best_auc = auc
                best_acc = acc
                best_epoch = epoch + 1
                os.makedirs('../model', exist_ok=True)
                torch.save(student_model.state_dict(), '../model/save' + C.H + 'modelS_weights.pth')

            logger.info(f"Epoch {epoch + 1}: Test AUC={auc:.4f}, Test ACC={acc:.4f}")
            logger.info(f"Best AUC: {best_auc:.4f}, Best ACC: {best_acc:.4f}, Best Epoch: {best_epoch}")


def KTtest():
    adj = hgut.generate_G_from_H(pd.read_csv(r'../../Dataset/H/' + C.H + '.csv', header=None))
    G = adj.to(device)
    adj_out, adj_in = get_adj()
    adj_in = adj_in.to(device)
    adj_out = adj_out.to(device)

    model = DKT(C.HIDDEN, C.LAYERS, G, adj_out, adj_in).to(device)
    model.load_state_dict(torch.load('../model/save' + C.H + 'modelS_weights.pth', map_location=device))
    model.eval()

    print('loading the best model...')
    with torch.no_grad():
        auc, acc = test_epoch(model, testLoader, device=device)
        print(f"Test AUC: {auc:.4f}, Test ACC: {acc:.4f}")


if __name__ == '__main__':
    KTtrain()
    KTtest()
