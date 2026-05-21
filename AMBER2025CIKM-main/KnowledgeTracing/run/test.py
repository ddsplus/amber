import torch
import tqdm
from sklearn import metrics
from KnowledgeTracing.Constant import Constants as C

def test_epoch(model, testLoader, device):
    model.eval()

    predictions = []
    ground_truths = []
    q = C.NUM_OF_QUESTIONS

    with torch.no_grad():
        for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
            logit_c, logit_t, ensemble_logit,_ ,_ = model(batch)


            p_c = torch.softmax(logit_c, dim=-1)
            p_t = torch.softmax(logit_t, dim=-1)
            p_enm = torch.softmax(ensemble_logit, dim=-1) if ensemble_logit is not None else None

            prediction = torch.tensor([], device=device)
            ground_truth = torch.tensor([], device=device)

            p_mean = (p_c + p_t + p_enm) / 3.0

            for student_idx in range(batch.shape[0]):
                student_data = batch[student_idx]
                delta = student_data[:, :q] + student_data[:, q:]
                a = (((student_data[:, :q] - student_data[:, q:]).sum(1) + 1) // 2)[1:]

                pred_c = p_c[student_idx][:C.max_step - 1].mm(delta[1:].t())
                pred_t = p_t[student_idx][:C.max_step - 1].mm(delta[1:].t())
                pred_enm = p_enm[student_idx][:C.max_step - 1].mm(delta[1:].t()) if p_enm is not None else None

                index = torch.arange(C.max_step - 1, device=device).unsqueeze(0)
                pc = pred_c.gather(0, index)[0]
                pt = pred_t.gather(0, index)[0]
                penm = pred_enm.gather(0, index)[0] if pred_enm is not None else None

                valid_idx = (pc > 0).nonzero(as_tuple=True)[0]
                if valid_idx.numel() > 0:
                    last_idx = valid_idx[-1]
                    pc = pc[:last_idx + 1]
                    pt = pt[:last_idx + 1]
                    if penm is not None:
                        penm = penm[:last_idx + 1]
                    a = a[:last_idx + 1]

                p_mean_student = (pt + pc + penm) / 3.0

                prediction = torch.cat([prediction, p_mean_student])
                ground_truth = torch.cat([ground_truth, a])




    auc, acc = performance(ground_truth, prediction)
    print(f'AUC: {auc:.4f}, Accuracy: {acc:.4f}')

    return auc, acc



def performance(ground_truth, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth.detach().cpu().numpy(),
                                             prediction.detach().cpu().numpy())
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(ground_truth.detach().cpu().numpy(), torch.round(prediction).detach().cpu().numpy())
    print(f'auc: {auc:.4f}, acc: {acc:.4f}')
    return auc, acc
