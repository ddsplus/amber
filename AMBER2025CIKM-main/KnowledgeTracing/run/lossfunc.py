import torch
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import logging
from geomloss import SamplesLoss

logger = logging.getLogger('main.eval')



class lossFunc(nn.Module):
    def __init__(self, hidden, max_step, device):
        super(lossFunc, self).__init__()
        self.crossEntropy = nn.BCELoss()
        self.q = C.NUM_OF_QUESTIONS
        self.hidden = hidden
        self.max_step = max_step
        self.device = device
        self.mse = nn.MSELoss()

        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.005, diameter=1, reach=1e3, backend="tensorized")

    def forward(self, logit_c, logit_t, logit_ensemble, batch, training_for_student=True):
        p_c = torch.softmax(logit_c, dim=-1).to(self.device)
        p_t = torch.softmax(logit_t, dim=-1).to(self.device)
        p_enm = torch.softmax(logit_ensemble, dim=-1).to(self.device) if logit_ensemble is not None else None

        if logit_ensemble is not None:
            p0_c = p_c.view(p_c.size(0), -1).to(self.device)  #(N, D)
            p0_t = p_t.view(p_t.size(0), -1).to(self.device)  # (N, D)
            p0_enm = p_enm.view(p_enm.size(0), -1).to(self.device)  # (N, D)

            loss_kd_c = self.sinkhorn_loss(p0_enm.unsqueeze(0), p0_c.unsqueeze(0))
            loss_kd_t = self.sinkhorn_loss(p0_enm.unsqueeze(0), p0_t.unsqueeze(0))
            loss_kd = C.kd_loss * (loss_kd_c + loss_kd_t)
        else:
            loss_kd = torch.zeros(1, device=self.device)  

        loss = torch.zeros(1, device=self.device)  
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        for student in range(batch.shape[0]):
            delta = (batch[student][:, :self.q] + batch[student][:, self.q:]).to(self.device)

            a = (((batch[student][:, 0:self.q] - batch[student][:, self.q:]).sum(1) + 1) // 2)[1:].to(self.device)

            temp_c = p_c[student][:self.max_step - 1].mm(delta[1:].t())
            temp_t = p_t[student][:self.max_step - 1].mm(delta[1:].t())
            if p_enm is not None:
                temp_enm = p_enm[student][:self.max_step - 1].mm(delta[1:].t())
            else:
                temp_enm = None

            index = torch.arange(self.max_step - 1, device=self.device).unsqueeze(0)
            pc = temp_c.gather(0, index)[0]
            pt = temp_t.gather(0, index)[0]
            if temp_enm is not None:
                penm = temp_enm.gather(0, index)[0]
            else:
                penm = None

            valid_idx = (pc > 0).nonzero(as_tuple=True)[0]
            if valid_idx.numel() > 0:
                last_idx = valid_idx[-1]
                pc = pc[:last_idx + 1]
                pt = pt[:last_idx + 1]
                if penm is not None:
                    penm = penm[:last_idx + 1]
                a = a[:last_idx + 1]

            loss = loss + self.crossEntropy(pt, a) + self.crossEntropy(pc, a)
            if penm is not None:
                loss = loss + self.crossEntropy(penm, a)

            if penm is not None:
                p_mean = (pt + pc + penm) / 3.0
            else:
                p_mean = (pt + pc) / 2.0

            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])

        return loss, loss_kd, prediction, ground_truth





