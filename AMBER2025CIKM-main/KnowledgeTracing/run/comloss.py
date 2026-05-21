import torch
import torch.nn.functional as F
import torch.nn as nn
from geomloss import SamplesLoss
from KnowledgeTracing.Constant import Constants as C

class CombinedLoss(nn.Module):
    def __init__(self, device, supervised_weight=1.0, distillation_weight=0.01, embed_weight=1.0, max_step=50):
        super(CombinedLoss, self).__init__()
        self.device = device

        
        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.005, diameter=1, reach=500, backend="tensorized")

        self.supervised_loss_fn = nn.CrossEntropyLoss()

        self.supervised_weight = supervised_weight
        self.distillation_weight = distillation_weight
        self.embed_weight = embed_weight

        self.max_step = max_step
        self.q = C.NUM_OF_QUESTIONS
        self.sigmoid = nn.Sigmoid()

    def forward(self, logit_c, logit_t, logit_ensemble, logit_teacher_c, logit_teacher_t, logit_teacher_ensemble, 
                out_h_student, out_h_teacher, out_d_student, out_d_teacher, batch):

        # ==========================
        # 1. Logit level kd loss
        # ==========================
        T = 0.5
        p_c = logit_c / T
        p_t = logit_t / T
        p_enm = logit_ensemble / T

        p_teacher_c = logit_teacher_c / T
        p_teacher_t = logit_teacher_t / T
        p_teacher_enm = logit_teacher_ensemble / T

        loss_kd = torch.zeros(1, device=self.device)
        loss_kd += self.sinkhorn_loss(p_c.view(p_c.size(0), -1), p_teacher_c.view(p_teacher_c.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_t.view(p_t.size(0), -1), p_teacher_t.view(p_teacher_t.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_enm.view(p_enm.size(0), -1), p_teacher_enm.view(p_teacher_enm.size(0), -1))

        # ==========================
        # 2. supervision loss
        # ==========================
        loss_supervised = torch.tensor(0.0, device=self.device)
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        for student in range(batch.shape[0]):
            delta = batch[student][:, :self.q] + batch[student][:, self.q:]

            a = (((batch[student][:, 0:self.q] -
                   batch[student][:, self.q:]).sum(1) + 1) //
                 2)[1:]  # [49]

            temp_c = p_c[student][:self.max_step - 1].mm(delta[1:].t())
            temp_t = p_t[student][:self.max_step - 1].mm(delta[1:].t())
            temp_enm = p_enm[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            pc = temp_c.gather(0, index)[0]
            pt = temp_t.gather(0, index)[0]
            penm = temp_enm.gather(0, index)[0]

            for i in range(len(pc) - 1, -1, -1):
                if pc[i] > 0:
                    pc = pc[:i + 1]
                    pt = pt[:i + 1]
                    penm = penm[:i + 1]
                    a = a[:i + 1]
                    break

            loss_supervised += (self.supervised_loss_fn(pt, a) + 
                                self.supervised_loss_fn(pc, a) + 
                                self.supervised_loss_fn(penm, a))

            p_mean = (pt + pc + penm) / 3.0
            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])

        # ==========================
        # 3. embedding level contrastive kd loss - l2 regularization
        # ==========================
        embed_loss_h = torch.norm(out_h_student - out_h_teacher, p=2).pow(2).mean()
        embed_loss_d = torch.norm(out_d_student - out_d_teacher, p=2).pow(2).mean()
        embed_loss = 0.5 * (embed_loss_h + embed_loss_d)

        # ==========================
        # 4. combined loss
        # ==========================
        print(loss_supervised)
        print(loss_kd)
        print(embed_loss)
        total_loss = C.loss_weight* (self.supervised_weight * loss_supervised + 
                      self.distillation_weight * loss_kd + 
                      self.embed_weight * embed_loss)

        return total_loss
    


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / self.temperature  # [512, 50]
        pos_sim = pos_sim.unsqueeze(1)  # [512, 1, 50]
        
        neg_sim = torch.stack([F.cosine_similarity(anchor, neg, dim=-1) / self.temperature for neg in negatives], dim=1)  # [512, num_negatives, 50]

        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [512, num_negatives+1, 50]

        
        labels = torch.zeros(logits.size(0), logits.size(2), dtype=torch.long, device=anchor.device)  # [512, 50]


        logits = logits.permute(0, 2, 1).reshape(-1, logits.size(1))  # [512*50, num_negatives+1]
        labels = labels.reshape(-1)  # [512*50]

        loss = F.cross_entropy(logits, labels)
        
        return loss



class CombinedLossI(nn.Module):
    def __init__(self, device, supervised_weight=1, distillation_weight=0.01, embed_weight=1.0, max_step=50):
        super(CombinedLossI, self).__init__()
        self.device = device

        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.005, diameter=1, reach=500, backend="tensorized")
        self.sig = nn.Sigmoid()


        self.supervised_loss_fn = nn.BCELoss()

        self.infonce_loss_fn = InfoNCELoss(temperature=0.5)

        self.supervised_weight = supervised_weight
        self.distillation_weight = distillation_weight
        self.embed_weight = embed_weight

        self.max_step = max_step
        self.q = C.NUM_OF_QUESTIONS
        self.sigmoid = nn.Sigmoid()

    def forward(self, logit_c, logit_t, logit_ensemble, logit_teacher_c, logit_teacher_t, logit_teacher_ensemble, 
                out_h_student, out_h_teacher, out_d_student, out_d_teacher, batch):


        T = 0.5

        p_c = logit_c / T
        p_t = logit_t / T
        p_enm = logit_ensemble / T

        p_teacher_c = logit_teacher_c / T
        p_teacher_t = logit_teacher_t / T
        p_teacher_enm = logit_teacher_ensemble / T

        loss_kd = torch.zeros(1, device=self.device)
        loss_kd += self.sinkhorn_loss(p_c.view(p_c.size(0), -1), p_teacher_c.view(p_teacher_c.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_t.view(p_t.size(0), -1), p_teacher_t.view(p_teacher_t.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_enm.view(p_enm.size(0), -1), p_teacher_enm.view(p_teacher_enm.size(0), -1))


        loss_supervised = torch.tensor(0.0, device=self.device)
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        p_c = self.sig(logit_c)
        p_t = self.sig(logit_t)
        p_enm = self.sig(logit_ensemble)

        for student in range(batch.shape[0]):
            delta = batch[student][:, :self.q] + batch[student][:, self.q:]

            a = (((batch[student][:, 0:self.q] -
                   batch[student][:, self.q:]).sum(1) + 1) //
                 2)[1:]  # [49]
            temp_c = p_c[student][:self.max_step - 1].mm(delta[1:].t())
            temp_t = p_t[student][:self.max_step - 1].mm(delta[1:].t())
            temp_enm = p_enm[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            pc = temp_c.gather(0, index)[0]
            pt = temp_t.gather(0, index)[0]
            penm = temp_enm.gather(0, index)[0]
            for i in range(len(pc) - 1, -1, -1):
                if pc[i] > 0:
                    pc = pc[:i + 1]
                    pt = pt[:i + 1]
                    penm = penm[:i + 1]
                    a = a[:i + 1]
                    break
            loss_supervised = loss_supervised+ self.supervised_loss_fn(pt, a) + self.supervised_loss_fn(pc, a)+ self.supervised_loss_fn(penm, a)
            p_mean = (pt + pc + penm)/3.0
            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])


        loss_embed = self.infonce_loss_fn(out_h_student, out_h_teacher, [out_d_student, out_d_teacher])


        print(self.supervised_weight * loss_supervised)
        print(self.distillation_weight * loss_kd)
        print(self.embed_weight * loss_embed)
        total_loss = (self.supervised_weight * loss_supervised + 
                      self.distillation_weight * loss_kd + 
                      self.embed_weight * loss_embed)

        return total_loss

class CombinedLossID(nn.Module):
    def __init__(self, device, supervised_weight=1, distillation_weight=0.1, embed_weight=1.0, max_step=50):
        super(CombinedLossID, self).__init__()
        self.device = device

        self.sinkhorn_loss = SamplesLoss("sinkhorn", p=2, blur=0.005, diameter=1, reach=500, backend="tensorized")
        self.sig = nn.Sigmoid()


        self.supervised_loss_fn = nn.BCELoss()

        self.infonce_loss_fn = InfoNCELoss(temperature=0.5)

        self.supervised_weight = supervised_weight
        self.distillation_weight = distillation_weight
        self.embed_weight = embed_weight

        self.max_step = max_step
        self.q = C.NUM_OF_QUESTIONS
        self.sigmoid = nn.Sigmoid()

    def forward(self, logit_c, logit_t, logit_ensemble, logit_teacher_c, logit_teacher_t, logit_teacher_ensemble, 
                out_h_student, out_h_teacher, out_d_student, out_d_teacher, batch):


        T = 0.5

        p_c = logit_c / T
        p_t = logit_t / T
        p_enm = logit_ensemble / T

        p_teacher_c = logit_teacher_c / T
        p_teacher_t = logit_teacher_t / T
        p_teacher_enm = logit_teacher_ensemble / T

        loss_kd = torch.zeros(1, device=self.device)
        loss_kd += self.sinkhorn_loss(p_c.view(p_c.size(0), -1), p_teacher_c.view(p_teacher_c.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_t.view(p_t.size(0), -1), p_teacher_t.view(p_teacher_t.size(0), -1))
        loss_kd += self.sinkhorn_loss(p_enm.view(p_enm.size(0), -1), p_teacher_enm.view(p_teacher_enm.size(0), -1))


        loss_supervised = torch.tensor(0.0, device=self.device)
        prediction = torch.tensor([], device=self.device)
        ground_truth = torch.tensor([], device=self.device)

        p_c = self.sig(logit_c)
        p_t = self.sig(logit_t)
        p_enm = self.sig(logit_ensemble)

        for student in range(batch.shape[0]):
            delta = batch[student][:, :self.q] + batch[student][:, self.q:]

            a = (((batch[student][:, 0:self.q] -
                   batch[student][:, self.q:]).sum(1) + 1) //
                 2)[1:]  # [49]
            temp_c = p_c[student][:self.max_step - 1].mm(delta[1:].t())
            temp_t = p_t[student][:self.max_step - 1].mm(delta[1:].t())
            temp_enm = p_enm[student][:self.max_step - 1].mm(delta[1:].t())
            index = torch.tensor([[i for i in range(self.max_step - 1)]],
                                 dtype=torch.long, device=self.device)
            pc = temp_c.gather(0, index)[0]
            pt = temp_t.gather(0, index)[0]
            penm = temp_enm.gather(0, index)[0]
            for i in range(len(pc) - 1, -1, -1):
                if pc[i] > 0:
                    pc = pc[:i + 1]
                    pt = pt[:i + 1]
                    penm = penm[:i + 1]
                    a = a[:i + 1]
                    break
            loss_supervised = loss_supervised+ self.supervised_loss_fn(pt, a) + self.supervised_loss_fn(pc, a)+ self.supervised_loss_fn(penm, a)
            p_mean = (pt + pc + penm)/3.0
            prediction = torch.cat([prediction, p_mean])
            ground_truth = torch.cat([ground_truth, a])


        loss_embed = self.infonce_loss_fn(out_h_student, out_h_teacher, [out_d_student, out_d_teacher])



        total_loss = (self.supervised_weight * loss_supervised + 
                      self.distillation_weight * loss_kd + 
                      self.embed_weight * loss_embed)

        return loss_supervised, loss_kd, loss_embed