from KnowledgeTracing.hgnn_models import HGNN
from KnowledgeTracing.Constant import Constants as C
import torch.nn as nn
import torch
from KnowledgeTracing.DirectedGCN.GCN import GCN


class DKT(nn.Module):
    def __init__(self, hidden_dim, layer_dim, G, adj_in, adj_out):
        super(DKT, self).__init__()
        emb_dim = C.EMB
        emb = nn.Embedding(2 * C.NUM_OF_QUESTIONS, emb_dim)
        self.ques = emb(torch.LongTensor([i for i in range(2 * C.NUM_OF_QUESTIONS)])).cuda()

        self.skill_graph = SkillGraphModule(emb_dim, hidden_dim, layer_dim, G)
        self.transition_graph = TransitionGraphModule(emb_dim, hidden_dim, layer_dim, adj_out, adj_in)
        self.kd_module = KDModule(hidden_dim)

    def forward(self, x):
        logit_c, out_h = self.skill_graph(self.ques, x)

        logit_t, out_d = self.transition_graph(self.ques, x)

        ensemble_logit = self.kd_module(out_h, out_d)

        return logit_c, logit_t, ensemble_logit, out_h, out_d



class KDModule(nn.Module):
    def __init__(self, hidden_dim):
        super(KDModule, self).__init__()
        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc_ensemble = nn.Linear(2 * hidden_dim, C.NUM_OF_QUESTIONS)

    def forward(self, out_h, out_d):

        theta = self.sigmoid(self.w1(out_h) + self.w2(out_d))

        out_d = theta * out_d.clone()  
        out_h = (1 - theta) * out_h.clone()

        ensemble_logit = self.fc_ensemble(torch.cat([out_d, out_h], -1))
        return ensemble_logit



class SkillGraphModule(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, G):
        super(SkillGraphModule, self).__init__()
        self.hgnn = HGNN(in_ch=emb_dim, n_hid=emb_dim, n_class=emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.G = G

    def forward(self, ques, x):
        with torch.autograd.set_detect_anomaly(True):
            ques_h = self.hgnn(ques, self.G)

            x_h = x.matmul(ques_h)

            out_h, _ = self.gru(x_h)

            logit_h = self.fc(out_h)
            return logit_h, out_h


class TransitionGraphModule(nn.Module):
    def __init__(self, emb_dim, hidden_dim, layer_dim, adj_out, adj_in):
        super(TransitionGraphModule, self).__init__()
        self.dgcn_out = GCN(nfeat=emb_dim, nhid=emb_dim, nclass=int(emb_dim / 2))
        self.dgcn_in = GCN(nfeat=emb_dim, nhid=emb_dim, nclass=int(emb_dim / 2))
        self.gru = nn.GRU(emb_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, C.NUM_OF_QUESTIONS)
        self.adj_out = adj_out
        self.adj_in = adj_in

    def forward(self, ques, x):
        with torch.autograd.set_detect_anomaly(True):
            ques_out = self.dgcn_out(ques, self.adj_out)
            ques_in = self.dgcn_in(ques, self.adj_in)

            ques_d = torch.cat([ques_in, ques_out], -1)

            x_d = x.matmul(ques_d)

            out_d, _ = self.gru(x_d)

            logit_d = self.fc(out_d)
            return logit_d, out_d