from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch


class OneHot(Dataset):
    def __init__(self, ques, ans):
        self.ques = torch.tensor(ques, dtype=torch.long)
        self.ans = torch.tensor(ans, dtype=torch.long)

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        return self.ques[index], self.ans[index]


class OneHotM(Dataset):
    def __init__(self, ques, ans):
        self.ques = torch.tensor(ques, dtype=torch.long)
        self.ans = torch.tensor(ans, dtype=torch.long)

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        return self.ques[index], self.ans[index]
