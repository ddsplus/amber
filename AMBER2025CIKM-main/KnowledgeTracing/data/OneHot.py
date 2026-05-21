from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch


class OneHot(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans
        self.numofques = C.NUM_OF_QUESTIONS
        self.labels = self._build_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]

    def _build_labels(self):
        n = len(self.ques)
        labels = torch.zeros(n, C.MAX_STEP, 2 * self.numofques, dtype=torch.float32)
        for idx in range(n):
            questions = self.ques[idx]
            answers = self.ans[idx]
            for i in range(C.MAX_STEP):
                q = int(questions[i])
                a = int(answers[i])
                if q <= 0:
                    continue
                if a > 0:
                    labels[idx, i, q - 1] = 1.0
                elif a == 0:
                    labels[idx, i, self.numofques + q - 1] = 1.0
        return labels

    def onehot(self, questions, answers):
        label = torch.zeros(C.MAX_STEP, 2 * self.numofques)
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                label[i][questions[i]-1] = 1
            elif answers[i] == 0:
                label[i][self.numofques + questions[i]-1] = 1
        return label


class OneHotM(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans
        self.numofques = C.NUM_OF_QUESTIONS
        self.labels = self._build_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.labels[index]

    def _build_labels(self):
        n = len(self.ques)
        labels = torch.zeros(n, C.MAX_STEP, 2 * self.numofques, dtype=torch.float32)
        for idx in range(n):
            questions = self.ques[idx]
            answers = self.ans[idx]
            for i in range(C.MAX_STEP):
                q = int(questions[i])
                a = int(answers[i])
                if q <= 0:
                    continue
                if a > 0:
                    labels[idx, i, q - 1] = 1.0
                elif a == 0:
                    labels[idx, i, self.numofques + q - 1] = 1.0
        return labels

    def onehot(self, questions, answers):
        label = torch.zeros(C.MAX_STEP, 2 * self.numofques)
        for i in range(C.MAX_STEP):
            if answers[i] > 0:
                label[i][questions[i]-1] = 1
            elif answers[i] == 0:
                label[i][self.numofques + questions[i]-1] = 1
        return label
