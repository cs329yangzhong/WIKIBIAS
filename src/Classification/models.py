import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import numpy as np


from transformers import BertModel


class LR(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)
        self.fc = nn.Linear(args.emb_dim, args.class_num)

    def forward(self, x):
        emb = self.embedding(x)
        rep = torch.mean(emb, 1)  # [B, L, H_e]
        logits = self.fc(rep)
        return logits


class CNN(nn.Module):
    def __init__(self, args):

        super(CNN, self).__init__()

        self.args = args
        self.pooling_type = args.pooling_type
        # V = args.embed_num
        D = 300
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = [3, 4, 5]

        self.embedding = nn.Embedding(args.n_emb, args.emb_dim)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)
        self.noise_layer = nn.Linear(C, C)
        self.noise_layer.weight.data.copy_(torch.eye(2))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)

        if self.pooling_type == "MAX":
            x, _ = torch.max(x, dim=2)
        elif self.pooling_type == "MEAN":
            x = torch.mean(x, dim=2)
        elif self.pooling_type == "LOGSUMEXP":
            x = torch.logsumexp(x, dim=2)
        return x

    def forward(self, noise, x):
        x = self.embedding(x)

        # if self.args.static:
        #     x = Variable(x)
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)
        logit = self.fc1(x)

        if noise:
            return self.noise_layer(logit)
        else:
            return logit


class BertMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bert_model)
        # for name, param in self.bert.named_parameters():
        #     if name.startswith('embeddings'):
        #         param.requires_grad = False
        self.fc = nn.Linear(args.n_bert_hid, args.class_num)
        # self.linear = nn.Sequential(nn.Linear(768, 128),
        #                           nn.Tanh(),
        #                          nn.Linear(128, args.class_num))

    def forward(self, x):
        mask = (x != 0).float()
        emb, _ = self.bert(x, attention_mask=mask)

        rep = emb[:, 0, :]
        logits = self.fc(rep)
        return logits


class BertMLP_with_loss_fn(nn.Module):
    def __init__(self, args, loss_fn):
        super().__init__()
        self.device = args.device
        self.bert = BertModel.from_pretrained(args.bert_model)
        # for name, param in self.bert.named_parameters():
        #     if name.startswith('embeddings'):
        #         param.requires_grad = False
        self.fc = nn.Linear(args.n_bert_hid, args.class_num)
        self.loss_fn = loss_fn
        # self.linear = nn.Sequential(nn.Linear(768, 128),
        #                           nn.Tanh(),
        #                          nn.Linear(128, args.class_num))

    def forward(self, x, teacher_prob=None, labels=None):
        mask = (x != 0).float()
        emb, _ = self.bert(x, attention_mask=mask)

        rep = emb[:, 0, :]
        logits = self.fc(rep)

        if labels is None:
            return logits
        loss = self.loss_fn.forward(
            logits, 0, teacher_prob, labels, device=self.device)
        return logits, loss
