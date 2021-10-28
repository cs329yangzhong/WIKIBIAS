import pickle
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
import random

from models import CNN, BertMLP
from dataset import BiasDataset, create_loader
from config import args
from sklearn.metrics import precision_recall_fscore_support


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


set_seed(args.rand_seed)
# model.


def select_model(args):
    model = None
    if args.model == "CNN":
        model = CNN
    elif args.model == "bert":
        model = BertMLP
    return model(args).to(args.device)


def _grad_step(args, i):
    return args.grad_step == -1 or i % args.grad_step == 0


class Logger:

    def __init__(self, name):
        self.name = name
        self.data = collections.defaultdict(list)

    def log(self, k, v):
        self.data[k].append(v)


def test(args):
    # model.load_state_dict(torch.load(args.ckpt))
    model.eval()
    test_loader = create_loader(args, test_ds, shuffle=False)
    y_true, y_pred = [], []
    for i, (inputs, labels) in enumerate(test_loader):
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            if args.model == "CNN":
                pred = model(False, inputs)
            else:
                pred = model(inputs)

        pred = (pred.argmax(dim=1)).cpu().numpy()

        y_true.append(torch.max(labels, 1)[1].cpu().numpy())
        y_pred.append(pred)

    y_true_flatten = [item for sublist in y_true for item in sublist]
    y_pred_flatten = [item for sublist in y_pred for item in sublist]

    p, r, f1, _ = precision_recall_fscore_support(
        y_true_flatten, y_pred_flatten, average='binary')
    with open(args.eval_on + "out", "w") as fin:
        for label in y_pred_flatten:
            fin.write(str(label) + "\n")
    return classification_report(y_true_flatten, y_pred_flatten, digits=4), (p, r, f1)


if __name__ == '__main__':

    test_tsv = "../../data/class_binary/test.tsv"
    train_tsv = "../../data/class_binary/train.tsv"
    ckpt_name = args.eval_ckpt

    all_p = []
    all_r = []
    all_f1 = []

    test_tsv = 'data/class_test_1030.tsv'
    logger = Logger(args.name)
    args.train_path = train_tsv

    # Setup datasets.
    test_ds = BiasDataset(args, test_tsv)
    train_ds = BiasDataset(args, train_tsv)

    # Setup vocab.
    if args.model == "bert":
        args.n_emb = 30522
    else:
        args.n_emb = len(train_ds.enc)

    # Set up training.
    model = select_model(args)
    model = model.to(args.device)
    ckpt_path = 'ckpt/%s' % ckpt_name

    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print("Test on ...", test_tsv)
    report, P_R_F1 = test(args)
    all_p.append(P_R_F1[0])
    all_r.append(P_R_F1[1])
    all_f1.append(P_R_F1[2])
    print(report)
    print(P_R_F1)
