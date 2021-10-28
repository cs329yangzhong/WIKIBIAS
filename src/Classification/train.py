import pickle
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
import random

from models import CNN, LR,  BertMLP
from dataset import BiasDataset, create_loader
from config import args
import time
import csv

# SEED.


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def select_model(args):
    model = None
    if args.model == "CNN":

        imodel = CNN
    elif args.model == "LR":
        model = LR
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


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.redution == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target):
        if self.weight is not None:
            self.weight = self.weight.to(pred.device)

        if self.training:
            n = preds.size(-1)
            log_pred = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_pred.sum(dim=-1))
            nll = F.nll_loss(log_pred, target,
                             reduction=self.reduction, weight=self.weight)
            return self.linear_combination(loss/n, nll)
        else:
            return torch.nn.functional.cross_entropy(preds, target, weight=self.weight)


def train(epoch):
    device = torch.device("cuda:%s" %
                          args.device if torch.cuda.is_available() else "cpu")
    train_loader = create_loader(args, train_ds, shuffle=True)
    train_loss = 0.
    global_steps = 0
    optimizer.zero_grad()

    loss_func = nn.CrossEntropyLoss()

    for i, a in enumerate(train_loader):
        model.train()
        inputs, labels = a
        optimizer.zero_grad()
        # CE loss.
        if args.model == "CNN":
            out = model(True, inputs)
        else:
            out = model(inputs)
        loss = loss_func(out, torch.max(labels, 1)[1])
        train_loss += (loss.item())
        loss.backward()

        if _grad_step(args, i):
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1

        logger.log("Loss: ", loss.item())
    print("Training Loss: ", train_loss / global_steps)
    return train_loss / global_steps


def self_paced_train(epoch, loss_cut_off):
    device = torch.device("cuda:%s" %
                          args.device if torch.cuda.is_available() else "cpu")
    train_loader = create_loader_withloss(args, train_ds, shuffle=True)
    train_loss = 0.
    global_steps = 0
    optimizer.zero_grad()

    for i, a in enumerate(train_loader):
        model.train()
        inputs, labels = a
        optimizer.zero_grad()
        # CE loss.
        if args.model == "CNN":
            out = model(True, inputs)
        else:
            out = model(inputs)

        loss = criterion(out, torch.max(labels, 1)[1])
        train_loss += (loss.item())
        loss.backward()

        if _grad_step(args, i):
            optimizer.step()
            optimizer.zero_grad()
            global_steps += 1

        logger.log("Loss: ", loss.item())
    print("Training Loss: ", train_loss / global_steps)
    return train_loss / global_steps


def valid_(args):
    model.eval()
    valid_loader = create_loader(args, valid_ds)
    dev_loss = 0.
    for i, (inputs, labels) in enumerate(valid_loader):
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            if args.model == "CNN":
                logits = model(True, inputs)
            else:
                logits = model(inputs)
        loss = criterion(logits, torch.max(labels, 1)[1])
        dev_loss += loss.item()

    dev_loss /= len(valid_loader)

    return dev_loss


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] == 0:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    return (TP, FP, TN, FN)


def valid(args):
    model.eval()
    valid_loader = create_loader(args, valid_ds)
    dev_loss = 0.
    y_true = []
    y_pred = []
    for i, (inputs, labels) in enumerate(valid_loader):
        if args.cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        with torch.no_grad():
            if args.model == "CNN":
                logits = model(True, inputs)
            else:
                logits = model(inputs)
        loss = criterion(logits, torch.max(labels, 1)[1])
        dev_loss += loss.item()
        pred = (logits.argmax(dim=1)).cpu().numpy()
        y_true.append(torch.max(labels, 1)[1].cpu().numpy())
        y_pred.append(pred)
    dev_loss /= len(valid_loader)
    y_true_flatten = [item for sublist in y_true for item in sublist]
    y_pred_flatten = [item for sublist in y_pred for item in sublist]

    acc = sum(1 for x, y in zip(y_true_flatten, y_pred_flatten)
              if x == y) / len(y_true_flatten)
    confusion_tuples = classification_report(
        y_true_flatten, y_pred_flatten, digits=4, output_dict=True)
    return dev_loss, acc, confusion_tuples


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

    return classification_report(y_true_flatten, y_pred_flatten, digits=4, output_dict=True)


if __name__ == '__main__':

    # set random seed.
    rand_seed = random.choice(range(1, 1000))
    set_seed(rand_seed)
    args.rand_seed = rand_seed
    print(args.rand_seed)

    # train, dev, test loader.
    test_tsv = "../../data/class_binary/test.tsv"
    train_tsv = "../../data/class_binary/train.tsv"
    dev_tsv = "../../data/class_binary/dev.tsv"

    args.train_path = train_tsv
    # Setup datasets.
    train_ds = BiasDataset(args, train_tsv)
    valid_ds = BiasDataset(args, dev_tsv)
    test_ds = BiasDataset(args, test_tsv)

    # Setup vocab.
    if args.model == "bert":
        args.n_emb = 30522
    else:
        args.n_emb = len(train_ds.enc)

    # Set up training.
    model = select_model(args)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    criterion = nn.CrossEntropyLoss()
    logger = Logger(args.name)
    best_loss = np.inf
    vl_history = []
    best_f1 = np.inf

    # Setup pre-trained weights.
    if args.model == "bert" and args.pretrained_ckpt != '':
        prev_state_dict = torch.load(args.pretrained_ckpt, map_location='cpu')
        for n, p in model.named_parameters():
            # if (
            #   n in prev_state_dict
            #  and n != 'fc.weight'
            # and n != 'fc.bias'
            # ):
            w = prev_state_dict[n]
            p.data.copy_(w.data)
        model = model.to(args.device)
        print("Load pretrained noisy model")

    # Setup embeddings (optional).
    if args.model != 'bert':
        glove = pickle.load(open("cache/glove.pkl", 'rb'))
        embs = model.embedding.weight.clone()
        found = 0
        for i, word in enumerate(train_ds.vocab):
            if word in glove:
                embs[i] = torch.tensor(glove[word])
                found += 1
        model.embedding.weight.data.copy_(embs)
        model = model.to(args.device)

    log_file = "logs/%s_%s_trainedOn_%s_epoch%s" % (
        args.model, args.train_set, "clean" if "class" in train_tsv else "noisy", args.epochs)
    if args.pretrained_ckpt != "":
        log_file = log_file + "further_tune"
    if "less_5" in train_tsv:
        log_file = log_file + "less_5"
    if "less_10" in train_tsv:
        log_file = log_file + "less_10"

    with open(log_file, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(['epoch', 'train loss', 'valid loss',
                           'valid acc',  'valid_P_R_F1', 'training time'])
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss = train(args)
        valid_loss, valid_acc, valid_P_R_F1 = valid(args)
        end_time = time.time()
        print(end_time - start_time)
        vl_history.append(valid_loss < best_loss)
        with open(log_file, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')

            logwriter.writerow([epoch, train_loss, valid_loss, valid_acc, str(
                valid_P_R_F1), end_time-start_time])
        print(
            f'epoch: {epoch} | '
            f'train loss: {train_loss:.6f} | '
            f'valid loss: {valid_loss:.6f} | '
            f"{'*' if vl_history[-1] else ''}"
        )
        report = test(args)
        print("EPOCH", epoch)
        print(report)
        valid_f1 = valid_P_R_F1['1']['f1-score']
        if valid_loss < best_loss or valid_f1 > best_f1:
            best_f1 = valid_f1
            best_loss = valid_loss
            args.ckpt = "ckpt/model_%s_%s_trainedOn_%s_seed%s_CE.pt" % (
                args.model, args.train_set, "noisy" if "class" not in train_tsv else "clean", args.rand_seed)
            if args.pretrained_ckpt != "":
                args.ckpt = args.ckpt + "further_tune"
            if "all" in train_tsv:
                args.ckpt = args.ckpt + "all_data"
            if "less_5" in train_tsv:
                args.ckpt = args.ckpt + "less_5"
            if "less_10" in train_tsv:
                args.ckpt = args.ckpt + "less_10"
            torch.save(model.state_dict(), args.ckpt)

            print('* saved')

    report = test(args)
    logger.log('test', report)
    print(report)
    pickle.dump(logger.data, open(f'log_{args.name}.pkl', 'wb'))
