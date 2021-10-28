import pickle
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve, auc
import numpy as np
import random
from sklearn import metrics
from models import CNN, BertMLP
from dataset import BiasDataset, BiasDataset_multilabel, create_loader
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


def cal_roc_auc(all_labels, all_logits):
    num_labels = 3
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc


def train(epoch):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_loader = create_loader(args, train_ds)
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


def valid(args):
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


def test(args, write_file):
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

        cur_label = labels.cpu().detach().numpy().tolist()
        y_true.extend(cur_label)
        prediction = torch.sigmoid(pred).cpu().detach().numpy().tolist()
        prediction_value = (np.array(prediction) >= 0.5).astype(int)
        y_pred.extend(prediction)
        for i in range(inputs.size()[0]):
            write_file.write(
                str(cur_label[i]) + " |||" + str(prediction_value[i]) + "\n")
    # print(, np.array(y_pred.shape))
    roc_auc = cal_roc_auc(np.array(y_true), np.array(y_pred))
    targets, outputs = y_true, np.array(y_pred) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    prf1 = precision_recall_fscore_support(
        targets, outputs, beta=0.5, average=None)
    print(str(prf1))
    print(classification_report(targets, outputs))
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    #print(y_true_flatten, y_pred_flatten)
    return accuracy, f1_score_micro,  f1_score_macro, prf1, classification_report, roc_auc


def write_out(args, write_file):
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

        cur_label = labels.cpu().detach().numpy().tolist()
        y_true.extend(cur_label)
        prediction = torch.sigmoid(pred).cpu().detach().numpy().tolist()
        prediction_value = (np.array(prediction) >= 0.5).astype(int)
        y_pred.extend(prediction)
        for i in range(inputs.size()[0]):
            write_file.write(
                str(cur_label[i]) + " |||" + str(prediction_value[i]) + "\n")

    return accuracy, f1_score_micro,  f1_score_macro, prf1, classification_report, roc_auc


if __name__ == '__main__':

    test_tsv = "../../data/class_finegrained/test.tsv"
    train_tsv = "../../data/class_finegrained/train.tsv"
    ckpt_name = ""

    file1 = open("output_%s_out.txt" % ckpt_name, "w")
    logger = Logger(args.name)

    args.train_path = train_tsv

    if args.model == "bert":
        args.n_emb = 30522
    # Set up training.
    args.class_num = 3
    model = select_model(args)
    model = model.to(args.device)
    ckpt_path = 'ckpt/%s' % ckpt_name
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    print("Test on ...", test_tsv)
    accuracy, f1_score_micro,  f1_score_macro, prf1, classification_report_, roc_auc = test(
        args, file1)
    print(classification_report_)
