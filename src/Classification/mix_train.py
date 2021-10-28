import pickle
import collections

import torch
import torch.nn as nn
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
    # random.seed(seed)
    np.random.seed(seed)

# set_seed(args.rand_seed)
# model.


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


def train(epoch, train_ds):
    device = torch.device("cuda:%s" %
                          args.device if torch.cuda.is_available() else "cpu")
    train_loader = create_loader(args, train_ds, shuffle=True)
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

    #print(y_true_flatten, y_pred_flatten)
#                    return classification_report(y_true_flatten, y_pred_flatten, digits=4)
    acc = sum(1 for x, y in zip(y_true_flatten, y_pred_flatten)
              if x == y) / len(y_true_flatten)
    confusion_tuples = perf_measure(y_true_flatten, y_pred_flatten)
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

    #print(y_true_flatten, y_pred_flatten)
    return classification_report(y_true_flatten, y_pred_flatten, digits=4, output_dict=True)


if __name__ == '__main__':

    # set random seed.
    rand_seed = random.choice(range(1, 1000))
    set_seed(rand_seed)
    args.rand_seed = rand_seed
    print(args.rand_seed)

    # train, dev, test loader.
    train_tsv = "data/train_%s_all.tsv" % args.train_set
    train_tsv = "data/train_classification_data_0402.tsv"
    train_tsv_2 = "data/class_train_0820.tsv"
    #train_tsv = "data/train_less_5_biased.tsv"
    #train_tsv = "data/class_dev_0820.tsv"
    #train_tsv = "data/test.tsv"
    dev_tsv = "data/class_dev_0820.tsv"
    test_tsv = "data/class_dev_0820.tsv"

    args.train_path = train_tsv
    # Setup datasets.
    train_ds = BiasDataset(args, train_tsv)
    valid_ds = BiasDataset(args, dev_tsv)
    test_ds = BiasDataset(args, test_tsv)
    train_ds_2 = BiasDataset(args, train_tsv_2)

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

    log_file = "logs/%s_%s_MixtrainedOn_%s_epoch%s.csv" % (
        args.model, args.train_set, "clean" if "class" in train_tsv else "noisy", args.epochs)
    if args.pretrained_ckpt != "":
        log_file = log_file + "further_tune"
    with open(log_file, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(['epoch', 'train data', 'train loss',
                           'valid loss', 'valid acc', "valid_P_R_F1", 'training time'])
    for epoch in range(1, args.epochs*2+4):
        start_time = time.time()
        if epoch > args.epochs*2 + 1:
            train_loss = train(args, train_ds=train_ds_2)
            training_dataset = "clean"

        elif epoch % 2 == 0:
            train_loss = train(args, train_ds=train_ds)
            training_dataset = "noisy"
        else:
            training_dataset = "clean"
            train_loss = train(args, train_ds=train_ds_2)
        valid_loss, valid_acc, valid_P_R_F1 = valid(args)
        end_time = time.time()
        print(end_time - start_time)
        vl_history.append(valid_loss < best_loss)
        with open(log_file, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')

            logwriter.writerow([epoch, training_dataset, train_loss, valid_loss, valid_acc, str(
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
        f1 = report['1']['f1-score']

        if valid_loss < best_loss or f1 > best_f1:
            best_f1 = f1
            best_loss = valid_loss
            args.ckpt = "ckpt/model_mix_%s_%s_trainedOn_%s_seed%s_epoch%s.pt" % (
                args.model, args.train_set, "noisy" if "class" not in train_tsv else "clean", args.rand_seed, args.epochs)
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
