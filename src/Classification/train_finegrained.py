import pickle
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import random

from models import CNN, LR,  BertMLP
from dataset import BiasDataset, BiasDataset_multilabel, create_loader
from config import args
import time
import csv

# SEED.


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

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


class FocalLoss2d(nn.modules.loss._WeightedLoss):

    def __init__(self, gamma=2, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='mean', balance_param=0.25):
        super(FocalLoss2d, self).__init__(
            weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.balance_param = balance_param

    def forward(self, input, target):

        # inputs and targets are assumed to be BatchxClasses
        assert len(input.shape) == len(target.shape)
        assert input.size(0) == target.size(0)
        assert input.size(1) == target.size(1)

        weight = Variable(self.weight)

        # compute the negative likelyhood
        logpt = - F.binary_cross_entropy_with_logits(
            input, target, pos_weight=weight, reduction=self.reduction)
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1-pt)**self.gamma) * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


def train(epoch, loss='ASYM'):
    device = torch.device("cuda:%s" %
                          args.device if torch.cuda.is_available() else "cpu")
    train_loader = create_loader(args, train_ds, shuffle=True)
    print('start train')

    if loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "ASYM":
        criterion = AsymmetricLoss()

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

        loss = criterion(out, labels)

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
        loss = criterion(logits, labels)
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
        loss = criterion(logits, labels)
        dev_loss += loss.item()

        y_true.extend(labels.cpu().detach().numpy().tolist())
        y_pred.extend(torch.sigmoid(logits).cpu().detach().numpy().tolist())
    dev_loss /= len(valid_loader)
    # print( y_true, y_pred)

    targets, outputs = y_true, np.array(y_pred) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    return dev_loss, accuracy, f1_score_micro,  f1_score_macro


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

        y_true.extend(labels.cpu().detach().numpy().tolist())
        y_pred.extend(torch.sigmoid(pred).cpu().detach().numpy().tolist())

    targets, outputs = y_true, np.array(y_pred) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    prf1 = precision_recall_fscore_support(
        targets, outputs, beta=0.5, average=None)
    print(prf1)
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    #print(y_true_flatten, y_pred_flatten)
    return accuracy, f1_score_micro,  f1_score_macro, classification_report(targets, outputs, target_names=['F-bias', 'E-bias', 'D-bias'])


class FocalLoss(object):

    def __call__(self, y_pred, y_true, gamma=2, alpha=0.25):
        self._gamma = gamma
        self._alpha = alpha
        m = nn.Sigmoid()

        self.criterion = nn.BCELoss()
        cross_entropy_loss = self.criterion(m(y_pred), y_true)
        # print(cross_entropy_loss)
        p_t = ((y_true * y_pred) +
               ((1 - y_true) * (1 - y_pred)))
        modulating_factor = 1.0
        if self._gamma:
            modulating_factor = torch.pow(1.0 - p_t, self._gamma)
        alpha_weight_factor = 1.0
        if self._alpha is not None:
            alpha_weight_factor = (y_true * self._alpha +
                                   (1 - y_true) * (1 - self._alpha))
        focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                    cross_entropy_loss)
        return focal_cross_entropy_loss.mean()


if __name__ == '__main__':

    # set random seed.
    rand_seed = random.choice(range(1, 1000))
    set_seed(rand_seed)
    args.rand_seed = rand_seed
    print(args.rand_seed)

    # train, dev, test loader.
    test_tsv = "../../data/class_finegrained/test.tsv"
    train_tsv = "../../data/class_finegrained/train.tsv"
    dev_tsv = "../../data/class_finegrained/dev.tsv"

    args.train_path = train_tsv
    # Setup datasets.
    train_ds = BiasDataset_multilabel(args, train_tsv)
    valid_ds = BiasDataset_multilabel(args, dev_tsv)
    test_ds = BiasDataset_multilabel(args, test_tsv)

    # Setup vocab.
    if args.model == "bert":
        args.n_emb = 30522
    else:
        args.n_emb = len(train_ds.enc)

    # Set up training.
    args.class_num = 3
    model = select_model(args)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.wd
    )
    criterion = nn.BCEWithLogitsLoss()
    logger = Logger(args.name)
    best_loss = np.inf
    vl_history = []
    best_f1 = np.inf

    # Setup pre-trained weights.
    if args.model == "bert" and args.pretrained_ckpt != '':
        prev_state_dict = torch.load(args.pretrained_ckpt, map_location='cpu')
        for n, p in model.named_parameters():
            if (
               n in prev_state_dict
               and n != 'fc.weight'
               and n != 'fc.bias'
               ):
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

    log_file = "logs/%s_%s_multilabel_trainedOn_%s_epoch%s.csv" % (
        args.model, args.bias_type, "clean" if "class" in train_tsv else "noisy", args.epochs)
    if args.pretrained_ckpt != "":
        log_file = log_file + "further_tune"
    if "less_5" in train_tsv:
        log_file = log_file + "less_5"
    if "less_10" in train_tsv:
        log_file = log_file + "less_10"

    with open(log_file, "w") as logfile:
        logwriter = csv.writer(logfile, delimiter=",")
        logwriter.writerow(['epoch', 'train loss', 'valid loss', 'valid acc',
                           'valid_f1_score_micro',  'valid_f1_score_macro', 'training time'])
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        train_loss = train(args)
        valid_loss, valid_accuracy, valid_f1_score_micro,  valid_f1_score_macro = valid(
            args)
        end_time = time.time()
        print(end_time - start_time)
        vl_history.append(valid_loss < best_loss)
        with open(log_file, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')

            logwriter.writerow([epoch, train_loss, valid_loss, valid_accuracy, str(
                valid_f1_score_micro),  str(valid_f1_score_macro), end_time-start_time])
        print(
            f'epoch: {epoch} | '
            f'train loss: {train_loss:.6f} | '
            f'valid loss: {valid_loss:.6f} | '
            f"{'*' if vl_history[-1] else ''}"
        )
        accuracy, f1_score_micro, f1_score_macro, report = test(args)
        print("EPOCH", epoch)
        print(report)
        valid_f1 = valid_f1_score_micro
        if valid_loss < best_loss or valid_f1 > best_f1:
            best_f1 = valid_f1
            best_loss = valid_loss
            args.ckpt = "ckpt/model_%s_%s_trainedOn_%s_seed%s_multilabel_Asym.pt" % (
                args.model, args.bias_type, "noisy" if "class" not in train_tsv else "clean", args.rand_seed)
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

    accuracy, f1_score_micro, f1_score_macro, prf1report = test(args)
    pickle.dump(logger.data, open(f'log_{args.name}.pkl', 'wb'))
