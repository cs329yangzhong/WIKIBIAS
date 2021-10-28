from data import CUDA
from args import ARGS
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from pytorch_transformers import AdamW, BertTokenizer
import sys
sys.path.append('.')


def is_ranking_hit(probs, labels, top=1):
    global ARGS

    # get rid of padding idx
    [probs, labels] = list(
        zip(*[(p, l) for p, l in zip(probs, labels) if l != ARGS.num_tok_labels - 1]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    print(probs_indices)
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    if sum([labels[i] for i in top_indices]) > 0:
        return 1
    else:
        return 0


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def tag_hits(logits, tok_labels, top=1):
    global ARGS

    probs = softmax(np.array(logits)[:, :, : ARGS.num_tok_labels - 1], axis=2)

    hits = [
        is_ranking_hit(prob_dist, tok_label, top=top)
        for prob_dist, tok_label in zip(probs, tok_labels)
    ]
    return hits


def tag_result(probs, labels, top=2):
    global ARGS
    # print(logits)
    [probs, labels] = list(
        zip(*[(p, l) for p, l in zip(probs, labels) if l != ARGS.num_tok_labels - 1]))
    probs_indices = list(zip(np.array(probs)[:, 1], range(len(labels))))
    # print(probs_indices)
    [_, top_indices] = list(zip(*sorted(probs_indices, reverse=True)[:top]))
    pred = [
        0. if index not in top_indices else 1. for index in range(len(labels))]
    return pred, labels
    # print(probs)
    # print(labels)
    # print()
    # if 1 in choice:
    #     print(choice)
    #     print(labels)
    # print(labels)
    # # probs = argsoftmax(np.array(logits)[:, :, : ARGS.num_tok_labels - 1], axis=2)
    # print(probs)
    # pred = [1 if y > threshold else 0 for y in probs]
    # print(pred)
    # print(labels)
