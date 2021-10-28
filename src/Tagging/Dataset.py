import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import string
from torch.utils.data import Dataset, DataLoader
import pickle
import collections


def tokenize_word(sent):
    return [y.lower() for y in sent.split()]


def readfile(filename):
    '''
    read a CONLL style file.
    '''
    f = open(filename, encoding="utf-8")
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART-') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []

    print("------------------------------------")
    print("Peek the data", data[0], data[0].count("B-bias"))
    print("Number of sentences in", filename, " : ", len(data))
    print("Average bias count per sentenece", np.mean(
        [pair[1].count("B-bias") for pair in data]))
    print("Max len of sentence has ", np.max([len(pair[0]) for pair in data]))
    print("------------------------------------")
    return data


def build_vocab(args):

    try:
        vocab = pickle.load(open("cache/vocab.pkl", "rb"))
        return vocab
    except:
        vocab = collections.Counter()
        sent_data = readfile(args.train_path)
        sents = [pair[0] for pair in sent_data]

        for line in sents:
            line = [y.lower() for y in line]
            vocab.update(line)
        words = ['<pad>', "<unk>", "<bos>", "<eos>"] + list(sorted(vocab))

        vocab_dump = pickle.dump((
            words,
            {w: i for i, w in enumerate(words)}
        ), open("cache/vocab.pkl", 'wb'))

        return (
            words,
            {w: i for i, w in enumerate(words)}
        )


def tag_mapping(args):
    '''
    Create dictionary and mapping of tags, sorted by frequency.
    '''
    try:
        tag = pickle.load(open("cache/tag.pkl", "rb"))
        return tag
    except:
        tag = collections.Counter()
        sent_data = readfile(args.train_path)
        sents = [pair[1] for pair in sent_data]

        for line in sents:
            line = line
            tag.update(line)
        tag_list = ['<pad>'] + list(sorted(tag))

        tag_dump = pickle.dump((
            tag_list,
            {w: i for i, w in enumerate(tag_list)}
        ), open("cache/tag.pkl", 'wb'))

        print(tag_list)

        return (
            tag_list,
            {w: i for i, w in enumerate(tag_list)}
        )


class BiasDetectdataset(Dataset):
    def __init__(self, args, ds_path):
        self.pad_idx = args.pad_idx
        self.unk_idx = args.unk_idx
        self.tokenizer = self._get_tokenizer(args)
        self.vocab, self.enc = build_vocab(args)
        self.tag_to_id, self.id_to_tag = tag_mapping(args)
        self.device = args.device
        self._cache = {}
        self.max_len = args.max_len
        self.data = readfile(ds_path)

    def _get_tokenizer(self, args):
        if args.tokenizer == "word":
            return tokenize_word
        else:
            raise NotImplementedError("the requested tokenizer does not exist")

    def _pad(self, vec, x):
        return np.pad(vec, (0, x), 'constant', constant_values=self.pad_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self._cache:
            entry = self.data[idx]
            sent, labels = entry[0], entry[1]
            tokens = self.tokenizer(" ".join(sent))[:self.max_len]
            token_ids = [self.enc.get(x, self.unk_idx) for x in tokens]
            token_ids = self._pad(token_ids, self.max_len - len(tokens))

            label_ids = [self.id_to_tag.get(x, self.unk_idx)
                         for x in labels[:self.max_len]]
            label_ids = self._pad(label_ids, self.max_len - len(tokens))

            x = torch.from_numpy(token_ids).long().to(self.device)
            y = torch.from_numpy(label_ids).long().to(self.device)

            self._cache[idx] = (x, y)
        return self._cache[idx]


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train_path", type=str, default="data/train_conll.txt")
#     parser.add_argument('--pad_idx', type=int, default=0)
#     parser.add_argument('--unk_idx', type=int, default=1)
#     parser.add_argument('--tokenizer', type=str, default="word")
#     parser.add_argument('--device', type=str, default="cpu")
#     parser.add_argument('--max_len', type=int, default=128)
#     args = parser.parse_args()
#     a = BiasDetectdataset(args, args.train_path)
