import os
import collections

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from utils import check_cache, load_cache, save_cache, load_file, load_sent
from transformers import BertTokenizer
import spacy

from config import args

bert_tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased', do_lower_case=True)


def tokenize_word(sent):
    return sent.split()


def tokenize_subword(sent):
    return bert_tokenizer.tokenize(sent)


def load_data(
    corpus_path,
    max_len,
    batch_size,
    clip_count,
    vocab=None
):
    if vocab is None:
        vocab = build_vocab(corpus_path, clip_count)
    ds = BiasDataset(vocab, corpus_path, max_len)
    return DataLoader(ds, batch_size, shuffle=True)


def build_vocab(args, tokenizer):
    vocab = collections.Counter()
    df = pd.read_csv(args.train_path, sep="\t")
    for i, row in df.iterrows():
        tokens = tokenizer(load_sent(row[0], -1))
        vocab.update(tokens)
    words = ['<pad>', '<unk>', '<bos>', '<eos>'] + list(sorted(vocab))

    return (
        words,
        {w: i for i, w in enumerate(words)}
    )


class BiasDataset(Dataset):
    def __init__(self, args, ds_path, with_loss=False, with_index=False):
        self.tokenizer = self._get_tokenizer(args)
        self.with_index = with_index
        self.df = pd.read_csv(ds_path, sep="\t\t",
                              engine="python", header=None)
        df_train = self.df
        print(len(self.df))
        df_class_0 = df_train[df_train.iloc[:, 1] == 0]
        df_class_1 = df_train[df_train.iloc[:, 1] == 1]
        # # if not balanced, do undersampling.
        # if len(df_class_0) != len(df_class_1) and "train" in ds_path:
        #     print("balance dataset")
        #     count_class_1 = len(df_class_1)
        #     df_class_0_under = df_class_0.sample(count_class_1)
        #     df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)
        #     self.df = df_test_under

        self.max_len = args.max_len
        self.use_bert_encoder = (args.model == 'bert')
        if not self.use_bert_encoder:
            self.vocab, self.enc = build_vocab(args, self.tokenizer)
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.unk_idx = args.unk_idx
        self.with_loss = with_loss
        self._cache = {}

    def _get_tokenizer(self, args):
        if args.tokenizer == "word":
            return tokenize_word
        else:
            return tokenize_subword

    def _pad(self, vec, x):
        return np.pad(vec, (0, x), 'constant', constant_values=self.pad_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx not in self._cache:

            entry = self.df.iloc[idx]
            sent, label = entry[0], (entry[1])

            if self.with_loss:  # if there are losses.
                loss = entry[2]

            tokens = self.tokenizer(sent)[:self.max_len]

            if self.use_bert_encoder:
                token_ids = bert_tokenizer.encode(
                    tokens, add_special_tokens=True
                )
            else:
                token_ids = [self.enc.get(x, self.unk_idx) for x in tokens]
            token_ids = self._pad(token_ids, self.max_len - len(tokens))

            x = torch.from_numpy(token_ids).long().to(self.device)
            y = torch.zeros(2)
            y[label] = 1.
            y = y.to(self.device)

            self._cache[idx] = (x, y)

            if self.with_loss:
                z = torch.tensor(loss).float()
                self._cache[idx] = (x, y, z)

            if self.with_index:
                self.cache[idx] = (x, y, idx)

        return self._cache[idx]


class BiasDataset_multilabel(Dataset):
    def __init__(self, args, ds_path, with_loss=False, with_index=False):
        self.tokenizer = self._get_tokenizer(args)
        self.with_index = with_index
        self.df = pd.read_csv(ds_path, sep="\t\t",
                              engine="python", header=None)
        df_train = self.df
        print(len(self.df))

        self.max_len = args.max_len
        self.use_bert_encoder = (args.model == 'bert')
        if not self.use_bert_encoder:
            self.vocab, self.enc = build_vocab(args, self.tokenizer)
        self.device = args.device
        self.pad_idx = args.pad_idx
        self.unk_idx = args.unk_idx
        self.with_loss = with_loss
        self._cache = {}

    def _get_tokenizer(self, args):
        if args.tokenizer == "word":
            return tokenize_word
        else:
            return tokenize_subword

    def _pad(self, vec, x):
        return np.pad(vec, (0, x), 'constant', constant_values=self.pad_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx not in self._cache:

            entry = self.df.iloc[idx]
            sent, labels = entry[0], (entry[1])

            labels = labels.split("|")
            # print(labels)
            if self.with_loss:  # if there are losses.
                loss = entry[2]

            tokens = self.tokenizer(sent)[:self.max_len]

            if self.use_bert_encoder:
                token_ids = bert_tokenizer.encode(
                    tokens, add_special_tokens=True
                )
            else:
                token_ids = [self.enc.get(x, self.unk_idx) for x in tokens]
            token_ids = self._pad(token_ids, self.max_len - len(tokens))
            label_mapping = {'1': 1.0, '0': 0.0}
            labels = [label_mapping[y] for y in labels]
            x = torch.from_numpy(token_ids).long().to(self.device)
            y = torch.as_tensor(labels)

            y = y.to(self.device)

            self._cache[idx] = (x, y)

            if self.with_loss:
                z = torch.tensor(loss).float()
                self._cache[idx] = (x, y, z)

            if self.with_index:
                self.cache[idx] = (x, y, idx)

        return self._cache[idx]


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def create_loader(args, ds, shuffle=True):
    return DataLoader(ds, args.batch_size, shuffle)


def create_infinite_loader(args, ds, shuffle=True):
    return InfiniteDataLoader(ds, args.batch_size, shuffle)
