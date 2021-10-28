from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
from tag_baseline.add_tags import *
from tag_baseline import features

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule, PretrainedConfig)

from pytorch_transformers import BertModel as BertModel
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import classification_report as cls_report

from seqeval.metrics import classification_report

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class ConcatCombine(nn.Module):
    def __init__(self, hidden_size, feature_size, out_size, layers,
                 dropout_prob, small=False, pre_enrich=False, activation=False,
                 include_categories=False, category_emb=False,
                 add_category_emb=False, CUDA=(torch.cuda.device_count() > 0)):
        super(ConcatCombine, self).__init__()

        if layers == 1:
            self.out = nn.Sequential(
                nn.Linear(hidden_size + feature_size, out_size),
                nn.Dropout(dropout_prob))
        elif layers == 2:
            waist_size = min(hidden_size, feature_size) if small else max(
                hidden_size, feature_size)
            if activation:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.ReLU(),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
            else:
                self.out = nn.Sequential(
                    nn.Linear(hidden_size + feature_size, waist_size),
                    nn.Dropout(dropout_prob),
                    nn.Linear(waist_size, out_size),
                    nn.Dropout(dropout_prob))
        if pre_enrich:
            if activation:
                self.enricher = nn.Sequential(
                    nn.Linear(feature_size, feature_size),
                    nn.ReLU())
            else:
                self.enricher = nn.Linear(feature_size, feature_size)
        else:
            self.enricher = None
        # manually set cuda because module doesn't see these combiners for bottom
        if CUDA:
            self.out = self.out.cuda()
            if self.enricher:
                self.enricher = self.enricher.cuda()

    def forward(self, hidden, features, categories=None):

        if self.enricher is not None:
            features = self.enricher(features)

        return self.out(torch.cat((hidden, features), dim=-1))


class JointTag_withfeature(nn.Module):
    def __init__(self, args, config, cls_num_labels=2, token_num_labels=None, tok2id=None):
        super().__init__()
        self.config = config
        self.args = args
        self.batch_size = args.train_batch_size
        self.bert = BertModel.from_pretrained(args.bert_model, config=config)

        self.num_labels = config.num_labels
        self.cls_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_classifier = nn.Linear(768, cls_num_labels)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.token_classifier = nn.Linear(768, token_num_labels)
        n_feats = 100
        self.featurizer = features.Featurizer(
            tok2id, lexicon_feature_bits=1, yang_set=True)
        self.token_classifier = ConcatCombine(768,
                                              n_feats,
                                              out_size=token_num_labels,
                                              layers=2,
                                              dropout_prob=0.1,
                                              small=False,
                                              pre_enrich=True,
                                              activation=True)

    def load_pretrain(self, bert_model, classifier):
        self.bert = bert_model
        # self.token_classifier = classifier

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None, class_label=None, pos_ids=None, rel_ids=None, w1=1, w2=2, w3=0.5, batch_size_=8, CUDA=(torch.cuda.device_count() > 0)):
        self.batch_size = batch_size_
        output = self.bert(input_ids, token_type_ids,
                           attention_mask, head_mask=None)
        # print(class_label)
        # print(labels)
        # features.
        features = self.featurizer.featurize_batch(input_ids.detach().cpu().numpy(),
                                                   rel_ids.detach().cpu().numpy(),
                                                   pos_ids.detach().cpu().numpy(),
                                                   padded_len=0)
        features = torch.tensor(features, dtype=torch.float)
        if CUDA:
            features = features.cuda()

        sequence_output = output[0]

        if self.config.output_hidden_states:
            hidden_states = output[1]
            sequence_output = hidden_states[-4]

        cls_rep = sequence_output[:, 0, :]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32, device="cuda:0")
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]

        # tagging loss only on tag labels.
        # valid_output = valid_output[:self.batch_size]
        sequence_output = self.dropout(valid_output)
        tag_logits = self.token_classifier(sequence_output, features)

        # tag_logits = self.token_classifier(sequence_output)
        cls_logits = self.cls_classifier(cls_rep)

        if labels is not None:
            bias_mask = class_label.view(-1) == 1
            neutral_mask = class_label.view(-1) == 0
            # print(bias_mask)
            labels = labels[bias_mask]
            tag_logits = tag_logits[bias_mask]
            attention_mask_label = attention_mask_label[bias_mask]

            token_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            class_loss_fct = nn.CrossEntropyLoss()

            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1

                # print(active_loss.shape)
                active_logits = tag_logits.view(-1,
                                                self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                token_loss = token_loss_fct(active_logits, active_labels)
            else:
                token_loss = token_loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))

            # combinatio of the loss. Loss = a*Loss_cls_joint + b * Loss_tagging + c *  Loss_cls_neg
            # print(cls_logits[neutral_mask])
            # print(class_label[neutral_mask])
            cls_joint_loss = class_loss_fct(
                cls_logits[bias_mask], class_label[bias_mask])
            cls_neg_loss = class_loss_fct(
                cls_logits[neutral_mask], class_label[neutral_mask])

            # print(token_loss)
            # print(cls_neg_loss)
            # print(cls_joint_loss)
            #print(w1, w2, w3)
            return float(w1) * cls_joint_loss + float(w2) * token_loss + float(w3) * cls_neg_loss
        else:
            tag_logits = self.token_classifier(
                sequence_output, features)[:batch_size_]
            cls_logits = self.cls_classifier(cls_rep)

            return cls_logits, tag_logits


class Tag(BertForTokenClassification):
    # print("TATGGG")
    # raise EOFError
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None, class_label=None):
        sequence_output = self.bert(
            input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(
            batch_size, max_len, feat_dim, dtype=torch.float32, device="cuda:0")
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, class_label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            class_label: (Optional) string. The class label of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.class_label = class_label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, class_label=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.class_label = class_label


def readfile(filename):
    '''
    read file
    '''
    f = open(filename, encoding="utf-8")
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue

        splits = line.split(' ')
        if splits[0] in ['``', "''"]:
            splits[0] = '"'
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []
    return data


def read_special_format(filename):
    data = []
    sents_all = []
    with open(filename, encoding='utf-8') as reader:
        i = -1
        for line in reader:

            i += 1
            if i % 5 == 0:
                sents = line.split(" ||| ")[0].split(" ")
                sents = [y.replace('``', '"') for y in sents]
                sents = [y.replace("''", '"') for y in sents]

                class_label = line.split(" ||| ")[1].strip()

            elif i % 5 == 3:
                labels = line.split(" ||| ")[0].split(" ")

            elif i % 5 == 4:
                data.append((sents, labels, class_label))
                sents_all.append(" ".join(sents))
            else:
                continue
    print("All sents", len(sents_all))
    print("All unique sents", len(set(sents_all)))

    return data


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    @classmethod
    def _read_joint_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return read_special_format(input_file)


class BiasProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train_all_conll.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_conll.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "test__500_conll.txt")), "test")
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "first_batch.txt")), "test")

    def get_labels(self):
        return ["O", "B-bias", "I-bias", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class JointBiasProcessor(DataProcessor):
    """Processor for the special format."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "train_finegrained.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "dev_finegrained.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "test_finegrained.tsv")), "test")

    def get_train_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "train_neg.tsv")), "train_neg")

    def get_dev_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "dev_neg.tsv")), "dev_neg")

    def get_test_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "test_neg.tsv")), "test_neg")

    def get_labels(self):
        return ["O", "B-frame_bias", "I-frame_bias", 'B-epistemological_bias', 'I-epistemological_bias', 'B-demographic_bias', "I-demographic_bias", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label, class_label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a,
                            text_b=text_b, label=label, class_label=class_label))
        return examples


class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "valid.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')

        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                try:
                    label_ids.append(label_map[labels[i]])
                except KeyError:
                    print(labels)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask))
    return features


def convert_examples_to_features_joint(examples, label_list, max_seq_length, tokenizer, class_label_list=["0", "1"]):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    class_lable_map = {label: i for i, label in enumerate(class_label_list)}
    print("CLass label", class_lable_map)

    features = []

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')

        labellist = example.label
        tokens = []
        labels = []
        valid = []
        label_mask = []
        try:
            for i, word in enumerate(textlist):
                token = tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = labellist[i]
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        valid.append(0)
        except:
            print(example.text_a)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0, 1)
        label_mask.insert(0, 1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                try:
                    label_ids.append(label_map[labels[i]])
                except KeyError:
                    print(labels)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(0)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)

        # Add class label.
        class_id = class_lable_map[example.class_label]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "class labels: %s" % class_id)
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid,
                          label_mask=label_mask,
                          class_label=class_id,
                          pos_ids=pre_pos,
                          dep_ids=pre_dep))
        pickle.dump(features, open("cache/%s.p" % name, "wb"))

    return features


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--pretrained_tagging_model", default="", type=str,
                        help="Pretrained tagging model")

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval or not.")
    parser.add_argument("--eval_on",
                        default="dev",
                        help="Whether to run eval on the dev set or test set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--loss_weight",
                        default="1_2_0.5",
                        type=str,
                        help="Weight of Losses.")

    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=100,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--task_ratio',
                        type=float, default=0.0,
                        help="")
    parser.add_argument('--server_ip', type=str, default='',
                        help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='',
                        help="Can be used for distant debugging.")

    # whether do joint training.
    parser.add_argument("--do_joint",
                        default=False,
                        type=bool,
                        help="Whether include classification as a auxilary task")
    args = parser.parse_args()

    # get loss weight.
    w1, w2, w3 = args.loss_weight.split("_")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {"ner": NerProcessor,
                  "tag": BiasProcessor, 'joint': JointBiasProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    # classification.
    class_label_list = ['0', "1"]
    num_class_label = len(class_label_list)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = 0
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_neg_example = processor.get_train_examples_neg(args.data_dir)

        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    # Prepare model
    config = BertConfig.from_pretrained(
        args.bert_model, num_labels=num_labels, finetuning_task=args.task_name, output_hidden_states=False)

    if args.do_joint != True:
        model = Tag.from_pretrained(args.bert_model,
                                    from_tf=False,
                                    config=config)
        print("Initialize Tag without Joint")

    else:
        model = JointTag_withfeature(
            args=args, config=config, token_num_labels=num_labels, tok2id=tokenizer.vocab)
        if args.pretrained_tagging_model != "":
            num_labels = 6
            load_model = JointTag_withfeature(
                args=args, config=config, token_num_labels=num_labels, tok2id=tokenizer.vocab)
            load_model.load_state_dict(torch.load(os.path.join(
                "out_tag_cased_0_1_0_extra_feature_3epoch", "model.pt")))
            model.load_pretrain(load_model.bert, load_model.token_classifier)
            print("load pretrained")
            # raise EOFError

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    # Modify train.
    if args.do_train:
        # train_examples = processor.get_test_examples(args.data_dir)
        if args.do_joint:
            train_features = convert_examples_to_features_with_addition(
                train_examples, label_list, args.max_seq_length, tokenizer, name_file='train')
            train_neg_features = convert_examples_to_features_with_addition(
                train_neg_example, label_list, args.max_seq_length, tokenizer, name_file='train_neg')
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
            train_neg_features = convert_examples_to_features(
                train_neg_example, label_list, args.max_seq_length, tokenizer)

        # load training data.
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_dataloader = create_loader(
            train_features, args, args.train_batch_size, test=False)
        train_neg_dataloader = create_loader(
            train_neg_features, args, args.train_batch_size, test=False)

        model.train()

        if args.do_joint:
            print("Start Training")
            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                neg_iterator = iter(train_neg_dataloader)
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)

                    # load neg example.
                    neg_batch = neg_iterator.next()
                    neg_batch = tuple(t.to(device) for t in neg_batch)

                    input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, cls_id, pos_id, dep_id = batch
                    input_ids_2, input_mask_2, segment_ids_2, label_ids_2, valid_ids_2, l_mask_2, cls_id_2, pos_id_2, dep_id_2 = neg_batch

                    loss = model(torch.cat([input_ids, input_ids_2], dim=0),
                                 torch.cat(
                                     [segment_ids, segment_ids_2], dim=0),
                                 torch.cat([input_mask, input_mask_2], dim=0),
                                 torch.cat([label_ids, label_ids_2], dim=0),
                                 torch.cat([valid_ids, valid_ids_2], dim=0),
                                 torch.cat([l_mask, l_mask_2], dim=0),
                                 torch.cat([cls_id, cls_id_2], dim=0),
                                 torch.cat([pos_id, pos_id_2], dim=0),
                                 torch.cat([dep_id, dep_id_2], dim=0),
                                 w1=w1, w2=w2, w3=w3)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1
        else:
            print("Start Training")
            for _ in trange(int(args.num_train_epochs), desc="Epoch"):
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    batch = tuple(t.to(device) for t in batch)

                    input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, cls_id = batch
                    loss = model(input_ids, segment_ids, input_mask,
                                 label_ids, valid_ids, l_mask, cls_id)

                    if n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu.
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

        # Save a trained model and the associated configuration

        # if joint model.
        if args.do_joint:
            torch.save(model.state_dict(), os.path.join(
                args.output_dir, "model.pt"))
            tokenizer.save_pretrained(args.output_dir)
        else:
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                            "max_seq_length": args.max_seq_length, "num_labels": len(label_list)+1, "label_map": label_map}
            json.dump(model_config, open(os.path.join(
                args.output_dir, "model_config.json"), "w"))

        # Load a trained model and config that you have fine-tuned
    else:

        if args.do_joint:
            tokenizer = BertTokenizer.from_pretrained(
                args.bert_model, do_lower_case=args.do_lower_case)
            model = JointTag_withfeature(
                args=args, config=config, token_num_labels=num_labels, tok2id=tokenizer.vocab)
            model.load_state_dict(torch.load(
                os.path.join(args.output_dir, "model.pt")))
        # Load a trained model and vocabulary that you have fine-tuned
        else:
            model = Tag.from_pretrained(args.output_dir)
            print("TAG MODEL")
        tokenizer = BertTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)

    model.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        # Write test file.
        if args.eval_on == "dev":
            eval_examples = processor.get_dev_examples(args.data_dir)
            eval_examples_neg = processor.get_dev_examples_neg(args.data_dir)
        elif args.eval_on == "test":
            eval_examples = processor.get_test_examples(args.data_dir)
            eval_examples_neg = processor.get_test_examples_neg(args.data_dir)
            print(len(eval_examples))

        else:
            raise ValueError("eval on dev or test set only")

        if args.do_joint:
            if args.eval_on == "dev":
                eval_features = convert_examples_to_features_with_addition(
                    eval_examples, label_list, args.max_seq_length, tokenizer, name_file='dev')
                eval_neg_features = convert_examples_to_features_with_addition(
                    eval_examples_neg, label_list, args.max_seq_length, tokenizer, name_file='dev_neg')
            else:
                eval_features = convert_examples_to_features_with_addition(
                    eval_examples, label_list, args.max_seq_length, tokenizer, name_file='test')
                eval_neg_features = convert_examples_to_features_with_addition(
                    eval_examples_neg, label_list, args.max_seq_length, tokenizer, name_file='test_neg')
        else:
            eval_features = convert_examples_to_features(
                eval_examples, label_list, args.max_seq_length, tokenizer, name_file='dev_neg')
            eval_neg_features = convert_examples_to_features(
                eval_examples_neg, label_list, args.max_seq_length, tokenizer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        eval_dataloader = create_loader(
            eval_features, args, args.eval_batch_size, test=True)
        eval_data_neg_loader = create_loader(
            eval_neg_features, args, args.eval_batch_size, test=True)

        if args.do_joint:
            # Run prediction for full data

            eval_neg_iterator = iter(eval_data_neg_loader)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []

            y_cls_true = []
            y_cls_pred = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}

            for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, class_id, pos_id, dep_id in tqdm(eval_dataloader, desc="Evaluating"):
                neg_batch = eval_neg_iterator.next()

                neg_batch = tuple(t.to(device) for t in neg_batch)

                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                class_id = class_id.to(device)
                pos_id = pos_id.to(device)
                dep_id = dep_id.to(device)

                input_ids_2, input_mask_2, segment_ids_2, label_ids_2, valid_ids_2, l_mask_2, cls_id_2, pos_id_2, dep_id_2 = neg_batch

                with torch.no_grad():
                    cls_logits, logits = model(torch.cat([input_ids, input_ids_2], dim=0),
                                               torch.cat(
                                                   [segment_ids, segment_ids_2], dim=0),
                                               torch.cat(
                                                   [input_mask, input_mask_2], dim=0),

                                               valid_ids=torch.cat(
                                                   [valid_ids, valid_ids_2], dim=0),
                                               attention_mask_label=torch.cat(
                        [l_mask, l_mask_2], dim=0),
                        pos_ids=torch.cat([pos_id, pos_id_2], dim=0),
                        rel_ids=torch.cat([dep_id, dep_id_2], dim=0),
                        w1=w1, w2=w2, w3=w3, batch_size_=args.eval_batch_size)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()

                # cls logits.
                print(cls_logits)
                cls_logits = torch.argmax(cls_logits, dim=1)
                cls_logits = cls_logits.detach().cpu().numpy()

                cls_label_ids = torch.cat(
                    [class_id, cls_id_2], dim=0).to('cpu').numpy()
                print(cls_logits)
                print(cls_label_ids)
                # raise EOFError
                y_cls_true.extend(cls_label_ids)
                y_cls_pred.extend(cls_logits)
                assert len(y_cls_true) == len(y_cls_pred)

                for i, label in enumerate(label_ids[:args.eval_batch_size]):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:

                            temp_1.append(label_map[label_ids[i][j]])
                            if logits[i][j] == 0:
                                temp_2.append("O")
                            else:
                                temp_2.append(label_map[logits[i][j]])

            print(len(y_true))
            print(len(y_pred))

            print(y_cls_true)
            print(y_cls_pred)
            # print classification result.

            classify_report = cls_report(y_cls_true,  y_cls_pred, digits=4)
            accuracy = sum(1 for x, y in zip(
                y_cls_true, y_cls_pred) if x == y) / len(y_cls_pred)
            print("Accuracy is ", accuracy)
            print(classify_report)
            import pickle
            file1 = pickle.dump([y_cls_true, y_cls_pred], open(
                "%s_out_%s_classification_result.p" % (args.eval_on, args.output_dir), "wb"))

            # print(y_true[0])
            write_file = open("%s_out_%s_task_ratio%s.txt" % (
                args.eval_on, args.output_dir, args.task_ratio), "w", encoding="utf-8")
            data_tuples = read_special_format(os.path.join(
                args.data_dir, "tag_%s_fine_grained_fixed.tsv" % args.eval_on))
            for id_pred, line in enumerate(y_pred):

                # raise EOFError
                for idx, g in enumerate(line):
                    # if g == "[SEP]":
                    #     g = 'O'
                    word = data_tuples[id_pred][0][idx]
                    # print(word)
                    gold = y_true[id_pred][idx]
                    write_file.write(word + " " + gold + " " + g + '\n')
                write_file.write("\n")
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)
            output_eval_file = os.path.join(
                args.output_dir, "%s_eval_results.txt" % args.eval_on)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Tagging Eval results *****")
                logger.info("\n%s", report)
                logger.info("***** Classification Eval results *****")
                logger.info("\n%s", classify_report)
                writer.write("***** Tagging Eval results *****")
                writer.write("\n%s" % report)
                writer.write("***** Classification Eval results *****")
                writer.write("\n%s" % classify_report)
                writer.write("\n")
        else:
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            y_true = []
            y_pred = []

            y_cls_true = []
            y_cls_pred = []

            label_map = {i: label for i, label in enumerate(label_list, 1)}

            for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask, class_id in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                valid_ids = valid_ids.to(device)
                label_ids = label_ids.to(device)
                l_mask = l_mask.to(device)
                class_id = class_id.to(device)

                with torch.no_grad():
                    logits = model(input_ids,
                                   segment_ids,
                                   input_mask,

                                   valid_ids=valid_ids,
                                   attention_mask_label=l_mask)

                logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()

                for i, label in enumerate(label_ids[:args.eval_batch_size]):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_map):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:

                            temp_1.append(label_map[label_ids[i][j]])
                            if logits[i][j] == 0:
                                temp_2.append("O")
                            else:
                                temp_2.append(label_map[logits[i][j]])

            write_file = open("%s_out_%s.txt" % (
                args.eval_on, args.output_dir), "w", encoding="utf-8")
            data_tuples = read_special_format(os.path.join(
                args.data_dir, "tag_%s.tsv" % args.eval_on))
            for id_pred, line in enumerate(y_pred):

                for idx, g in enumerate(line):

                    word = data_tuples[id_pred][0][idx]
                    gold = y_true[id_pred][idx]
                    write_file.write(word + " " + gold + " " + g + '\n')
                write_file.write("\n")
            report = classification_report(y_true, y_pred, digits=4)
            logger.info("\n%s", report)
            output_eval_file = os.path.join(
                args.output_dir, "%s_eval_results.txt" % args.eval_on)
            with open(output_eval_file, "w") as writer:
                logger.info("***** Tagging Eval results *****")
                logger.info("\n%s", report)

                writer.write("***** Tagging Eval results *****")
                writer.write("\n%s" % report)


if __name__ == "__main__":

    main()
