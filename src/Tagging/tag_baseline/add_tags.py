"""
add tags to a corpusfile (output of gen_data_from_crawl.py)

"""
import torch
from pytorch_transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                                  BertForTokenClassification, BertTokenizer,
                                  WarmupLinearSchedule, PretrainedConfig)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import sys
import spacy
from tqdm import tqdm
import pickle
import os
NLP = spacy.load('en_core_web_sm')

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False)
# from https://spacy.io/api/annotation#section-dependency-parsing
RELATIONS = [
    'det', 'amod', 'nsubj', 'prep', 'pobj', 'ROOT',
    'attr', 'punct', 'advmod', 'compound', 'acl', 'agent',
    'aux', 'ccomp', 'dobj', 'cc', 'conj', 'appos', 'nsubjpass',
    'auxpass', 'poss', 'nummod', 'nmod', 'relcl', 'mark',
    'advcl', 'pcomp', 'npadvmod', 'preconj', 'neg', 'xcomp',
    'csubj', 'prt', 'parataxis', 'expl', 'case', 'acomp', 'predet',
    'quantmod', 'dep', 'oprd', 'intj', 'dative', 'meta', 'csubjpass',
    '<UNK>'
]
REL2ID = {x: i for i, x in enumerate(RELATIONS, 1)}
REL2ID['<PAD>'] = 0
# from https://spacy.io/api/annotation#section-pos-tagging
POS_TAGS = [
    'DET', 'ADJ', 'NOUN', 'ADP', 'NUM', 'VERB', 'PUNCT', 'ADV',
    'PART', 'CCONJ', 'PRON', 'X', 'INTJ', 'PROPN', 'SYM',
    '<UNK>'
]
POS2ID = {x: i for i, x in enumerate(POS_TAGS, 1)}
POS2ID['<PAD>'] = 0


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


class JointBiasProcessor(DataProcessor):
    """Processor for the special format."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "noisy_train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "tag_dev_0820.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""

        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "tag_test_0820.tsv")), "test")

    def get_train_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "tag_train_neg_0820.tsv")), "train")

    def get_dev_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "tag_dev_neg_0820.tsv")), "dev")

    def get_test_examples_neg(self, data_dir):
        return self._create_examples(
            self._read_joint_tsv(os.path.join(data_dir, "tag_test_neg_0820.tsv")), "test")

    def get_labels(self):
        return ["O", "B-bias", "I-bias", "[CLS]", "[SEP]"]

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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None, class_label=None, pos_ids=None, dep_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask
        self.class_label = class_label
        self.pos_ids = pos_ids
        self.dep_ids = dep_ids


def convert_examples_to_features_with_addition(examples, label_list, max_seq_length, tokenizer, class_label_list=["0", "1"], name_file="train_pos"):
    """Loads a data file into a list of `InputBatch`s."""
    global REL2ID
    global POS2ID

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    class_lable_map = {label: i for i, label in enumerate(class_label_list)}
    print("CLass label", class_lable_map)

    features = []
    examples = examples
    try:
        print("!")
        features = pickle.load(
            open(os.path.join("cache", "%s.p" % name_file), 'rb'))
        print("Load from saved file")
        return features
    except:
        for example in tqdm(examples):
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

            pre_pos, pre_dep = get_pos_dep(tokens)
            assert len(pre_pos) == len(tokens)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                pre_pos = pre_pos[0:(max_seq_length - 2)]
                pre_dep = pre_dep[0:(max_seq_length - 2)]
                valid = valid[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            label_ids = []
            ntokens.append("[CLS]")
            segment_ids.append(0)
            valid.insert(0, 1)
            label_mask.insert(0, 1)
            pre_pos.insert(0, '<PAD>')
            pre_dep.insert(0, '<PAD>')
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
            pre_pos.append('<PAD>')
            pre_dep.append('<PAD>')

            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            label_mask = [1] * len(label_ids)
            assert len(pre_pos) == len(input_ids)
            # print(len(input_ids), len(pre_pos))
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                valid.append(0)
                label_mask.append(0)
                pre_pos.append('<PAD>')
                pre_dep.append('<PAD>')
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
                label_mask.append(0)

            #
            pos_ids = [POS2ID.get(x, POS2ID['<UNK>'])for x in pre_pos]
            dep_ids = [REL2ID.get(x, REL2ID['<UNK>'])for x in pre_dep]
            # Add class label.
            class_id = class_lable_map[example.class_label]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length
            assert len(dep_ids) == max_seq_length
            assert len(pos_ids) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask,
                              class_label=class_id,
                              pos_ids=pos_ids,
                              dep_ids=dep_ids))
        print("SAVE .... ")
        pickle.dump(features, open(os.path.join(
            "cache", "%s.p" % name_file), "wb"))

    return features


def get_pos_dep(toks):
    def words_from_toks(toks):
        words = []
        word_indices = []
        for i, tok in enumerate(toks):
            if tok.startswith('##'):
                words[-1] += tok.replace('##', '')
                word_indices[-1].append(i)
            else:
                words.append(tok)
                word_indices.append([i])
        return words, word_indices

    out_pos, out_dep = [], []
    words, word_indices = words_from_toks(toks)
    analysis = NLP(' '.join(words))

    if len(analysis) != len(words):
        out_pos = ['<UNK>'] * len(toks)
        out_rels = ['<UNK>'] * len(toks)

        return out_pos, out_rels

    for analysis_tok, idx in zip(analysis, word_indices):
        out_pos += [analysis_tok.pos_] * len(idx)
        out_dep += [analysis_tok.dep_] * len(idx)

    assert len(out_pos) == len(out_dep) == len(toks)

    return out_pos, out_dep


def create_loader(features, args=None, batch_size=8, test=False):
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_dep_ids = torch.tensor([f.dep_ids for f in features], dtype=torch.long)
    all_cls_label_ids = torch.tensor(
        [f.class_label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
                            all_valid_ids, all_lmask_ids, all_cls_label_ids, all_pos_ids, all_dep_ids)
    dataloader = DataLoader(
        dataset,
        sampler=(SequentialSampler(dataset)
                 if test else RandomSampler(dataset)),
        batch_size=batch_size)

    return dataloader


if __name__ == '__main__':
    # name_file = "train_pos"
    # features = pickle.load(open(os.path.join("cache", "%s.p"%name_file), 'rb'))
    processors = {'joint': JointBiasProcessor}
    processor = processors['joint']()
    dev_data = processor.get_train_examples(
        data_dir="../data/data_joint_model_0820")

    label_list = ["O", "B-bias", "I-bias", "[CLS]", "[SEP]"]

    g = convert_examples_to_features_with_addition(
        dev_data, label_list, 128, tokenizer, name_file="train_pos")
    print(g[1].pos_ids)
    print(g[1].dep_ids)
    x = create_loader(g)

    processors = {'joint': JointBiasProcessor}
    processor = processors['joint']()
    dev_data = processor.get_dev_examples(
        data_dir="../data/data_joint_model_0820")

    label_list = ["O", "B-bias", "I-bias", "[CLS]", "[SEP]"]

    g = convert_examples_to_features_with_addition(
        dev_data, label_list, 128, tokenizer, name_file="dev_pos")
    print(g[1].pos_ids)
    print(g[1].dep_ids)
    x = create_loader(g)
