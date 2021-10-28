import os
import pickle
import re
import torch
import numpy as np


def check_cache(fname):
    path = os.path.join("cache", fname + '.pkl')
    return os.path.exists(path)


def load_cache(fname):
    path = os.path.join('cache', fname + '.pkl')
    f = open(path, 'rb')
    return pickle.load(f)


def save_cache(obj, fname):
    path = os.path.join('cache', fname + '.pkl')
    f = open(path, 'wb')
    pickle.dump(obj, f)


def load_file(path, max_len):
    with open(path, 'r') as f:
        sents = [s.lower() for s in f.read().strip().split() if
                 len(s.strip()) > 0]
        if max_len > 0:
            sents = sents[:max_len]
    return sents


def load_sent(sent, max_len):
    sent = clean_str(sent).split()
    if max_len > 0:
        sent = sent[:max_len]
    return " ".join(sent)


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def create_pretrained(vocab):
    np.random.seed(1)
    glove_words, glove_vecs = load_cache('glove')
    embeddings = np.zeros((vocab.size(), 300))
    for i in range(vocab.size()):
        word = vocab.decoding[i]
        if word in glove_words:
            embeddings[i, :] = glove_vecs[glove_words[word]]
        else:
            embeddings[i, :] = np.random.uniform(-1.0, 1.0, size=(300,))
    embeddings = torch.from_numpy(embeddings).float()
    return embeddings


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


class Trainer(object):

    def __init__(self,
                 option,
                 model=None,
                 train_dataloader=None,
                 dev_dataloader=None,
                 evaluator=None):

        self.option = option
        self.model = model
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.evaluate = evaluator
        self.set_up_optimizer()

    def set_up_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_train = []
        if self.option['freeze'] == 'embedding':
            # freeze emb layer
            no_train.append('embedding')
        elif self.option['freeze'] != '0':
            layer = int(self.option['freeze'])
            no_train.append('embedding')
            for i in range(layer):
                no_train.append('layer.%d' % i)

        param_optmizer = [(n, p) for n, p in param_optimizer if not any(
            nd in n for nd in no_train)]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                               lr=self.option['learning_rate'],
                               correct_bias=False)

        num_train_examples = self.train_dataloader.size

        num_train_optimization_steps = int(
            num_train_examples / self.option['train_batchsize'] / self.option['gradient_accumulation_steps']) * self.option['max_epoch']
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(
                                                             self.option['warmup_proportion'] * num_train_optimization_steps),
                                                         num_training_steps=num_train_optimization_steps)

    def train(self):
        device = self.model.get_device()
        best_res = 0

        for epoch in range(self.option['max_epoch']):
            train_loss = 0
            self.model.train()
            for step, batch in enumerate(self.train_dataloader.dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids, noisy_loss = batch

                loss, count = self.model(input_ids,
                                         label_ids=label_ids)

            loss = loss / self.option['gradient_accumulation_steps']
            loss.backward()

            if (step + 1) % self.option['gradient_accumulation_steps'] == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            print('Epoch :%d, step:%d, training loss: %s.5f' %
                  (epoch, step, loss.item()))
            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.train_dataloader)
        print("Epoch: %d, average training loss: %.5f" %
              (epoch, avg_train_loss))
        res, avg_dev_loss = self.evaluate(self.model, self.dev_dataloader)
        print("Epoch: %d, RES: %.5f, average dev loss: %.5f" %
              (epoch, res, avg_dev_loss))

        if res > best_res:
            best_res = res
            # save best ckpt
            model_to_save = self.model.module if hasattr(
                self.model, 'module') else self.model
            output_model_file = self.option['model_name']
            torch.save(model_to_save.state_dict(), output_model_file)

        self.model.load_state_dict(torch.load(
            self.option['model_name'], map_location=device))
