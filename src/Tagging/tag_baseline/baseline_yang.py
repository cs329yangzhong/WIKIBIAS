from sklearn.svm import SVC
from add_tags import create_loader
import sys
import os
from transformers import BertTokenizer
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
import scipy
from args import ARGS
from data import get_dataloader, get_dataloader_
from add_tags import *
from utils import is_ranking_hit, tag_result
from features import Featurizer
import sys
sys.path.append('.')
if not os.path.exists(ARGS.working_dir):
    os.makedirs(ARGS.working_dir)

    # Load data.

print('LOADING DATA...')
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False)

tok2id = tokenizer.vocab
tok2id['<del>'] = len(tok2id)
featurizer_yang = Featurizer(tok2id, yang_set=True)
featurizer = Featurizer(tok2id, yang_set=False)
print()
train_dataloader = create_loader(pickle.load(open("cache/dev_pos.p", "rb")))
eval_dataloader = create_loader(pickle.load(
    open("cache/dev_pos.p", "rb")), test=True)


# train_dataloader, num_train_examples = get_dataloader(
#     ARGS.train,
#     tok2id, ARGS.train_batch_size,  ARGS.working_dir + '/train_data.pkl')

eval_dataloader_, num_eval_examples = get_dataloader_(
    ARGS.test,
    tok2id, ARGS.test_batch_size,  ARGS.working_dir + '/test_data.pkl', test=True)


def data_for_scipy(dataloader, by_seq=False):
    outX = []
    outY = []
    for batch in tqdm(dataloader):
        (
            pre_id, pre_mask, pre_len,
            post_in_id, post_out_id,
            tok_label_id, _, rel_ids, pos_ids, categories
        ) = batch
        pre_id = pre_id.numpy()
        pre_len = pre_len.numpy()
        rel_ids = rel_ids.numpy()
        pos_ids = pos_ids.numpy()
        tok_label_id = tok_label_id.numpy()

        features = featurizer.featurize_batch(pre_id, rel_ids, pos_ids)
        for id_seq, seq_feats, seq_len, label_seq in zip(pre_id, features, pre_len, tok_label_id):
            seqX = []
            seqY = []
            for ti in range(seq_len):
                word_features = np.zeros(len(tok2id))
                word_features[id_seq[ti]] = 1.0

                timestep_vec = seq_feats[ti]
                #timestep_vec = np.concatenate((word_features, seq_feats[ti]))

                seqX.append(csr_matrix(timestep_vec))
                seqY.append(label_seq[ti])

            if by_seq:
                outX.append(scipy.sparse.vstack(seqX))
                outY.append(seqY)
            else:
                outX += seqX
                outY += seqY
    if by_seq:
        return outX, outY

    return scipy.sparse.vstack(outX), outY


def data_for_scipy_yang(dataloader, by_seq=False, name_file="train"):
    try:
        print("load from cache")
        outX = pickle.load(open("cache/%s_all_feature.p" % name_file, "rb"))
        outY = pickle.load(open("cache/%s_label.p" % name_file, "rb"))

        print("load done ...")
    except:
        outX = []
        outY = []
        outlabel = []

        for batch in tqdm(dataloader):
            (
                pre_id, pre_mask, _, tok_label_id, valid_id, lmask_id, cls_ids, pos_ids, rel_ids
            ) = batch
            # print(pre_id[0])

            pre_len = [y.tolist().index(5) for y in tok_label_id]

            pre_id = pre_id.numpy()
            # pre_len = pre_len.numpy()
            rel_ids = rel_ids.numpy()
            pos_ids = pos_ids.numpy()
            tok_label_id = tok_label_id.numpy()

            valid_id = valid_id.numpy()

            features = featurizer_yang.featurize_batch(
                pre_id, rel_ids, pos_ids)

            for id_seq, seq_feats, seq_len, label_seq, seq_valid_id in zip(pre_id, features, pre_len, tok_label_id, valid_id):

                seqX = []
                seqY = []
                # print("   ##", seq_len)
                # print(label_seq)
                # print(len(label_seq))
                # print(label_seq[0], label_seq[seq_len-1])
                # print(seq_feats.shape)
                for ti in range(1, seq_len):

                    word_features = np.zeros(len(tok2id))
                    word_features[id_seq[ti]] = 1.0

                    timestep_vec = seq_feats[ti]
                    # print(id_seq[ti])
                    # print(timestep_vec)

                    #timestep_vec = np.concatenate((word_features, seq_feats[ti]))
                    seqX.append(csr_matrix(timestep_vec))
                    seqY.append(label_seq[ti])
                # print(seqX)
                # raise EOFError
                # raise EOFError
                seqY = [1. if x in [2, 3] else 0. for x in seqY]
                # print(seqX)

                if by_seq:
                    outX.append(scipy.sparse.vstack(seqX))
                    outY.append(seqY)
                else:
                    outX += seqX
                    outY += seqY
                outlabel.append(seq_valid_id)
                # print(id_seq)
                # print(id_seq[1:seq_len])
                # print("###", seq_valid_id)
                # print("###", seq_valid_id[1:seq_len])
                # raise EOFError
        pickle.dump(outlabel, open("cache/%s_label_id.p" % name_file, "wb"))
        pickle.dump(outX, open("cache/%s_all_feature.p" % name_file, "wb"))
        pickle.dump(outY, open("cache/%s_label.p" % name_file, "wb"))

    if by_seq:
        return outX, outY

    return scipy.sparse.vstack(outX), outY


trainX, trainY = data_for_scipy_yang(
    train_dataloader, by_seq=False, name_file='train')
# trainX, trainY = data_for_scipy(eval_dataloader_, by_seq=False)
print(trainX.shape)
# trainX, trainY = data_for_scipy(eval_dataloader, by_seq=False)
testX, testY = data_for_scipy_yang(
    eval_dataloader, by_seq=True, name_file='dev')
print(testX[0].shape)
trainX, trainY = shuffle(trainX, trainY)

print('TRAINIG...')
# model = SVC(max_iter=200)
model = LogisticRegression()
model.fit(trainX, trainY)

print('TESTING...')
valid_label = pickle.load(open("cache/dev_label_id.p", "rb"))
# print(valid_label)
# raise EOFError
gold_hit = 0
hits, total = 0, 0
idx = 0
top = 3

with open("result/dev_top%s.txt" % top, "w") as fin:
    for seqX, seqY in tqdm(zip(testX, testY)):

        # print(valid_label)
        Y_proba = model.predict_proba(seqX)
        # print(Y_proba)

        pred, gold = tag_result(Y_proba, seqY, top)
        id2tag = {0: "O", 1: "I-bias"}
        cur_valid = valid_label[idx]
        idx += 1

        # raise EOFError
        pred_out = []
        gold_out = []

        for index, item in enumerate(pred):
            if cur_valid[index+1] == 1:
                if pred[index] == 1:
                    if pred_out == []:
                        pred_out.append('B-bias')
                    elif pred_out[-1] == 'O':
                        pred_out.append('B-bias')
                    else:
                        pred_out.append('I-bias')
                else:
                    pred_out.append('O')

                if gold[index] == 1:
                    if gold_out == []:
                        gold_out.append('B-bias')
                    elif gold_out[-1] == "O":
                        gold_out.append('B-bias')
                    else:
                        gold_out.append('I-bias')
                else:
                    gold_out.append('O')

        assert len(pred_out) == len(gold_out)

        for index in range(len(pred_out)):
            fin.write(pred_out[index] + " " + gold_out[index] + "\n")
        fin.write("\n")
