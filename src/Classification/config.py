import argparse
import torch
import os
import datetime
import random
import numpy as np

parser = argparse.ArgumentParser()
# learning.
parser.add_argument("--name", type=str, default='')
parser.add_argument("--verbose", action='store_true', default=False)
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--ckpt', type=str, default="ckpt/model.pt")
parser.add_argument('--pretrained-ckpt', type=str, default='')
parser.add_argument("--train_set", type=str, default="biased")
parser.add_argument('--pad_idx', type=int, default=3)
parser.add_argument('--unk_idx', type=int, default=1)
parser.add_argument('--rand_seed', type=int, default=100)
parser.add_argument('--tokenizer', type=str, default='word')
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument('--grad-step', type=int, default=-1)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--max-len', type=int, default=128)
parser.add_argument('--model', type=str, default="bert")
parser.add_argument('--eval_ckpt', type=str, default="")
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--n_emb', type=int, default=-1)
parser.add_argument('--clip-grad', type=float, default=3)
parser.add_argument('--bert-model', type=str, default='bert-base-uncased')
parser.add_argument('--train_path', type=str, default='data/train.tsv')

parser.add_argument("--add_neg", type=str, default="no")
# CNN model.
parser.add_argument('--pooling-type', type=str, default="MAX")
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('-kernel-num', type=int, default=100,
                    help='number of each kind of kernel')
parser.add_argument('--class_num', type=int, default=2)
parser.add_argument('--hidden-size', type=int, default=100)
parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('--n_bert_hid', type=int, default=768)
parser.add_argument('--interval', type=int)
parser.add_argument('--cuda', type=bool)
parser.add_argument('--save_dir', type=str, default="cache")
parser.add_argument('--mode_SP', type=str, default='number',
                    help='Self-paced Learning apporaches ["number", "noisyLoss"]')
parser.add_argument('--noisy_method', type=str, default='GLC')
parser.add_argument('--eval_on', type=str, default='data/test_0820.tsv')
parser.add_argument('--bias_type', type=str, default='frame',
                    help='select from {frame, demographic, epistemological}')
parser.add_argument('--strategy', type=str, default='none',
                    help='select from ["none", "reweight_teacher", "focal_loss", "product_expert"]')
args = parser.parse_args()
print(args)

# update args and print
args.class_num = 2
args.cuda = torch.cuda.is_available()

args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(
    args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args = parser.parse_args()
args.ckpt = "ckpt/model_%s_%s.pt" % (args.model, args.train_set)
if args.model == "bert":
    args.tokenizer = "bert"
