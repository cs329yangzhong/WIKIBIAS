import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str, default="data/train_conll.txt")
parser.add_argument('--pad_idx', type=int, default=0)
parser.add_argument('--unk_idx', type=int, default=1)
parser.add_argument('--tokenizer', type=str, default="word")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--max_len', type=int, default=128)
parser.add_argument('--seed_num', type=int, default=203)
args = parser.parse_args()
