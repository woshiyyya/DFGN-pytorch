'''
An entry or sent looks like ...

SOCCER NN B-NP O
- : O O
JAPAN NNP B-NP B-LOC
GET VB B-VP O
LUCKY NNP B-NP O
WIN NNP I-NP O
, , O O
CHINA NNP B-NP B-PER
IN IN B-PP O
SURPRISE DT B-NP O
DEFEAT NN I-NP O
. . O O

Each mini-batch returns the followings:
words: list of input sents. ["The 26-year-old ...", ...]
x: encoded input sents. [N, T]. int64.
is_heads: list of head markers. [[1, 1, 0, ...], [...]]
tags: list of tags.['O O B-MISC ...', '...']
y: encoded tags. [N, T]. int64
seqlens: list of seqlens. [45, 49, 10, 50, ...]
'''
import numpy as np
import torch
from torch.utils import data
import json

from pytorch_pretrained_bert import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
VOCAB = ('<PAD>', 'O', 'I-LOC', 'B-PER', 'I-PER', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC', 'B-ORG')
tag2idx = {tag: idx for idx, tag in enumerate(VOCAB)}
idx2tag = {idx: tag for idx, tag in enumerate(VOCAB)}


class NerDataset(data.Dataset):
    def __init__(self, fpath):
        """
        fpath: [train|valid|test].txt
        """
        entries = open(fpath, 'r').read().strip().split("\n\n")
        sents, tags_li = [], []  # list of lists
        for entry in entries:
            words = [line.split()[0] for line in entry.splitlines()]
            tags = ([line.split()[-1] for line in entry.splitlines()])
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<PAD>"] + tags + ["<PAD>"])
        self.sents, self.tags_li = sents, tags_li

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags = self.sents[idx], self.tags_li[idx] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        word_tokens = []
        head_list = []
        for i, (w, t) in enumerate(zip(words[:], tags)):
            tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            word_tokens.extend(tokens)
            xx = tokenizer.convert_tokens_to_ids(tokens)

            if len(xx) == 0:
                words.remove(w)
                continue
            if len(tokens) + len(x) > 512:
                words = words[:i]
                break

            head_list.append([1] + [0] * (len(tokens) - 1))
            is_head = [1] + [0] * (len(tokens) - 1)
            t = [t] + ["<PAD>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [tag2idx[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)


        # if len(x) != len(is_heads):
        #     print(f'len(words)={len(words)}, len(tags)={len(tags)}')
        #     print(f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}")
        #     maxlen = max(len(x), len(is_heads))
        #     x.extend(['[XXX]'] * (maxlen - len(x)))
        #     is_heads.extend(['[XXX]'] * (maxlen - len(is_heads)))
        #     word_tokens.extend(['[XXX]'] * (maxlen - len(word_tokens)))
        #     for W, X, H in zip(word_tokens, x, is_heads):
        #         print(f"{W}  {X}  {H}")
        #     print('' in words)
        #     input()
        assert len(x)==len(y)==len(is_heads), f"len(x)={len(x)}, len(y)={len(y)}, len(is_heads)={len(is_heads)}"
        # if len(words)!=sum(is_heads):
        #     print(len(words), sum(is_heads), len(head_list))
        #     print(words)
        #     print(head_list)
        #     for a,b in zip(words, head_list):
        #         print(a, b)
        assert len(words)==sum(is_heads)

        # seqlen
        seqlen = len(y)

        # to string
        # words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


class EvalDataset(NerDataset):
    def __init__(self, fpath, debug=False):
        self.data = json.load(open(fpath, 'r'))
        print(f'loading data from:{fpath}')
        self.keys = list(self.data.keys())
        self.sents = []
        self.tags_li = []
        self.sent_id = []
        self.orig_para = []
        for k, key in enumerate(self.keys):
            if debug and k > 100:
                break
            for i, para in enumerate(self.data[key]):
                self.orig_para.append(para[1])
                self.sent_id.append((key, para[0]))
                para = " ".join(para[1])
                words = para.split()
                clean_words = []

                for w in words:
                    head_words = []
                    tail_words = []
                    while len(w) > 0 and w[0] in ['(', '\"']:
                        head_words.append(w[0])
                        w = w[1:]

                    while len(w) > 0 and w[-1] in [')', '\"', ',', '.', ';']:
                        tail_words.insert(0, w[-1])
                        w = w[:-1]

                    sub_words = head_words + [w] + tail_words
                    clean_words.extend(sub_words)

                trash_token = ['', ' ']
                clean_words = list(filter(lambda x: x not in trash_token, clean_words))

                self.sents.append(["[CLS]"] + clean_words + ["[SEP]"])
                self.tags_li.append(["<PAD>"] * len(self.sents[-1]))


class QueryDataset(NerDataset):
    def __init__(self, fpath, debug=False):
        self.data = json.load(open(fpath, 'r'))
        print(f'loading data from:{fpath}')
        self.sent_id= [[case['_id'], 0] for case in self.data]
        self.sents = []
        self.tags_li = []
        self.orig_para = []
        for k, case in enumerate(self.data):
            if debug and k > 100:
                break

            sent = case['question']
            self.orig_para.append(sent)
            words = sent.split()
            clean_words = []
            for w in words:
                head_words = []
                tail_words = []
                while len(w) > 0 and w[0] in ['(', '\"']:
                    head_words.append(w[0])
                    w = w[1:]
                while len(w) > 0 and w[-1] in [')', '\"', ',', '.', ';', '?']:
                    tail_words.insert(0, w[-1])
                    w = w[:-1]
                sub_words = head_words + [w] + tail_words
                clean_words.extend(sub_words)
            trash_token = ['', ' ']
            clean_words = list(filter(lambda x: x not in trash_token, clean_words))
            self.sents.append(["[CLS]"] + clean_words + ["[SEP]"])
            self.tags_li.append(["<PAD>"] * len(self.sents[-1]))


def pad(batch):
    '''Pads to the longest sample'''
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)


    f = torch.LongTensor

    return words, f(x), is_heads, tags, f(y), seqlens


