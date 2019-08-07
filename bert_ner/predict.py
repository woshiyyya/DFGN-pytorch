"""
For hotpot Pipeline in codalab
Rewrite eval.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from model import Net
from data_load import NerDataset, pad, VOCAB, tokenizer, tag2idx, idx2tag, EvalDataset, QueryDataset
import os
import numpy as np
import argparse
from tqdm import tqdm
import json
import re


def tag_numbers(words, preds):
    MONTH = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    rule = re.compile('^[0-9]+$')

    def date_pattern_1(ptr):
        # e.g. 21 March 1873
        if ptr + 2 < len(words):
            return (words[ptr + 1] in MONTH) and re.match(rule, words[ptr]) and re.match(rule, words[ptr + 2])
        else:
            return False

    def date_pattern_2(ptr):
        # e.g. December 12, 2019
        if ptr + 3 < len(words):
            return (words[ptr] in MONTH) and re.match(rule, words[ptr + 1]) and words[ptr + 2] == ',' and re.match(rule, words[ptr + 3])
        else:
            return False

    ptr = 0
    while ptr < len(words):
        if preds[ptr] != 'O':
            ptr += 1
        elif date_pattern_1(ptr):
            preds[ptr:ptr+3] = ['J-DATE'] * 3
            ptr += 3
        elif date_pattern_2(ptr):
            preds[ptr:ptr+4] = ['J-DATE'] * 4
            ptr += 4
        elif re.match(rule, words[ptr]):
            preds[ptr] = 'J-NUM'
            ptr += 1
        else:
            ptr += 1
    return preds


def get_entities(words, preds):
    entities = []
    ptr = 0
    while ptr < len(words):
        FLAG = False
        for prefix in ['I-', 'J-']:
            sub_words = []
            while ptr < len(words) and preds[ptr].startswith(prefix):
                sub_words.append(words[ptr])
                ptr += 1
            if len(sub_words) > 0:
                entity = " ".join(sub_words).replace(' .', '.').replace(' ,', ',')  # Rearrange blank
                entities.append([entity, preds[ptr - 1]])
                FLAG = True

        if not FLAG:
            ptr += 1

    return entities


def eval_para(model, iterator, sent_ids, output_path):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    entities = {k: dict() for k, sid in sent_ids}
    # gets results and save
    for i, (words, is_heads, tags, y_hat) in enumerate(zip(Words, Is_heads, Tags, Y_hat)):
        y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
        preds = [idx2tag[hat] for hat in y_hat]
        assert len(preds) == len(words), f'len(preds)={len(preds)}, len(words)={len(words)}'
        words, preds = words[1:-1], preds[1:-1]
        preds = tag_numbers(words, preds)

        entity = get_entities(words, preds)
        key, sid = sent_ids[i][0], sent_ids[i][1]
        entities[key][sid] = entity
    json.dump(entities, open(output_path, 'w'))
    return


def eval_query(model, iterator, sent_ids, output_path):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator)):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    entities = dict()
    # gets results and save
    with open("result.txt", 'w') as fout:
        for i, (words, is_heads, tags, y_hat) in enumerate(zip(Words, Is_heads, Tags, Y_hat)):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words), f'len(preds)={len(preds)}, len(words)={len(words)}'

            words, preds = words[1:-2], preds[1:-2]  # remove the last punctuation "?"
            preds = tag_numbers(words, preds)
            for w, p in zip(words, preds):
                fout.write(f"{w} {p}\n")
            fout.write("\n")

            entity = get_entities(words, preds)
            key = sent_ids[i][0]
            entities[key] = entity
    json.dump(entities, open(output_path, 'w'))
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default='work_dir/bert_ner.pt')
    parser.add_argument('--input_path', type=str, default='work_dir/selected_paras.json')
    parser.add_argument('--output_path', type=str, default='work_dir/entities.json')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    eval_dataset = EvalDataset(args.input_path, debug=False)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 collate_fn=pad)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(top_rnns=False, vocab_size=len(VOCAB), device=device, finetuning=True).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    eval_para(model, eval_iter, eval_dataset.sent_id, args.output_path)
