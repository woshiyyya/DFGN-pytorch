from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
from os.path import join
from collections import Counter

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


class Example(object):

    def __init__(self,
                 qas_id,
                 qas_type,
                 doc_tokens,
                 question_text,
                 sent_num,
                 sent_names,
                 sup_fact_id,
                 sent_start_end_position,
                 entity_start_end_position,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.doc_tokens = doc_tokens
        self.question_text = question_text
        self.sent_num = sent_num
        self.sent_names = sent_names
        self.sup_fact_id = sup_fact_id
        self.sent_start_end_position = sent_start_end_position
        self.entity_start_end_position = entity_start_end_position
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qas_id,
                 doc_tokens,
                 doc_input_ids,
                 doc_input_mask,
                 doc_segment_ids,
                 query_tokens,
                 query_input_ids,
                 query_input_mask,
                 query_segment_ids,
                 sent_spans,
                 entity_spans,
                 sup_fact_ids,
                 token_to_orig_map,
                 start_position=None,
                 end_position=None):

        self.qas_id = qas_id
        self.doc_tokens = doc_tokens
        self.doc_input_ids = doc_input_ids
        self.doc_input_mask = doc_input_mask
        self.doc_segment_ids = doc_segment_ids

        self.query_tokens = query_tokens
        self.query_input_ids = query_input_ids
        self.query_input_mask = query_input_mask
        self.query_segment_ids = query_segment_ids

        self.sent_spans = sent_spans
        self.entity_spans = entity_spans
        self.sup_fact_ids = sup_fact_ids

        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


def clean_entity(entity):
    Type = entity[1]
    Text = entity[0]
    if Type == "DATE" and ',' in Text:
        Text = Text.replace(' ,', ',')
    if '?' in Text:
        Text = Text.split('?')[0]
    Text = Text.replace("\'\'", "\"")
    Text = Text.replace("# ", "#")
    return Text


def check_in_full_paras(answer, paras):
    full_doc = ""
    for p in paras:
        full_doc += " ".join(p[1])
    return answer in full_doc


def read_hotpot_examples(para_file, full_file, entity_file):
    with open(para_file, 'r', encoding='utf-8') as reader:
        para_data = json.load(reader)

    with open(full_file, 'r', encoding='utf-8') as reader:
        full_data = json.load(reader)

    with open(entity_file, 'r', encoding='utf-8') as reader:
        entity_data = json.load(reader)

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []

    # for debug
    actually_in_case = 0
    failed_case = 0
    failed_sup_case = 0
    failed_case_paranum = []
    para_length = []
    bert_cutted_para_length = []
    for case in tqdm(full_data):
        key = case['_id']
        qas_type = case['type']
        sup_facts = set([(sp[0], sp[1])for sp in case['supporting_facts']])
        orig_answer_text = case['answer']

        sent_id = 0
        doc_tokens = []
        sent_names = []
        sup_facts_sent_id = []
        sent_start_end_position = []
        entity_start_end_position = []

        JUDGE_FLAG = orig_answer_text == 'yes' or orig_answer_text == 'no'
        FIND_FLAG = False

        char_to_word_offset = []  # Accumulated along all sentences
        prev_is_whitespace = True

        ans_start_position = None
        ans_end_position = None

        # for debug
        titles = set()

        for paragraph in para_data[key]:
            title = paragraph[0]
            sents = paragraph[1]
            if title in entity_data[key]:
                entities = entity_data[key][title]
            else:
                entities = []

            titles.add(title)

            for local_sent_id, sent in enumerate(sents):
                # Determine the global sent id for supporting facts
                local_sent_name = (title, local_sent_id)
                sent_names.append(local_sent_name)
                if local_sent_name in sup_facts:
                    sup_facts_sent_id.append(sent_id)
                sent_id += 1
                sent += " "

                sent_start_word_id = len(doc_tokens)
                sent_start_char_id = len(char_to_word_offset)

                for c in sent:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                sent_end_word_id = len(doc_tokens) - 1
                sent_start_end_position.append((sent_start_word_id, sent_end_word_id))

                # Answer char position
                answer_offset = sent.find(orig_answer_text)
                if not JUDGE_FLAG and not FIND_FLAG and answer_offset != -1:
                    FIND_FLAG = True
                    start_char_position = sent_start_char_id + answer_offset
                    end_char_position = start_char_position + len(orig_answer_text) - 1
                    ans_start_position = char_to_word_offset[start_char_position]
                    ans_end_position = char_to_word_offset[end_char_position]

                # Find Entity Position
                entity_pointer = 0
                for entity in entities:
                    entity_text = clean_entity(entity)
                    entity_offset = sent.find(entity_text)
                    if entity_offset != -1:
                        entity_pointer += 1
                        start_char_position = sent_start_char_id + entity_offset
                        end_char_position = start_char_position + len(entity_text) - 1
                        ent_start_position = char_to_word_offset[start_char_position]
                        ent_end_position = char_to_word_offset[end_char_position]
                        entity_start_end_position.append((ent_start_position, ent_end_position, entity_text))
                    else:
                        break
                entities = entities[entity_pointer:]

                # Truncate longer document
                if len(doc_tokens) > 382:
                    break

            para_length.append(len(doc_tokens))

        # for ent in entity_start_end_position:
        #     print(ent)
        #     print(" ".join(doc_tokens[ent[0]:ent[1]+1]))
        # input()

        example = Example(
            qas_id=key,
            qas_type=qas_type,
            doc_tokens=doc_tokens,
            question_text=case['question'],
            sent_num=sent_id + 1,
            sent_names=sent_names,
            sup_fact_id=sup_facts_sent_id,
            sent_start_end_position=sent_start_end_position,
            entity_start_end_position=entity_start_end_position,
            orig_answer_text=orig_answer_text,
            start_position=ans_start_position,
            end_position=ans_end_position)
        examples.append(example)

    return examples

    #     # Check Finding Answer position
    #     if not JUDGE_FLAG:
    #         if end_position is None:
    #             failed_case += 1
    #             failed_case_paranum.append(len(para_data[key]))
    #             print(orig_answer_text)
    #             print(key)
    #             print("({}, {})".format(start_position, end_position))
    #             actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
    #             cleaned_answer_text = " ".join(whitespace_tokenize(orig_answer_text))
    #             if actual_text.find(cleaned_answer_text) == -1:
    #                 print("Could not find answer:")
    #             print("ACTUAL: ", actual_text)
    #             print("CLEANED: ", cleaned_answer_text)
    #             print(doc_tokens)
    #             input()
    #
    #     # Check finding supporting facts
    #     if len(sup_facts) != len(sup_facts_sent_id):
    #         failed_sup_case += 1
    #         print("gets:", titles)
    #         print("facts:", set([sp[0] for sp in sup_facts]))
    #         print("facts id:", sup_facts_sent_id)
    #
    # print("no answer: ", failed_case)
    # print("lack sup fact: ", failed_sup_case)
    # print("total cases: ", len(full_data))
    # print("noanswer para num: ", Counter(failed_case_paranum))
    # print("Avg bert len: ", np.average(np.array(bert_cutted_para_length)))
    # print("Max bert len: ", np.max(np.array(bert_cutted_para_length)))
    # np.array(bert_cutted_para_length).dump("bert_doc_len_aug.np")


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    features = []
    failed = 0
    ans_failed = 0
    for (example_index, example) in enumerate(tqdm(examples)):
        query_tokens = ["[CLS]"] + tokenizer.tokenize(example.question_text)
        if len(query_tokens) > max_query_length - 1:
            query_tokens = query_tokens[:max_query_length - 1]
        query_tokens.append("[SEP]")

        tok_to_orig_index = [0]
        orig_to_tok_index = []
        orig_to_tok_back_index = []
        all_doc_tokens = ["[CLS]"]
        sentence_spans = []
        entity_spans = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            orig_to_tok_back_index.append(len(all_doc_tokens) - 1)

        def relocate_tok_span(orig_start_position, orig_end_position, orig_text):
            if orig_start_position is None:
                return 0, 0

            global tokenizer
            nonlocal orig_to_tok_index, example, all_doc_tokens

            tok_start_position = orig_to_tok_index[orig_start_position]
            if orig_end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[orig_end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # Make answer span more accurate.
            return _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, orig_text)

        ans_start_position, ans_end_position \
            = relocate_tok_span(example.start_position, example.end_position, example.orig_answer_text)

        for entity_span in example.entity_start_end_position:
            ent_start_position, ent_end_position \
                = relocate_tok_span(entity_span[0], entity_span[1], entity_span[2])
            entity_spans.append((ent_start_position, ent_end_position, entity_span[2]))

        for sent_span in example.sent_start_end_position:
            if sent_span[0] >= len(orig_to_tok_index) or sent_span[0] >= sent_span[1]:
                continue
            sent_start_position = orig_to_tok_index[sent_span[0]]
            sent_end_position = orig_to_tok_back_index[sent_span[1]]
            sentence_spans.append((sent_start_position, sent_end_position))

        # Padding Document
        all_doc_tokens = all_doc_tokens[:max_seq_length - 1] + ["[SEP]"]
        doc_input_ids = tokenizer.convert_tokens_to_ids(all_doc_tokens)
        doc_input_mask = [1] * len(doc_input_ids)
        doc_segment_ids = [0] * len(doc_input_ids)

        while len(doc_input_ids) < max_seq_length:
            doc_input_ids.append(0)
            doc_input_mask.append(0)
            doc_segment_ids.append(0)

        # Padding Question
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        query_input_mask = [1] * len(query_input_ids)
        query_segment_ids = [0] * len(query_input_ids)

        while len(query_input_ids) < max_query_length:
            query_input_ids.append(0)
            query_input_mask.append(0)
            query_segment_ids.append(0)

        assert len(doc_input_ids) == max_seq_length
        assert len(doc_input_mask) == max_seq_length
        assert len(doc_segment_ids) == max_seq_length
        assert len(query_input_ids) == max_query_length
        assert len(query_input_mask) == max_query_length
        assert len(query_segment_ids) == max_query_length

        # Dropout out-of-bound span
        entity_spans = entity_spans[:_largest_valid_index(entity_spans, max_seq_length)]
        sentence_spans = sentence_spans[:_largest_valid_index(sentence_spans, max_seq_length)]

        sup_fact_ids = example.sup_fact_id
        sent_num = len(sentence_spans)
        sup_fact_ids = [sent_id for sent_id in sup_fact_ids if sent_id < sent_num]
        if len(sup_fact_ids) != len(example.sup_fact_id):
            failed += 1
        if ans_start_position >= 512:
            ans_failed += 1

        # print("ALL:\n", all_doc_tokens)
        # print("MASK:\n", doc_input_mask)
        # print("ANS:\n", all_doc_tokens[ans_start_position: ans_end_position + 1])
        # print("SP_FACTS: \n", sup_fact_ids)
        # print("ANSWER:\n", example.orig_answer_text)
        # print("ANSWER:\n", all_doc_tokens[ans_start_position: ans_end_position + 1])
        # for i, sent in enumerate(sentence_spans):
        #     print("sent{}:\n".format(i), all_doc_tokens[sent[0]: sent[1] + 1])
        #     os, oe = tok_to_orig_index[sent[0]], tok_to_orig_index[sent[1]]
        #     print("ORI_SENT:\n", example.doc_tokens[os: oe + 1])
        #     input()
        #     for ent in entity_spans:
        #         if ent[0] >= sent[0] and ent[1] <= sent[1]:
        #             print("ORI:  ", ent[2])
        #             print("NEW:  ", all_doc_tokens[ent[0] : ent[1] + 1])
        #     input()

        features.append(
            InputFeatures(qas_id=example.qas_id,
                          doc_tokens=all_doc_tokens,
                          doc_input_ids=doc_input_ids,
                          doc_input_mask=doc_input_mask,
                          doc_segment_ids=doc_segment_ids,
                          query_tokens=query_tokens,
                          query_input_ids=query_input_ids,
                          query_input_mask=query_input_mask,
                          query_segment_ids=query_segment_ids,
                          sent_spans=sentence_spans,
                          entity_spans=entity_spans,
                          sup_fact_ids=sup_fact_ids,
                          token_to_orig_map=tok_to_orig_index,
                          start_position=ans_start_position,
                          end_position=ans_end_position)
        )
    # print("Failed: ", failed)
    # print("Ans_Failed: ", ans_failed)
    return features


def _largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default='bert-base-cased', type=str)

    ## Other parameters
    parser.add_argument("--do_lower_case", default=False, action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--layers", default="-1,-2,-3", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--batch_size", default=15, type=int, help="Batch size for predictions.")

    args = parser.parse_args()


    data_path = '/home/yunxuanxiao/xyx/data/HotpotQA/'
    dev_para_path = join(data_path, 'Selected_Paras', 'dev_paras.json')
    dev_full_path = join(data_path, 'hotpot_dev_distractor_v1.json')
    dev_entity_path = join(data_path, 'Selected_Paras', 'dev_entity.json')
    # examples = read_hotpot_examples(para_file=dev_para_path, full_file=dev_full_path, entity_file=dev_entity_path)
    # examples = pickle.load(open(join(data_path, 'BERT_Features', 'examples', 'dev_example.pkl'), 'rb'))
    # pickle.dump(examples, open(join(data_path, 'BERT_Features', 'examples', 'dev_example.pkl'), 'wb'))
    # features = convert_examples_to_features(examples, tokenizer, max_seq_length=512, max_query_length=50)
    # pickle.dump(features, open(join(data_path, 'BERT_Features', 'features', 'dev_feature.pkl'), 'wb'))
    features = pickle.load(open(join(data_path, 'BERT_Features', 'features', 'dev_feature.pkl'), 'rb'))

    model = BertModel.from_pretrained(args.bert_model)
    model.cuda()
    model = torch.nn.DataParallel(model)
    model.eval()

    doc_input_ids = torch.LongTensor([f.doc_input_ids for f in features])
    doc_input_mask = torch.LongTensor([f.doc_input_mask for f in features])
    query_input_ids = torch.LongTensor([f.query_input_ids for f in features])
    query_input_mask = torch.LongTensor([f.query_input_mask for f in features])
    all_example_indices = torch.arange(doc_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(doc_input_ids, doc_input_mask, query_input_ids, query_input_mask, all_example_indices)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    layer_indices = [int(x) for x in args.layers.split(",")]

    output_json = dict()
    out_feature_dir = join(data_path, 'BERT_Features', 'layers')
    file_id = 0
    for doc_ids, doc_mask, query_ids, query_mask, example_indices in tqdm(eval_dataloader):
        doc_ids = doc_ids.cuda()
        doc_mask = doc_mask.cuda()

        all_doc_encoder_layers, _ = model(doc_ids, token_type_ids=None, attention_mask=doc_mask)
        all_query_encoder_layers, _ = model(query_ids, token_type_ids=None, attention_mask=query_mask)
        selected_doc_layers = [all_doc_encoder_layers[layer_index].detach().cpu().numpy() for layer_index in layer_indices]
        selected_query_layers = [all_query_encoder_layers[layer_index].detach().cpu().numpy() for layer_index in layer_indices]

        for b, example_index in enumerate(example_indices):
            feature = features[example_index.item()]
            case = dict()
            case['query'] = [layer[b] for layer in selected_query_layers]
            case['doc'] = [layer[b] for layer in selected_doc_layers]
            output_json[feature.qas_id] = case

    output_file = join(out_feature_dir, "dev_layers.pkl")
    pickle.dump(output_json, open(output_file, 'wb'))
    output_json = dict()
    file_id += 1










