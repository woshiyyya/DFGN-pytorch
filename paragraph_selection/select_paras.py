from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import logging
import pandas
from tqdm import tqdm
from config import set_args
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from collections import Counter
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "hotpot_ss_train.csv")))
        train_path = os.path.join(data_dir, "hotpot_ss_train.csv")
        # train_path = os.path.join(data_dir, "hotpot_ss_small.csv")
        return self.create_examples(
            pandas.read_csv(train_path), set_type='train')

    def get_dev_examples(self, data_dir):
        dev_path = os.path.join(data_dir, "hotpot_ss_dev.csv")
        # dev_path = os.path.join(data_dir, "hotpot_ss_small.csv")
        return self.create_examples(
            pandas.read_csv(dev_path), set_type='dev')

    def get_labels(self):
        return [False, True]

    def create_examples(self, df, set_type):
        examples = []
        for (i, row) in df.iterrows():
            guid = "%s-%s" % (set_type, i)
            text_a = row['question']
            text_b = '{} {}'.format(row['context'], row['title'])
            label = row['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # Feature ids
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a ) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Mask and Paddings
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        if ex_index < 5 and verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate():
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(input_ids, input_mask, segment_ids, label_ids)
    # Run prediction for full data
    sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=args.eval_batch_size)

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(dataloader, desc="Evaluation"):
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()
        label_ids = label_ids.cuda()

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids)

        predictions.append(logits)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy}

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        # writer.write("%s = %s\n" % (key, str(result[key])))

    logger.info("***** Writting Predictions ******")
    logits0 = np.concatenate(predictions, axis=0)[:, 0]
    logits1 = np.concatenate(predictions, axis=0)[:, 1]
    ground_truth = [fea.label_id for fea in features]
    score = pandas.DataFrame({'logits0': logits0, 'logits1': logits1, 'label': ground_truth})
    score.to_csv('pred_score.csv')
    return score, eval_loss, eval_accuracy


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)


def get_selected_paras(data, pred_score, output_path):

    logits = np.array([pred_score['logits0'], pred_score['logits1']]).transpose()
    pred_score['prob'] = softmax(logits)[:, 1]

    Paragraphs = dict()
    cur_ptr = 0

    for case in tqdm(data):
        key = case['_id']
        tem_ptr = cur_ptr
        all_paras = []
        selected_paras = []
        while cur_ptr < tem_ptr + len(case['context']):
            score = pred_score.ix[cur_ptr, 'prob']
            all_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            if score >= 0.05:  # 0.05
                selected_paras.append((score, case['context'][cur_ptr - tem_ptr]))
            cur_ptr += 1
        sorted_all_paras = sorted(all_paras, key=lambda x: x[0], reverse=True)
        sorted_selected_paras = sorted(selected_paras, key=lambda x: x[0], reverse=True)
        Paragraphs[key] = [p[1] for p in sorted_selected_paras]
        while len(Paragraphs[key]) < 3:
            if len(Paragraphs[key]) == len(all_paras):
                break
            Paragraphs[key].append(sorted_all_paras[len(Paragraphs[key])][1])

    Selected_paras_num = [len(Paragraphs[key]) for key in Paragraphs]
    print("Selected Paras Num:", Counter(Selected_paras_num))

    json.dump(Paragraphs, open(output_path, 'w'))


def get_dataframe(file_path):
    source_data = json.load(open(file_path, 'r'))
    sentence_pair_list = []
    for case in source_data:
        for para in case['context']:
            pair_dict = dict()
            pair_dict['label'] = 0
            pair_dict['title'] = para[0]
            pair_dict['context'] = " ".join(para[1])
            pair_dict['question'] = case['question']
            sentence_pair_list.append(pair_dict)

    return source_data, pandas.DataFrame(sentence_pair_list)


if __name__ == "__main__":
    args = set_args()

    # Load a trained model that you have fine-tuned
    model_state_dict = torch.load(args.ckpt_path)
    model = BertForSequenceClassification.from_pretrained(args.bert_model, state_dict=model_state_dict)
    model.cuda()
    model = torch.nn.DataParallel(model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    processor = DataProcessor()
    label_list = processor.get_labels()
    source_data, dataframe = get_dataframe(args.input_path)

    examples = processor.create_examples(dataframe, 'test')
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, verbose=True)

    score, _, _ = evaluate()
    get_selected_paras(source_data, score, args.output_path)

