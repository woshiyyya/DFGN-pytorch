import sys
import ujson as json
import re
import string
from collections import Counter
import pickle
from os import listdir
from os.path import isfile, join

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['pr'] += prec
    metrics['re'] += recall
    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_pr'] += prec
    metrics['sp_re'] += recall
    return em, prec, recall


def eval(prediction_file, gold_file):
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'pr': 0, 're': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_pr': 0, 'sp_re': 0,
        'jt_em': 0, 'jt_f1': 0, 'jt_pr': 0, 'jt_re': 0}
    for dp in gold:
        cur_id = dp['_id']
        can_eval_joint = True
        if cur_id not in prediction['answer']:
            # print('missing answer {}'.format(cur_id))
            can_eval_joint = False
        else:
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        if cur_id not in prediction['sp']:
            # print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['jt_em'] += joint_em
            metrics['jt_f1'] += joint_f1
            metrics['jt_pr'] += joint_prec
            metrics['jt_re'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    # print(metrics)
    return metrics


if __name__ == '__main__':
    onlyfiles = sorted([join(sys.argv[1], f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f)) and f.startswith('pred')], key=lambda x: int(x.strip('.json').split('_')[-1]))
    metrics = []
    for f in onlyfiles:
        try:
            metrics.append(eval(f, sys.argv[2]))
        except KeyError as e:
            print(e)


    keys = ['em', 'f1', 'pr', 're', 'sp_em', 'sp_f1', 'sp_pr', 'sp_re', 'jt_em', 'jt_f1', 'jt_pr', 'jt_re']
    print('\t' + '\t'.join(keys))

    temp = 'ep%02d\t' + '\t'.join(['%.4f']*12)
    # print(temp)
    best_iter = -1
    best_em = -1
    for i, me in enumerate(metrics):
        if me['em'] > best_em:
            best_em = me['em']
            best_iter = i
        print(temp % tuple([i] + list(map(lambda x: me[x], keys))))

    print('best_iter = %d' % best_iter)
    print(temp % tuple([best_iter] + list(map(lambda x: metrics[best_iter][x], keys))))
