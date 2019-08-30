import pickle
import torch
import json
import collections
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from torch import nn
import numpy as np
import string
import re
import os
import shutil


def load_settings(args, setting_fn):
    if setting_fn is None:
        return
    with open(setting_fn, 'r') as f:
        settings = json.load(f)
        for k, v in settings.items():
            if k not in ['ckpt_id', 'sp_threshold', 'name']:
                args.__dict__[k] = v


def get_weights(size, gain=1.414):
    weights = nn.Parameter(torch.zeros(size=size))
    nn.init.xavier_uniform_(weights, gain=gain)
    return weights


def get_bias(size):
    bias = nn.Parameter(torch.zeros(size=size))
    return bias


def get_act(act):
    if act.startswith('lrelu'):
        return nn.LeakyReLU(float(act.split(':')[1]))
    elif act == 'relu':
        return nn.ReLU()
    else:
        raise NotImplementedError


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


def report_tensor(tensor, name, dim=-1, verbose=False):
    print('{}: shape={}, mean={}, std={}, min={}, max={}'.
          format(name, tensor.shape, torch.mean(tensor), torch.std(tensor), torch.min(tensor), torch.max(tensor)))

    if verbose and len(tensor.shape) > 1:
        matrix = tensor.view(tensor.shape[0], -1)
        # if dim is None:
        #     check_dim = -1 if len(tensor.shape) < 3 else tuple(range(1-len(tensor.shape), 0))
        # else:
        #     check_dim = dim
        print('details: mean={},\n\t\tstd={},\n\t\tmin={},\n\t\tmax={}'.
              format(torch.mean(matrix, dim=dim), torch.std(matrix, dim=dim),
                     torch.min(matrix, dim=dim), torch.max(matrix, dim=dim)))


def encode(bert_model, batch, encoder_gpus, dest_gpu):
    doc_ids, doc_mask = batch['context_idxs'], batch['context_mask']
    query_ids, query_mask = batch['query_idxs'], batch['query_mask']
    doc_ids = doc_ids.cuda(encoder_gpus[0])
    doc_mask = doc_mask.cuda(encoder_gpus[0])
    query_ids = query_ids.cuda(encoder_gpus[0])
    query_mask = query_mask.cuda(encoder_gpus[0])

    all_doc_encoder_layers, _ = bert_model(doc_ids, token_type_ids=None, attention_mask=doc_mask)
    all_query_encoder_layers, _ = bert_model(query_ids, token_type_ids=None, attention_mask=query_mask)
    doc_encoding = all_doc_encoder_layers[-1].detach().to('cuda:{}'.format(dest_gpu))
    query_encoding = all_query_encoder_layers[-1].detach().to('cuda:{}'.format(dest_gpu))

    return doc_encoding, query_encoding


def load_data(args, debug=True):
    print("Loading data...")
    data = {}
    data['dev_example'] = pickle.load(open(args.dev_example_file, 'rb'))
    data['dev_feature'] = pickle.load(open(args.dev_feature_file, 'rb'))
    data['dev_graph'] = pickle.load(open(args.dev_graph_file, 'rb'))
    # data['dev_entity_type'] = pickle.load(open(args.dev_entity_type_file, 'rb'))
    if debug:
        return data
    # train_example = pickle.load(open(args.train_example_file, 'rb'))
    data['train_feature'] = pickle.load(open(args.train_feature_file, 'rb'))
    data['train_graph'] = pickle.load(open(args.train_graph_file, 'rb'))
    # data['train_entity_type'] = pickle.load(open(args.train_entity_type_file, 'rb'))
    return data


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def convert_to_tokens(example, features, ids, y1, y2, q_type):
    answer_dict = dict()
    for i, qid in enumerate(ids):
        answer_text = ''
        if q_type[i] == 0:
            doc_tokens = features[qid].doc_tokens
            tok_tokens = doc_tokens[y1[i]: y2[i] + 1]
            tok_to_orig_map = features[qid].token_to_orig_map
            if y2[i] < len(tok_to_orig_map):
                orig_doc_start = tok_to_orig_map[y1[i]]
                orig_doc_end = tok_to_orig_map[y2[i]]
                orig_tokens = example[qid].doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens).strip('[,.;]')

                final_text = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=False)
                answer_text = final_text
        elif q_type[i] == 1:
            answer_text = 'yes'
        elif q_type[i] == 2:
            answer_text = 'no'
        answer_dict[qid] = answer_text
    return answer_dict


def direct_predict(examples, features, pred_file):
    answer_dict = dict()
    sp_dict = dict()
    ids = list(examples.keys())
    for i, qid in enumerate(ids):
        answer_text = ''
        feature = features[qid]
        example = examples[qid]
        q_type = feature.ans_type
        y1, y2 = feature.start_position, feature.end_position
        if q_type == 0:
            doc_tokens = feature.doc_tokens
            tok_tokens = doc_tokens[y1: y2 + 1]
            tok_to_orig_map = feature.token_to_orig_map
            if y2 < len(tok_to_orig_map):
                orig_doc_start = tok_to_orig_map[y1]
                orig_doc_end = tok_to_orig_map[y2]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens).strip('[,.;]')

                final_text = get_final_text(tok_text, orig_text, do_lower_case=False, verbose_logging=False)
                answer_text = final_text
        elif q_type == 1:
            answer_text = 'yes'
        elif q_type == 2:
            answer_text = 'no'
        answer_dict[qid] = answer_text

        cur_sp = []
        for sent_id in feature.sup_fact_ids:
            cur_sp.append(example.sent_names[sent_id])
        sp_dict[qid] = cur_sp

    final_pred = {'answer': answer_dict, 'sp': sp_dict}
    json.dump(final_pred, open(pred_file, 'w'))


def _same_para(ent1, ent2, para_span):
    ent1_para = None
    ent2_para = None
    for span in para_span:
        if ent1[0] >= span[0] and ent1[1] <= span[1] and ent1_para is None:
            ent1_para = span[2]
        if ent2[0] >= span[0] and ent2[1] <= span[1] and ent2_para is None:
            ent2_para = span[2]

    if ent1_para is None or ent2_para is None:
        return False
    return ent1_para == ent2_para


# TODO other sim metric
# from difflib import SequenceMatcher

ENTITY_TYPES = {
    None: -1,
    'PERSON': 0,
    'LOCATION': 0,
    'ORGANIZATION': 0,
    'DATE': 1,
    'DURATION': 2,
    'NUMBER': 3,
    'ORDINAL': 3,
    'MONEY': 3,
    'PERCENT': 3,
    'TIME': 4,
    'SET': 5,
}


LIST = set()
QAS_ID = None


# TODO
def _same_ent(a, b, a_type=None, b_type=None, same_sent=False, same_para=False):
    if len(a) > len(b):
        a, b = b, a
        a_type, b_type = (b_type, a_type)
    if a.lower() == b.lower():
        return 1

    a_type = ENTITY_TYPES[a_type]
    b_type = ENTITY_TYPES[b_type]

    if a_type == -1 and b_type == -1:

        def lower_and_clean(_s):
            def remove_special_char(_s):
                _s.replace('&#34;', '"').replace('&quot;', '"').replace('&#38;', '&').replace('&amp;', '&'). \
                    replace('&#60;', '<').replace('&lt;', '<').replace('&#62;', '>').replace('&gt;', '>'). \
                    replace('-', ' ').replace(',', '').replace('&', '').replace('\'', '')
                _s = ' '.join(_s.split())
                return _s

            def remove_prep_article(_s):
                _st = [x for x in _s.split() if
                       x.lower() not in {"about", "beside", "near", "to", "above", "between", "of", "towards", "across",
                                         "beyond", "off", "under", "after", "by", "on", "underneath", "against",
                                         "despite", "onto", "unlike", "along", "down", "opposite", "until", "among",
                                         "during", "out", "up", "around", "except", "outside", "upon", "as", "for",
                                         "over", "via", "at", "from", "past", "with", "before", "in", "round", "within",
                                         "behind", "inside", "since", "without", "below", "into", "than", "beneath",
                                         "like", "through", "a", "an", "the", "un", "une", "des", "le", "la", "les",
                                         "l'", "du", "de", "à", "après", "avant", "avec", "chez", "contre", "dans",
                                         "de", "depuis", "derrière", "devant", "en", "entre", "envers", "environ",
                                         "par", "pendant", "pour", "sans", "sauf", "selon", "sous", "sur", "vers",
                                         "ante", "bajo", "con", "contra", "de", "desde", "detrás", "en", "entre",
                                         "hacia", "hasta", "para", "por", "según", "sin", "sobre", "tras", "el", "la",
                                         "los", "las", "un", "una", "unos", "unas", "lo", }]
                return ' '.join(_st)

            _s = remove_special_char(_s)
            _s = remove_prep_article(_s)

            def split_name_abbr(_s):
                _st = []
                _na = []
                _suf = []
                for token in _s.split():
                    if '.' in token:
                        if len(token) == 2 and token[1] == '.' and token[0] in set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                            _na.append(token[0])
                        else:
                            _st.append(token.replace('.', ''))
                    else:
                        if token in {'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII',
                                     'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'}:
                            _suf.append(token)
                        else:
                            _st.append(token)
                return _st, _na, _suf

            _st, _na, _suf = split_name_abbr(_s)

            return _s, _s.lower(), [x.lower() for x in _st], [x.lower() for x in _na], [x.lower() for x in _suf]

        a_clean, a_lower, a_other_tokens, a_name_abbr, a_name_suffix = lower_and_clean(a)
        b_clean, b_lower, b_other_tokens, b_name_abbr, b_name_suffix = lower_and_clean(b)

        if a_lower == b_lower:
            LIST.add('~~' + a + '##' + b)
            return 1

        if a_lower == '' or b_lower == '':
            return 0

        a_lower_tokens = set(a_lower.split())
        b_lower_tokens = set(b_lower.split())

        if a_lower_tokens == b_lower_tokens:
            LIST.add('!!' + a + '##' + b)
            return 1

        if len(a_name_suffix) > 0 and len(b_name_suffix) > 0:
            suffix_match = set(a_name_suffix) == set(b_name_suffix)
        elif len(a_name_suffix) == 0 and len(b_name_suffix) == 0:
            suffix_match = True
        else:
            suffix_match = False

        if not suffix_match:
            return 0

        if len(a_name_abbr) > 0 and len(b_name_abbr) > 0:
            a_other_tokens = set(a_other_tokens)
            b_other_tokens = set(b_other_tokens)
            a_name_abbr = set(a_name_abbr)
            b_name_abbr = set(b_name_abbr)

            if a_other_tokens == b_other_tokens:
                if len(a_name_abbr) > len(b_name_abbr):
                    less_name, more_name = b_name_abbr, a_name_abbr
                else:
                    less_name, more_name = a_name_abbr, b_name_abbr
                flag = False
                while len(less_name):
                    _a = less_name.pop()
                    flag = False
                    for _b in more_name:
                        if _b.startswith(_a):
                            more_name.remove(_b)
                            flag = True
                            break
                    if not flag:
                        return 0
                if flag:
                    LIST.add('@@' + a + '##' + b)
                    return 1
        elif len(a_name_abbr) == 0 and len(b_name_abbr) == 0:
            if set(a_other_tokens) == set(b_other_tokens):
                LIST.add('##' + a + '##' + b)
                return 1
        else:
            if len(a_name_abbr) == 0:
                c_other_tokens = a_other_tokens
                d_other_tokens, d_name_abbr = b_other_tokens, b_name_abbr
            else:
                c_other_tokens = b_other_tokens
                d_other_tokens, d_name_abbr = a_other_tokens, a_name_abbr

            if len(d_other_tokens) >= len(c_other_tokens):
                if ' '.join(c_other_tokens) == ' '.join(d_other_tokens[-len(c_other_tokens):]):
                    LIST.add('$$' + a + '##' + b)
                    return 1

            if set(d_other_tokens).issubset(set(c_other_tokens)):
                less_name = set(d_name_abbr)
                more_name = set(c_other_tokens) - set(d_other_tokens)
                flag = False
                while len(less_name):
                    _a = less_name.pop()
                    flag = False
                    for _b in more_name:
                        if _b.startswith(_a):
                            more_name.remove(_b)
                            flag = True
                            break
                    if not flag:
                        return 0
                if flag:
                    LIST.add('%%' + a + '##' + b)
                    return 1

        if same_para:
            if a_clean == a_clean.upper():
                c_abbr = a_clean
                d_abbr = ''.join([x[0] for x in b_clean.split()])
            elif b_clean == b_clean.upper():
                c_abbr = b_clean
                d_abbr = ''.join([x[0] for x in a_clean.split()])
            else:
                return 0

            if c_abbr == d_abbr:
                LIST.add('^^' + a + '##' + b)
                return 1

        return 0
    else:
        pass


def sent_mapping(sent_spans, entity_spans):
    if len(sent_spans) == 0:
        return {}, {}
    sent_ent_dict = {i: [] for i in range(len(sent_spans))}
    ent_sent_dict = {i: -1 for i in range(len(entity_spans))}
    si = 0
    for i in range(len(entity_spans)):
        ss, se = sent_spans[si]
        es, ee, _, _ = entity_spans[i]

        while es > se and si + 1 < len(sent_spans):
            si += 1
            ss, se = sent_spans[si]

        if ee < ss:
            continue

        if es >= ss and ee <= se:
            sent_ent_dict[si].append(i)
            ent_sent_dict[i] = si

    return sent_ent_dict, ent_sent_dict


def para_mapping(para_spans, entity_spans):
    if len(para_spans) == 0:
        return {}, {}

    para_ent_dict = {i: [] for i in range(len(para_spans))}
    ent_para_dict = {i: -1 for i in range(len(entity_spans))}
    pi = 0
    for i in range(len(entity_spans)):
        ps, pe, _ = para_spans[pi]
        es, ee, _, _ = entity_spans[i]

        while es > pe and pi + 1 < len(para_spans):
            pi += 1
            ps, pe, _ = para_spans[pi]

        if ee < ps:
            continue

        if es >= ps and ee <= pe:
            para_ent_dict[pi].append(i)
            ent_para_dict[i] = pi

    return para_ent_dict, ent_para_dict


def get_title_entities(para_spans, entity_spans):
    pi = 0
    para_entity_ids = [[] for _ in range(len(para_spans))]
    title_entity_ids = [[] for _ in range(len(para_spans))]
    for i in range(len(entity_spans)):
        ps, pe, pn = para_spans[pi]
        es, ee, en, _ = entity_spans[i]

        if es > pe:
            pi += 1
            ps, pe, pn = para_spans[pi]

        if es >= ps and ee <= pe:
            para_entity_ids[pi].append(i)
            if _same_ent(pn, en) > 0.8:
                title_entity_ids[pi].append(i)

    return para_entity_ids, title_entity_ids


def create_entity_graph(case, max_entity_num, para_limit, graph_type, self_loop, single_entity, relational=False, debug=False):
    # print('\n\n' + case.qas_id)
    # TODO check the graph
    global QAS_ID
    QAS_ID = case.qas_id

    if self_loop:
        adj = np.eye(max_entity_num, dtype=np.float32)
    else:
        adj = np.zeros((max_entity_num, max_entity_num), dtype=np.float32)

    if graph_type.startswith('win'):
        assert False
        window_threshold = 40
        entities = case.entity_spans
        para_spans = case.para_spans

        for i, ent1 in enumerate(entities):
            if i == max_entity_num:
                break
            for j, ent2 in enumerate(entities):
                if j == max_entity_num:
                    break
                if (ent1[2] == ent2[2]) or (ent1[2] in ent2[2]) or (ent2[2] in ent1[2]):
                    adj[i][j] = 1
                if _same_para(ent1, ent2, para_spans) and abs(ent1[0] - ent2[0]) <= window_threshold:
                    adj[i][j] = 1
        answer_entities = np.zeros(max_entity_num, dtype=np.float32)
        for i, ent in enumerate(entities):
            if normalize_answer(ent[2]) == normalize_answer(case.answer):
                answer_entities[i] = 1

    elif graph_type.startswith('sent') and single_entity:
        entities = case.entity_spans
        sent_spans = case.sent_spans
        para_spans = case.para_spans

        if debug:
            print(len(para_spans), len(sent_spans), len(entities))

        for i in range(len(entities)-1, -1, -1):
            es, ed, _, _ = entities[i]
            if es >= para_limit:
                del entities[i]

        sent_ent_dict, ent_sent_dict = sent_mapping(sent_spans, entities)

        para_ent_dict, ent_para_dict = para_mapping(para_spans, entities)

        ent_parent = list(range(len(entities)))

        def find(_x):
            if ent_parent[_x] == _x:
                return _x
            return find(ent_parent[_x])

        def union(_x, _y):
            _xp = find(_x)
            _yp = find(_y)
            if _xp == _yp:
                return
            if _xp < _yp:
                ent_parent[_yp] = _xp
            else:
                ent_parent[_xp] = _yp

        same_pairs = []
        sim_pairs = []
        for i in range(len(entities) - 1):
            _, _, a, _ = entities[i]
            for j in range(i + 1, len(entities)):
                _, _, b, _ = entities[j]
                same_sent = ent_sent_dict[i] == ent_sent_dict[j]
                same_para = ent_para_dict[i] == ent_para_dict[j]
                score = _same_ent(a, b, same_sent=same_sent, same_para=same_para)
                if score == 1:
                    union(i, j)
                    same_pairs.append([i, j])
                elif score > 0:
                    sim_pairs.append([i, j])

        em_entities = {}
        for i in range(len(entities)):
            _id = find(i)
            if _id not in em_entities:
                em_entities[_id] = [i]
            else:
                em_entities[_id].append(i)

        ent_id_dict = {}
        id_ent_dict = {}
        for i, k in enumerate(sorted(em_entities.keys())):
            id_ent_dict[i] = em_entities[k]
            for v in em_entities[k]:
                ent_id_dict[v] = i

        if debug:
            print(sent_ent_dict)
            print(ent_sent_dict)
            print(para_ent_dict)
            print(ent_para_dict)
            print(id_ent_dict)
            print(ent_id_dict)
            for k, v in id_ent_dict.items():
                print('uid = {}'.format(k))
                print([entities[x][2] for x in v])

        if len(id_ent_dict) > max_entity_num:
            truncated_eids = set()
            for uid in sorted(id_ent_dict.keys()):
                if uid >= max_entity_num:
                    eids = id_ent_dict[uid]
                    truncated_eids |= set(eids)
                    del id_ent_dict[uid]
            for eid in truncated_eids:
                del ent_id_dict[eid], ent_sent_dict[eid], ent_para_dict[eid]
            for k in sent_ent_dict.keys():
                sent_ent_dict[k] = [x for x in sent_ent_dict[k] if x not in truncated_eids]
            for k in para_ent_dict.keys():
                para_ent_dict[k] = [x for x in para_ent_dict[k] if x not in truncated_eids]

        # for i, j in same_pairs:
        #     if ent_id_dict[i] != ent_id_dict[j]:
        #         print(QAS_ID + '!!' + entities[i][2] + '##' + entities[j][2])
        #         assert False

        # for k, vs in id_ent_dict.items():
        #     print('{}\t<{}>'.format(k, '><'.join([entities[v][2] for v in vs])))

        for k, vs in sent_ent_dict.items():
            for i in range(len(vs)-1):
                _i = ent_id_dict[vs[i]]
                for j in range(i+1, len(vs)):
                    _j = ent_id_dict[vs[j]]
                    adj[_i, _j] = adj[_j, _i] = k+1 if debug else 1

        for i, para in enumerate(para_spans):
            _, _, pn = para
            uids = set()
            for ent in para_ent_dict[i]:
                uids.add(ent_id_dict[ent])
            matched_uids = set()
            if debug:
                print('para {}, name = {}'.format(i, pn))
                print('para entities', [entities[x][2] for x in para_ent_dict[i]])
            for uid in uids:
                for eid in id_ent_dict[uid]:
                    if _same_ent(entities[eid][2], pn, same_para=True) > 0:
                        matched_uids.add(uid)
                        if debug:
                            print('matched', entities[eid][2])
                        break
            for uid in uids:
                for muid in matched_uids:
                    if uid != muid:
                        adj[uid, muid] = adj[muid, uid] = -i-1 if debug else 1

        for i, j in sim_pairs:
            if i not in ent_id_dict or j not in ent_id_dict:
                continue
            _i = ent_id_dict[i]
            _j = ent_id_dict[j]
            if _i != _j:
                adj[_i, _j] = adj[_j, _i] = 1

        entity_mapping = np.zeros((max_entity_num, para_limit), dtype=np.float32)
        for eid, uid in ent_id_dict.items():
            es, ed, _, _ = entities[eid]
            entity_mapping[uid, es:ed+1] = 1

        start_entities = np.zeros(max_entity_num, dtype=np.float32)
        for qe in case.query_entities:
            for uid, eids in id_ent_dict.items():
                for eid in eids:
                    if _same_ent(qe, entities[eid][2], same_para=True):
                        start_entities[uid] = 1
        answer_entities = np.zeros(max_entity_num, dtype=np.float32)
        for i, ent in enumerate(entities):
            if normalize_answer(ent[2]) == normalize_answer(case.answer):
                answer_entities[i] = 1

        entity_length = len(id_ent_dict)

        if debug:
            for row in adj:
                print(('%3d' * len(row)) % tuple(row))
    elif graph_type.startswith('sent') and not single_entity:
        entities = case.entity_spans[:max_entity_num]
        sent_spans = case.sent_spans
        para_spans = case.para_spans

        for i in range(len(entities)-1, -1, -1):
            es, ed, _, _ = entities[i]
            if es >= para_limit:
                del entities[i]

        sent_ent_dict, ent_sent_dict = sent_mapping(sent_spans, entities)
        para_ent_dict, ent_para_dict = para_mapping(para_spans, entities)

        same_pairs = []
        for i in range(len(entities)-1):
            _, _, a, _ = entities[i]
            for j in range(i+1, len(entities)):
                _, _, b, _ = entities[j]
                same_sent = ent_sent_dict[i] == ent_sent_dict[j]
                same_para = ent_para_dict[i] == ent_para_dict[j]
                if _same_ent(a, b, same_sent=same_sent, same_para=same_para) > 0:
                    same_pairs.append([i, j])

        for k, vs in sent_ent_dict.items():
            for i in range(len(vs)-1):
                _i = vs[i]
                for j in range(i+1, len(vs)):
                    _j = vs[j]
                    # adj[_i, _j] = adj[_j, _i] = k+1 if debug else 1
                    adj[_i, _j] = adj[_j, _i] = 1

        if debug:
            print('\n\n' + QAS_ID)
        # TODO cannot match para name
        for i, para in enumerate(para_spans):
            _, _, pn = para
            uids = set(para_ent_dict[i])
            matched_uids = set()
            for uid in uids:
                if _same_ent(entities[uid][2], pn, same_para=True) > 0:
                    matched_uids.add(uid)
            if debug:
                print('para {}, name = {}'.format(i, pn))
                print('para entities', [entities[x][2] for x in uids])
                print('matched entities', [entities[x][2] for x in matched_uids])
            for uid in matched_uids:
                for eid in para_ent_dict[i]:
                    if uid != eid:
                        # if adj[uid, eid] == 0:
                            adj[uid, eid] = adj[eid, uid] = 1 if not relational else 2

        for i, j in same_pairs:
            # if adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1 if not relational else 2

        entity_mapping = np.zeros((max_entity_num, para_limit), dtype=np.float32)
        for i, ent in enumerate(entities):
            es, ed, _, _ = ent
            entity_mapping[i, es:ed+1] = 1

        start_entities = np.zeros(max_entity_num, dtype=np.float32)
        for qe in case.query_entities:
            for i, ent in enumerate(entities):
                if _same_ent(qe, ent[2], same_para=True):
                    start_entities[i] = 1

        answer_entities = np.zeros(max_entity_num, dtype=np.float32)
        for i, ent in enumerate(entities):
            if normalize_answer(ent[2]) == normalize_answer(case.answer):
                answer_entities[i] = 1
        ent_id_dict = {}
        id_ent_dict = {}
        entity_length = len(entities)

        if debug:
            for row in adj:
                print(('%2d' * len(row)) % tuple(row))
    else:
        raise NotImplementedError

    return {'adj': adj,
            # 'sent_ent_dict': sent_ent_dict,
            # 'ent_sent_dict': ent_sent_dict,
            # 'para_ent_dict': para_ent_dict,
            # 'ent_para_dict': ent_para_dict,
            # 'ent_id_dict': ent_id_dict,
            # 'id_ent_dict': id_ent_dict,
            'entity_mapping': entity_mapping,
            'entity_length': entity_length,
            'start_entities': start_entities,
            'entity_label': answer_entities,}


def bfs_step(start_vec, graph):
    """
    :param start_vec:   [E]
    :param graph:       [E x E]
    :return: next_vec:  [E]
    """
    next_vec = torch.matmul(start_vec.float().unsqueeze(0), graph)
    next_vec = (next_vec > 0).long().squeeze(0)
    return next_vec


def save_scripts(path):
    scripts_to_save = ['model/layers.py', 'model/GFN.py', 'sync_train.py', 'utils.py', 'data_iterator.py',
                       'Feature_extraction/Get_paras.py', 'Feature_extraction/text_to_tok.py', 'test.py']
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        if not os.path.exists(path):
            os.mkdir(path)
        for script in scripts_to_save:
            dst_file = os.path.join(path, os.path.basename(script))
            shutil.copyfile(script, dst_file)
