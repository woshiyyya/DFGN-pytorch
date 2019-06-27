import torch
import numpy as np
from numpy.random import shuffle
from utils import create_entity_graph, bfs_step, normalize_answer
from random import choice

IGNORE_INDEX = -100


class DataIteratorPack(object):
    def __init__(self, features, example_dict, graph_dict, bsz, device, sent_limit, entity_limit, n_layers,
                 entity_type_dict=None, sequential=False,):
        self.bsz = bsz
        self.device = device
        self.features = features
        self.example_dict = example_dict
        self.graph_dict = graph_dict
        self.entity_type_dict = entity_type_dict
        self.sequential = sequential
        self.sent_limit = sent_limit
        self.entity_limit = entity_limit
        self.example_ptr = 0
        self.n_layers = n_layers
        if not sequential:
            shuffle(self.features)

    def refresh(self):
        self.example_ptr = 0
        if not self.sequential:
            shuffle(self.features)

    def empty(self):
        return self.example_ptr >= len(self.features)

    def __len__(self):
        return int(np.ceil(len(self.features)/self.bsz))

    def __iter__(self):
        # BERT input
        context_idxs = torch.LongTensor(self.bsz, 512)
        context_mask = torch.LongTensor(self.bsz, 512)
        segment_idxs = torch.LongTensor(self.bsz, 512)

        # Graph and Mappings
        entity_graphs = torch.Tensor(self.bsz, self.entity_limit, self.entity_limit).cuda(self.device)
        query_mapping = torch.Tensor(self.bsz, 512).cuda(self.device)
        start_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        end_mapping = torch.Tensor(self.bsz, self.sent_limit, 512).cuda(self.device)
        all_mapping = torch.Tensor(self.bsz, 512, self.sent_limit).cuda(self.device)
        entity_mapping = torch.Tensor(self.bsz, self.entity_limit, 512).cuda(self.device)

        # Label tensor
        y1 = torch.LongTensor(self.bsz).cuda(self.device)
        y2 = torch.LongTensor(self.bsz).cuda(self.device)
        q_type = torch.LongTensor(self.bsz).cuda(self.device)
        is_support = torch.FloatTensor(self.bsz, self.sent_limit).cuda(self.device)

        start_mask = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
        start_mask_weight = torch.FloatTensor(self.bsz, self.entity_limit).cuda(self.device)
        bfs_mask = torch.FloatTensor(self.bsz, self.n_layers, self.entity_limit).cuda(self.device)
        entity_label = torch.LongTensor(self.bsz).cuda(self.device)

        while True:
            if self.example_ptr >= len(self.features):
                break
            start_id = self.example_ptr
            cur_bsz = min(self.bsz, len(self.features) - start_id)
            cur_batch = self.features[start_id: start_id + cur_bsz]
            cur_batch.sort(key=lambda x: sum(x.doc_input_mask), reverse=True)

            ids = []
            max_sent_cnt = 0
            max_entity_cnt = 0
            for mapping in [start_mapping, end_mapping, all_mapping, entity_mapping, query_mapping]:
                mapping.zero_()
            entity_label.fill_(IGNORE_INDEX)
            is_support.fill_(IGNORE_INDEX)
            # is_support.fill_(0)  # BCE

            for i in range(len(cur_batch)):
                case = cur_batch[i]
                context_idxs[i].copy_(torch.Tensor(case.doc_input_ids))
                context_mask[i].copy_(torch.Tensor(case.doc_input_mask))
                segment_idxs[i].copy_(torch.Tensor(case.doc_segment_ids))

                for j in range(case.sent_spans[0][0] - 1):
                    query_mapping[i, j] = 1

                tem_graph = self.graph_dict[case.qas_id]
                adj = torch.from_numpy(tem_graph['adj'])
                start_entities = torch.from_numpy(tem_graph['start_entities'])
                entity_graphs[i] = adj
                for l in range(self.n_layers):
                    bfs_mask[i][l].copy_(start_entities)
                    start_entities = bfs_step(start_entities, adj)

                start_mask[i].copy_(start_entities)
                start_mask_weight[i, :tem_graph['entity_length']] = start_entities.byte().any().float()
                # if case.ans_type == 0:
                #     num_ans = len(case.start_position)
                #     if num_ans == 0:
                #         y1[i] = y2[i] = 0
                #     else:
                #         ans_id = choice(range(num_ans))
                #         start_position = case.start_position[ans_id]
                #         end_position = case.end_position[ans_id]
                #         if end_position < 512:
                #             y1[i] = start_position
                #             y2[i] = end_position
                #         else:
                #             y1[i] = y2[i] = 0
                #     q_type[i] = 0
                if case.ans_type == 0:
                    if len(case.end_position) == 0:
                        y1[i] = y2[i] = 0
                    elif case.end_position[0] < 512:
                        y1[i] = case.start_position[0]
                        y2[i] = case.end_position[0]
                    else:
                        y1[i] = y2[i] = 0
                    q_type[i] = 0
                elif case.ans_type == 1:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 1
                elif case.ans_type == 2:
                    y1[i] = IGNORE_INDEX
                    y2[i] = IGNORE_INDEX
                    q_type[i] = 2

                for j, sent_span in enumerate(case.sent_spans[:self.sent_limit]):
                    is_sp_flag = j in case.sup_fact_ids
                    start, end = sent_span
                    if start < end:
                        is_support[i, j] = int(is_sp_flag)
                        all_mapping[i, start:end+1, j] = 1
                        start_mapping[i, j, start] = 1
                        end_mapping[i, j, end] = 1

                ids.append(case.qas_id)
                answer = self.example_dict[case.qas_id].orig_answer_text
                for j, entity_span in enumerate(case.entity_spans[:self.entity_limit]):
                    _, _, ent, _ = entity_span
                    if normalize_answer(ent) == normalize_answer(answer):
                        entity_label[i] = j
                        break

                entity_mapping[i] = torch.from_numpy(tem_graph['entity_mapping'])
                max_sent_cnt = max(max_sent_cnt, len(case.sent_spans))
                max_entity_cnt = max(max_entity_cnt, tem_graph['entity_length'])

            entity_lengths = (entity_mapping[:cur_bsz] > 0).float().sum(dim=2)
            entity_lengths = torch.where((entity_lengths > 0), entity_lengths, torch.ones_like(entity_lengths))
            entity_mask = (entity_mapping > 0).any(2).float()

            input_lengths = (context_mask[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            self.example_ptr += cur_bsz

            yield {
                'context_idxs': context_idxs[:cur_bsz, :max_c_len].contiguous(),
                'context_mask': context_mask[:cur_bsz, :max_c_len].contiguous(),
                'segment_idxs': segment_idxs[:cur_bsz, :max_c_len].contiguous(),
                'query_mapping': query_mapping[:cur_bsz, :max_c_len].contiguous(),
                'entity_graphs': entity_graphs[:cur_bsz, :max_entity_cnt, :max_entity_cnt].contiguous(),
                'context_lens': input_lengths.to(self.device),
                'y1': y1[:cur_bsz],
                'y2': y2[:cur_bsz],
                'ids': ids,
                'q_type': q_type[:cur_bsz],
                'is_support': is_support[:cur_bsz, :max_sent_cnt].contiguous(),
                'start_mapping': start_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'end_mapping': end_mapping[:cur_bsz, :max_sent_cnt, :max_c_len],
                'all_mapping': all_mapping[:cur_bsz, :max_c_len, :max_sent_cnt],
                'entity_mapping': entity_mapping[:cur_bsz, :max_entity_cnt, :max_c_len],
                'entity_lens': entity_lengths[:cur_bsz, :max_entity_cnt],
                'entity_mask': entity_mask[:cur_bsz, :max_entity_cnt],
                'entity_label': entity_label[:cur_bsz],
                'start_mask': start_mask[:cur_bsz, :max_entity_cnt].contiguous(),
                'start_mask_weight': start_mask_weight[:cur_bsz, :max_entity_cnt].contiguous(),
                'bfs_mask': bfs_mask[:cur_bsz, :, :max_entity_cnt]
            }
